/**
 * EKS Cluster Setup
 *
 * Creates and configures the EKS cluster with CPU and GPU nodegroups.
 * Uses eksctl for cluster management and polls AWS directly for status.
 */

import { Logger, executeWithExitCode } from "@hyperfocal/env-base";
import * as fs from "fs";

import {
  AWS_REGION,
  CLUSTER_NAME,
  EKS_CREATION_TIMEOUT,
  GPU_NODEGROUP_TIMEOUT,
  NODEGROUP_POLL_INTERVAL,
  USE_SPOT_INSTANCES,
} from "../config.js";
import { getEnvWithAws, pollUntil, KubernetesNodeList } from "../helpers.js";

/**
 * Check if EKS cluster exists and is active.
 */
export async function clusterExists(): Promise<boolean> {
  const env = getEnvWithAws();
  const result = await executeWithExitCode(
    `aws eks describe-cluster --name ${CLUSTER_NAME} --region ${AWS_REGION} --query 'cluster.status' --output text`,
    { env, silent: true }
  );
  return result.success && result.output.trim() === "ACTIVE";
}

/**
 * Check if a nodegroup exists.
 */
export async function nodegroupExists(name: string): Promise<boolean> {
  const env = getEnvWithAws();
  const result = await executeWithExitCode(
    `aws eks describe-nodegroup --cluster-name ${CLUSTER_NAME} --nodegroup-name ${name} --region ${AWS_REGION} --query 'nodegroup.status' --output text`,
    { env, silent: true }
  );
  return result.success && result.output.trim() === "ACTIVE";
}

/**
 * Create EKS cluster with CPU nodegroup only.
 * GPU nodegroup is added separately to handle longer provisioning times.
 */
export async function createCluster(logger: Logger): Promise<void> {
  const env = getEnvWithAws();

  if (await clusterExists()) {
    logger.info("EKS cluster already exists, skipping creation");
    return;
  }

  logger.info("Creating EKS cluster with CPU nodes (takes ~15 min)");

  const clusterConfig = `
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: ${CLUSTER_NAME}
  region: ${AWS_REGION}

managedNodeGroups:
  - name: cpu-nodes
    instanceType: t3.medium
    desiredCapacity: 1
    minSize: 1
    maxSize: 2
    labels:
      role: cpu
    tags:
      nodegroup-role: orchestrator
`;

  const configPath = "/tmp/eksctl-config.yaml";
  fs.writeFileSync(configPath, clusterConfig);

  const result = await executeWithExitCode(`eksctl create cluster -f ${configPath}`, {
    env,
    timeout: EKS_CREATION_TIMEOUT,
  });

  if (!result.success) {
    throw new Error(`Failed to create EKS cluster: ${result.output.slice(-2000)}`);
  }

  logger.info("EKS cluster created successfully");
}

/**
 * Create GPU nodegroup and wait for it to become active.
 * Uses AWS API polling for more reliable status tracking.
 */
export async function createGpuNodegroup(logger: Logger): Promise<void> {
  const env = getEnvWithAws();

  if (await nodegroupExists("gpu-nodes")) {
    logger.info("GPU nodegroup already exists, skipping creation");
    return;
  }

  logger.info(
    `Creating GPU nodegroup with ${USE_SPOT_INSTANCES ? "spot" : "on-demand"} instances...`
  );

  const gpuConfig = `
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: ${CLUSTER_NAME}
  region: ${AWS_REGION}

managedNodeGroups:
  - name: gpu-nodes
    instanceType: g5.xlarge
    desiredCapacity: 2
    minSize: 0
    maxSize: 4
    ${USE_SPOT_INSTANCES ? "spot: true" : "# Using on-demand instances"}
    labels:
      role: gpu
      nvidia.com/gpu.present: "true"
    tags:
      nodegroup-role: gpu-training
`;

  const configPath = "/tmp/eksctl-gpu-nodegroup.yaml";
  fs.writeFileSync(configPath, gpuConfig);

  // Start nodegroup creation
  const createResult = await executeWithExitCode(
    `eksctl create nodegroup -f ${configPath}`,
    { env, timeout: GPU_NODEGROUP_TIMEOUT }
  );

  // Check for fatal errors
  if (!createResult.success) {
    const output = createResult.output;
    if (output.includes("MaxSpotInstanceCountExceeded")) {
      throw new Error("Spot instance quota exceeded. Set USE_SPOT_INSTANCES=false");
    }
    if (output.includes("VcpuLimitExceeded")) {
      throw new Error("vCPU limit exceeded. Request quota increase for G instances");
    }
    if (output.includes("InsufficientInstanceCapacity")) {
      throw new Error(`Insufficient capacity for g5.xlarge in ${AWS_REGION}`);
    }
    // For other errors, check AWS status below
    logger.warn("eksctl returned error, checking AWS for actual status");
  }

  // Poll AWS for nodegroup status
  logger.info("Polling AWS for GPU nodegroup status");

  const pollResult = await pollUntil(
    async () => {
      const statusResult = await executeWithExitCode(
        `aws eks describe-nodegroup --cluster-name ${CLUSTER_NAME} --nodegroup-name gpu-nodes --region ${AWS_REGION} --query 'nodegroup.status' --output text`,
        { env, silent: true }
      );

      if (!statusResult.success) {
        if (statusResult.output.includes("ResourceNotFoundException")) {
          return { done: false, message: "Nodegroup not found yet" };
        }
        return { done: false, message: `AWS error: ${statusResult.output.slice(0, 200)}` };
      }

      const status = statusResult.output.trim();

      if (status === "ACTIVE") {
        return { done: true };
      }

      if (status === "CREATE_FAILED" || status === "DEGRADED") {
        throw new Error(`GPU nodegroup failed with status: ${status}`);
      }

      return { done: false, message: `GPU nodegroup status: ${status}` };
    },
    {
      timeout: GPU_NODEGROUP_TIMEOUT,
      interval: NODEGROUP_POLL_INTERVAL,
      description: "GPU nodegroup creation",
      logger,
    }
  );

  if (!pollResult.success) {
    throw new Error(pollResult.error || "GPU nodegroup creation timed out");
  }

  logger.info("GPU nodegroup is ACTIVE");
}

/**
 * Update kubeconfig to access the cluster.
 */
export async function updateKubeconfig(logger: Logger): Promise<void> {
  const env = getEnvWithAws();

  const result = await executeWithExitCode(
    `aws eks update-kubeconfig --name ${CLUSTER_NAME} --region ${AWS_REGION}`,
    { env }
  );

  if (!result.success) {
    throw new Error(`Failed to update kubeconfig: ${result.output}`);
  }

  logger.info("Kubeconfig updated");
}

/**
 * Wait for all nodes to be ready.
 */
export async function waitForNodes(logger: Logger, expectedCount: number = 3): Promise<void> {
  const env = getEnvWithAws();

  const result = await pollUntil(
    async () => {
      const nodesResult = await executeWithExitCode("kubectl get nodes -o json", {
        env,
        silent: true,
      });

      if (!nodesResult.success) {
        return { done: false, message: "Cannot reach cluster" };
      }

      try {
        const nodes: KubernetesNodeList = JSON.parse(nodesResult.output);
        const readyNodes = nodes.items.filter((node) =>
          node.status.conditions.some(
            (c) => c.type === "Ready" && c.status === "True"
          )
        );

        if (readyNodes.length >= expectedCount) {
          return { done: true };
        }

        return {
          done: false,
          message: `Nodes ready: ${readyNodes.length}/${expectedCount}`,
        };
      } catch {
        return { done: false, message: "Failed to parse node status" };
      }
    },
    {
      timeout: 5 * 60 * 1000,
      interval: 15000,
      description: "Waiting for nodes",
      logger,
    }
  );

  if (!result.success) {
    throw new Error(result.error || "Nodes not ready");
  }

  logger.info(`All ${expectedCount} nodes are ready`);
}
