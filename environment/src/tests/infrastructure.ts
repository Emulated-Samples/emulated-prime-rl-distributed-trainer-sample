import {
  Logger,
  SimpleTest,
  SimpleTestResult,
  executeWithExitCode,
} from "@hyperfocal/env-base";
import * as fs from "fs";
import * as path from "path";

import {
  WORKSPACE_PATH,
  AWS_REGION,
  CLUSTER_NAME,
  NODE_READY_TIMEOUT,
  POD_READY_TIMEOUT,
  GPU_OPERATOR_TIMEOUT,
  GPU_NODE_COUNT,
} from "../config.js";

import {
  getEnvWithAws,
  pollUntil,
  KubernetesNodeList,
  KubernetesPodList,
} from "../helpers.js";

function preconditionPassed(logger: Logger, message: string): SimpleTestResult {
  logger.info(message);
  return {
    success: true,
    skipped: true,
    error: "Environment precondition passed; excluded from rollout score",
  };
}

/**
 * Verify prerequisites are in place.
 */
export function createPrerequisiteTests(): SimpleTest[] {
  return [
    {
      id: "aws-credentials",
      name: "AWS Credentials Valid",
      description: "Verify AWS credentials are configured and working",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();
        const result = await executeWithExitCode("aws sts get-caller-identity", {
          env,
        });

        if (!result.success) {
          return {
            success: false,
            error: `AWS credentials invalid: ${result.output}`,
            errored: true,
          };
        }

        return preconditionPassed(logger, "AWS identity confirmed");
      },
    },
    {
      id: "cli-tools-installed",
      name: "CLI Tools Installed",
      description: "Verify kubectl, helm, and eksctl are available",
      run: async (logger: Logger) => {
        const tools = ["kubectl", "helm", "eksctl"];
        const missing: string[] = [];

        for (const tool of tools) {
          const result = await executeWithExitCode(`which ${tool}`, { silent: true });
          if (!result.success) {
            missing.push(tool);
          }
        }

        if (missing.length > 0) {
          return {
            success: false,
            error: `Missing CLI tools: ${missing.join(", ")}`,
            errored: true,
          };
        }

        return preconditionPassed(logger, "All CLI tools installed");
      },
    },
    {
      id: "workspace-exists",
      name: "Workspace Contains Prime-RL",
      description: "Verify prime-rl Helm chart exists in workspace",
      run: async (logger: Logger) => {
        const helmChartPath = path.join(WORKSPACE_PATH, "k8s", "prime-rl");
        if (!fs.existsSync(helmChartPath)) {
          return {
            success: false,
            error: `Helm chart not found at ${helmChartPath}`,
            errored: true,
          };
        }

        const valuesPath = path.join(helmChartPath, "values.yaml");
        if (!fs.existsSync(valuesPath)) {
          return {
            success: false,
            error: `values.yaml not found at ${valuesPath}`,
            errored: true,
          };
        }

        return preconditionPassed(logger, "Prime-RL Helm chart found");
      },
    },
  ];
}

/**
 * Verify EKS cluster is set up correctly.
 */
export function createEksClusterTests(): SimpleTest[] {
  return [
    {
      id: "cluster-active",
      name: "EKS Cluster Active",
      description: "Verify EKS cluster exists and is ACTIVE",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();

        const result = await executeWithExitCode(
          `aws eks describe-cluster --name ${CLUSTER_NAME} --region ${AWS_REGION} --query 'cluster.status' --output text`,
          { env }
        );

        if (!result.success) {
          return {
            success: false,
            error: `EKS cluster ${CLUSTER_NAME} not found or not accessible`,
            errored: true,
          };
        }

        const status = result.output.trim();
        if (status !== "ACTIVE") {
          return {
            success: false,
            error: `EKS cluster status is ${status}, expected ACTIVE`,
            errored: true,
          };
        }

        return preconditionPassed(logger, `EKS cluster ${CLUSTER_NAME} is ACTIVE`);
      },
    },
    {
      id: "kubeconfig-valid",
      name: "Kubeconfig Valid",
      description: "Verify kubectl can connect to the cluster",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();

        // Update kubeconfig first
        await executeWithExitCode(
          `aws eks update-kubeconfig --name ${CLUSTER_NAME} --region ${AWS_REGION}`,
          { env, silent: true }
        );

        const result = await executeWithExitCode("kubectl cluster-info", { env });

        if (!result.success) {
          return {
            success: false,
            error: `Cannot connect to cluster: ${result.output}`,
            errored: true,
          };
        }

        return preconditionPassed(logger, "kubectl can connect to cluster");
      },
    },
    {
      id: "cpu-nodegroup-active",
      name: "CPU Nodegroup Active",
      description: "Verify CPU nodegroup exists and is ACTIVE",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();

        const result = await executeWithExitCode(
          `aws eks describe-nodegroup --cluster-name ${CLUSTER_NAME} --nodegroup-name cpu-nodes --region ${AWS_REGION} --query 'nodegroup.status' --output text`,
          { env }
        );

        if (!result.success) {
          return {
            success: false,
            error: "CPU nodegroup not found",
            errored: true,
          };
        }

        const status = result.output.trim();
        if (status !== "ACTIVE") {
          return {
            success: false,
            error: `CPU nodegroup status is ${status}, expected ACTIVE`,
            errored: true,
          };
        }

        return preconditionPassed(logger, "CPU nodegroup is ACTIVE");
      },
    },
    {
      id: "gpu-nodegroup-active",
      name: "GPU Nodegroup Active",
      description: "Verify GPU nodegroup exists and is ACTIVE",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();

        const result = await executeWithExitCode(
          `aws eks describe-nodegroup --cluster-name ${CLUSTER_NAME} --nodegroup-name gpu-nodes --region ${AWS_REGION} --query 'nodegroup.status' --output text`,
          { env }
        );

        if (!result.success) {
          return {
            success: false,
            error: "GPU nodegroup not found",
            errored: true,
          };
        }

        const status = result.output.trim();
        if (status !== "ACTIVE") {
          return {
            success: false,
            error: `GPU nodegroup status is ${status}, expected ACTIVE`,
            errored: true,
          };
        }

        return preconditionPassed(logger, "GPU nodegroup is ACTIVE");
      },
    },
    {
      id: "all-nodes-ready",
      name: "All Nodes Ready",
      description: "Verify all 3 nodes (1 CPU + 2 GPU) are Ready in Kubernetes",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();

        const result = await pollUntil(
          async () => {
            const nodesResult = await executeWithExitCode(
              "kubectl get nodes -o json",
              { env, silent: true }
            );

            if (!nodesResult.success) {
              return { done: false, message: "Cannot get nodes" };
            }

            try {
              const nodes: KubernetesNodeList = JSON.parse(nodesResult.output);
              const readyNodes = nodes.items.filter((node) =>
                node.status.conditions.some(
                  (c) => c.type === "Ready" && c.status === "True"
                )
              );

              const total = nodes.items.length;
              const ready = readyNodes.length;

              const expectedNodes = 1 + GPU_NODE_COUNT; // 1 CPU + GPU nodes
              if (total >= expectedNodes && ready >= expectedNodes) {
                return { done: true };
              }

              return {
                done: false,
                message: `Nodes: ${ready}/${total} ready (need ${expectedNodes})`,
              };
            } catch {
              return { done: false, message: "Failed to parse node status" };
            }
          },
          {
            timeout: NODE_READY_TIMEOUT,
            interval: 10000,
            description: "Checking nodes",
            logger,
          }
        );

        if (!result.success) {
          return { success: false, error: result.error, errored: true };
        }

        // Log node info
        const nodesOutput = await executeWithExitCode(
          "kubectl get nodes -o wide",
          { env }
        );
        logger.info(`Cluster nodes:\n${nodesOutput.output}`);

        return preconditionPassed(logger, "Cluster nodes are ready");
      },
    },
  ];
}

/**
 * Verify GPU Operator is installed and GPUs are allocatable.
 */
export function createGpuOperatorTests(): SimpleTest[] {
  return [
    {
      id: "gpu-operator-installed",
      name: "GPU Operator Installed",
      description: "Verify NVIDIA GPU Operator Helm release exists",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();

        const result = await executeWithExitCode(
          "helm list -n gpu-operator -o json",
          { env }
        );

        if (!result.success || !result.output.includes("gpu-operator")) {
          return {
            success: false,
            error: "GPU Operator Helm release not found",
            errored: true,
          };
        }

        return preconditionPassed(logger, "GPU Operator Helm release exists");
      },
    },
    {
      id: "gpu-operator-pods-running",
      name: "GPU Operator Pods Running",
      description: "Verify GPU Operator pods are running",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();

        const result = await pollUntil(
          async () => {
            const podsResult = await executeWithExitCode(
              "kubectl get pods -n gpu-operator -o json",
              { env, silent: true }
            );

            if (!podsResult.success) {
              return { done: false, message: "Cannot get GPU operator pods" };
            }

            try {
              const pods: KubernetesPodList = JSON.parse(podsResult.output);
              const runningPods = pods.items.filter(
                (pod) =>
                  pod.status.phase === "Running" ||
                  pod.status.phase === "Succeeded"
              );

              // Need multiple pods: operator, device plugin, driver daemonsets
              if (runningPods.length >= 5) {
                return { done: true };
              }

              return {
                done: false,
                message: `GPU Operator pods: ${runningPods.length} running`,
              };
            } catch {
              return { done: false, message: "Failed to parse pod status" };
            }
          },
          {
            timeout: GPU_OPERATOR_TIMEOUT,
            interval: 15000,
            description: "Checking GPU Operator pods",
            logger,
          }
        );

        if (!result.success) {
          return { success: false, error: result.error, errored: true };
        }

        return preconditionPassed(logger, "GPU Operator pods are running");
      },
    },
    {
      id: "gpus-allocatable",
      name: "GPUs Allocatable",
      description: "Verify nvidia.com/gpu resources are allocatable on GPU nodes",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();

        const result = await pollUntil(
          async () => {
            const nodesResult = await executeWithExitCode(
              'kubectl get nodes -l nvidia.com/gpu.present=true -o json',
              { env, silent: true }
            );

            if (!nodesResult.success) {
              return { done: false, message: "Cannot get GPU nodes" };
            }

            try {
              const nodes: KubernetesNodeList = JSON.parse(nodesResult.output);
              let gpuReadyCount = 0;
              const issues: string[] = [];

              for (const node of nodes.items) {
                const allocatable = node.status?.allocatable?.["nvidia.com/gpu"];
                if (allocatable && allocatable !== "0") {
                  gpuReadyCount++;
                } else {
                  issues.push(`${node.metadata.name}: 0 GPUs`);
                }
              }

              if (gpuReadyCount >= 2) {
                return { done: true };
              }

              // If some nodes have 0 GPUs, try restarting device plugin
              for (const node of nodes.items) {
                const allocatable = node.status?.allocatable?.["nvidia.com/gpu"];
                if (!allocatable || allocatable === "0") {
                  // Restart device plugin on this node
                  const podResult = await executeWithExitCode(
                    `kubectl get pods -n gpu-operator -l app=nvidia-device-plugin-daemonset --field-selector spec.nodeName=${node.metadata.name} -o name`,
                    { env, silent: true }
                  );

                  if (podResult.success && podResult.output.trim()) {
                    logger.warn(
                      `Restarting device plugin on ${node.metadata.name}`
                    );
                    await executeWithExitCode(
                      `kubectl delete ${podResult.output.trim()} -n gpu-operator`,
                      { env }
                    );
                  }
                }
              }

              return {
                done: false,
                message: `GPU nodes ready: ${gpuReadyCount}/2. ${issues.join(", ")}`,
              };
            } catch {
              return { done: false, message: "Failed to parse GPU status" };
            }
          },
          {
            timeout: 5 * 60 * 1000,
            interval: 30000,
            description: "Checking GPU allocatability",
            logger,
          }
        );

        if (!result.success) {
          return { success: false, error: result.error, errored: true };
        }

        const gpuInfo = await executeWithExitCode(
          "kubectl describe nodes -l nvidia.com/gpu.present=true | grep -A5 'Allocatable:'",
          { env }
        );
        logger.info(`GPU resources:\n${gpuInfo.output}`);

        return preconditionPassed(logger, "GPU resources are allocatable");
      },
    },
  ];
}
