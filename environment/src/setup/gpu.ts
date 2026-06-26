/**
 * GPU Operator Setup
 *
 * Installs and configures the NVIDIA GPU Operator for Kubernetes.
 * This is a prerequisite for prime-rl which expects nvidia.com/gpu resources.
 */

import { Logger, executeWithExitCode } from "@hyperfocal/env-base";

import { GPU_OPERATOR_TIMEOUT, GPU_NODE_COUNT } from "../config.js";
import { getEnvWithAws, pollUntil, sleep, KubernetesPodList, KubernetesNodeList } from "../helpers.js";

/**
 * Check if GPU Operator is already installed.
 */
export async function gpuOperatorInstalled(): Promise<boolean> {
  const env = getEnvWithAws();
  const result = await executeWithExitCode("helm list -n gpu-operator -o json", {
    env,
    silent: true,
  });
  return result.success && result.output.includes("gpu-operator");
}

/**
 * Install NVIDIA GPU Operator via Helm.
 */
export async function installGpuOperator(logger: Logger): Promise<void> {
  const env = getEnvWithAws();

  if (await gpuOperatorInstalled()) {
    logger.info("GPU Operator already installed, skipping");
    return;
  }

  logger.info("Adding NVIDIA Helm repository");
  await executeWithExitCode(
    "helm repo add nvidia https://helm.ngc.nvidia.com/nvidia && helm repo update",
    { env }
  );

  logger.info("Installing NVIDIA GPU Operator");
  const result = await executeWithExitCode(
    `helm install gpu-operator nvidia/gpu-operator \
      --namespace gpu-operator \
      --create-namespace \
      --set driver.enabled=true \
      --set toolkit.enabled=true \
      --wait \
      --timeout 10m`,
    { env, timeout: GPU_OPERATOR_TIMEOUT }
  );

  if (!result.success && !result.output.includes("already exists")) {
    throw new Error(`Failed to install GPU Operator: ${result.output}`);
  }

  logger.info("GPU Operator installed");
}

/**
 * Wait for GPU Operator pods to be running.
 */
export async function waitForGpuOperator(logger: Logger): Promise<void> {
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
          (pod) => pod.status.phase === "Running" || pod.status.phase === "Succeeded"
        );

        // Need at least operator + device plugin pods
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
      description: "Waiting for GPU Operator",
      logger,
    }
  );

  if (!result.success) {
    throw new Error(result.error || "GPU Operator not ready");
  }

  logger.info("GPU Operator pods are running");
}

/**
 * Verify GPUs are allocatable on all GPU nodes.
 * Restarts device plugin if needed.
 */
export async function verifyGpuAllocatable(logger: Logger): Promise<void> {
  const env = getEnvWithAws();

  logger.info("Verifying GPU allocatable on nodes");

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
        let allReady = true;
        let readyCount = 0;

        for (const node of nodes.items) {
          const gpuCount = node.status?.allocatable?.["nvidia.com/gpu"];

          if (!gpuCount || gpuCount === "0") {
            allReady = false;
            // Try to restart device plugin on this node
            logger.warn(`Node ${node.metadata.name} has 0 GPUs, restarting device plugin...`);
            
            const podResult = await executeWithExitCode(
              `kubectl get pods -n gpu-operator -l app=nvidia-device-plugin-daemonset --field-selector spec.nodeName=${node.metadata.name} -o name`,
              { env, silent: true }
            );

            if (podResult.success && podResult.output.trim()) {
              await executeWithExitCode(
                `kubectl delete ${podResult.output.trim()} -n gpu-operator`,
                { env }
              );
            }
          } else {
            readyCount++;
          }
        }

        if (allReady && readyCount >= GPU_NODE_COUNT) {
          return { done: true };
        }

        return {
          done: false,
          message: `GPU nodes with allocatable GPUs: ${readyCount}/${nodes.items.length}`,
        };
      } catch {
        return { done: false, message: "Failed to parse GPU status" };
      }
    },
    {
      timeout: 5 * 60 * 1000,
      interval: 30000,
      description: "Waiting for GPUs to be allocatable",
      logger,
    }
  );

  if (!result.success) {
    throw new Error(result.error || "GPUs not allocatable");
  }

  logger.info("All GPU nodes have allocatable GPUs");
}
