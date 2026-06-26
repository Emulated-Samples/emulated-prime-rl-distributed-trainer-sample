/**
 * Setup Orchestrator
 *
 * Provisions base infrastructure for prime-rl:
 * 1. EKS Cluster + Nodegroups
 * 2. GPU Operator
 *
 * Storage and Helm deployment are the solver's responsibility.
 * All functions are idempotent - safe to run multiple times.
 */

import { Logger } from "@hyperfocal/env-base";

import { GPU_NODE_COUNT } from "../config.js";

import {
  createCluster,
  createGpuNodegroup,
  updateKubeconfig,
  waitForNodes,
} from "./cluster.js";

import {
  installGpuOperator,
  waitForGpuOperator,
  verifyGpuAllocatable,
} from "./gpu.js";

/**
 * Run all setup steps in sequence.
 * Each step is idempotent - skips if already done.
 */
export async function runSetup(logger: Logger): Promise<void> {
  logger.info("Starting environment setup");

  logger.info("\n[Phase 1/2] EKS cluster");
  await createCluster(logger);
  await createGpuNodegroup(logger);
  await updateKubeconfig(logger);
  await waitForNodes(logger, 1 + GPU_NODE_COUNT); // 1 CPU + GPU nodes

  logger.info("\n[Phase 2/2] GPU operator");
  await installGpuOperator(logger);
  await waitForGpuOperator(logger);
  await verifyGpuAllocatable(logger);

  logger.info("Setup complete — cluster ready for deployment");
}
