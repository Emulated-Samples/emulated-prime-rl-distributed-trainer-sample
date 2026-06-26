export const WORKSPACE_PATH =
  process.env.WORKSPACE_PATH || "/hyperfocal/env/workspace";

export const AWS_REGION = process.env.AWS_REGION || "us-west-2";
export const CLUSTER_NAME = process.env.CLUSTER_NAME || "prime-rl-test";
export const RELEASE_NAME = "prime-rl-test";

export const BASE_RUNTIME_IMAGE = "primeintellect/prime-rl:commit-f00925c";
export const EVAL_RUNTIME_ECR_REPOSITORY = "prime-rl-eval-runtime";
export const EVAL_RUNTIME_IMAGE_TAG = "sanitized-commit-f00925c-v1";

export function getEvalRuntimeImageUri(accountId: string): string {
  return `${accountId}.dkr.ecr.${AWS_REGION}.amazonaws.com/${EVAL_RUNTIME_ECR_REPOSITORY}:${EVAL_RUNTIME_IMAGE_TAG}`;
}

// Set USE_SPOT_INSTANCES=true for ~60-70% cheaper GPU nodes (risk of interruption)
export const USE_SPOT_INSTANCES = process.env.USE_SPOT_INSTANCES === "true";

export const EKS_CREATION_TIMEOUT = 25 * 60 * 1000;
export const GPU_NODEGROUP_TIMEOUT = 35 * 60 * 1000;
export const NODE_READY_TIMEOUT = 20 * 60 * 1000;
export const POD_READY_TIMEOUT = 8 * 60 * 1000;
export const TRAINING_PROGRESS_TIMEOUT = 10 * 60 * 1000;
export const TRAINING_COMPLETION_TIMEOUT = 15 * 60 * 1000;
export const TRAINING_TIMEOUT = TRAINING_COMPLETION_TIMEOUT;
export const GPU_OPERATOR_TIMEOUT = 15 * 60 * 1000;
export const INFERENCE_STARTUP_TIMEOUT = 5 * 60 * 1000;
export const PVC_BIND_TIMEOUT = 3 * 60 * 1000;
export const CHECKPOINT_TIMEOUT = 2 * 60 * 1000;

export const VF_EVAL_TIMEOUT = 5 * 60 * 1000;
export const VF_EVAL_REWARD_THRESHOLD = 0.5;

export const CPU_INSTANCE_TYPE = "t3.medium";
export const GPU_INSTANCE_TYPE = "g5.xlarge";
export const GPU_NODE_COUNT = 2;

export const DEFAULT_POLL_INTERVAL = 15 * 1000;
export const NODEGROUP_POLL_INTERVAL = 30 * 1000;
