import { Logger, executeWithExitCode } from "@hyperfocal/env-base";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";

import {
  AWS_REGION,
  BASE_RUNTIME_IMAGE,
  EVAL_RUNTIME_ECR_REPOSITORY,
  EVAL_RUNTIME_IMAGE_TAG,
  WORKSPACE_PATH,
  getEvalRuntimeImageUri,
} from "../config.js";
import { getEnvWithAws } from "../helpers.js";

function shellQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`;
}

async function getAwsAccountId(): Promise<string> {
  const env = getEnvWithAws();
  if (env.AWS_ACCOUNT_ID?.trim()) {
    return env.AWS_ACCOUNT_ID.trim();
  }

  const result = await executeWithExitCode(
    `aws sts get-caller-identity --query Account --output text`,
    { env, silent: true, timeout: 30000 }
  );

  if (!result.success || !result.output.trim()) {
    throw new Error(`Failed to resolve AWS account ID for runtime image: ${result.output}`);
  }

  return result.output.trim();
}

async function ensureEcrRepository(logger: Logger): Promise<void> {
  const env = getEnvWithAws();
  const describe = await executeWithExitCode(
    `aws ecr describe-repositories --region ${AWS_REGION} --repository-names ${EVAL_RUNTIME_ECR_REPOSITORY}`,
    { env, silent: true, timeout: 30000 }
  );

  if (describe.success) {
    return;
  }

  const create = await executeWithExitCode(
    `aws ecr create-repository --region ${AWS_REGION} --repository-name ${EVAL_RUNTIME_ECR_REPOSITORY} --image-scanning-configuration scanOnPush=false`,
    { env, timeout: 60000 }
  );

  if (!create.success && !create.output.includes("RepositoryAlreadyExistsException")) {
    throw new Error(`Failed to create ECR repository ${EVAL_RUNTIME_ECR_REPOSITORY}: ${create.output}`);
  }

  logger.info(`[Setup] Created ECR repository ${EVAL_RUNTIME_ECR_REPOSITORY}`);
}

async function ecrImageExists(): Promise<boolean> {
  const env = getEnvWithAws();
  const result = await executeWithExitCode(
    `aws ecr describe-images --region ${AWS_REGION} --repository-name ${EVAL_RUNTIME_ECR_REPOSITORY} --image-ids imageTag=${EVAL_RUNTIME_IMAGE_TAG}`,
    { env, silent: true, timeout: 30000 }
  );
  return result.success;
}

function sanitizedDockerfile(): string {
  return `FROM ${BASE_RUNTIME_IMAGE}
USER root
COPY trainer_stub.py /tmp/trainer_stub.py
RUN set -eux; \\
    mkdir -p /app/src/prime_rl/trainer/rl; \\
    for file in loss.py data.py packer.py train.py; do \\
      cp /tmp/trainer_stub.py "/app/src/prime_rl/trainer/rl/$file"; \\
      chown appuser:appuser "/app/src/prime_rl/trainer/rl/$file"; \\
      chmod 0644 "/app/src/prime_rl/trainer/rl/$file"; \\
    done; \\
    find /app -path '*/prime_rl/trainer/rl/__pycache__' -type d -prune -exec rm -rf {} +; \\
    find /app -path '*/prime_rl/trainer/rl/*.pyc' -type f -delete; \\
    rm -f /tmp/trainer_stub.py
USER appuser
`;
}

function trainerStubSource(): string {
  return `"""Evaluation runtime stub.

The trainer implementation must be supplied from the rollout workspace through
the Helm chart's trainer.sourceOverride ConfigMap.
"""

raise RuntimeError(
    "This evaluation runtime image intentionally contains no trainer solution. "
    "Deploy workspace trainer source with trainer.sourceOverride."
)
`;
}

async function buildAndPushImage(logger: Logger, imageUri: string, accountId: string): Promise<void> {
  const env = getEnvWithAws();
  const registry = `${accountId}.dkr.ecr.${AWS_REGION}.amazonaws.com`;
  const buildDir = fs.mkdtempSync(path.join(os.tmpdir(), "prime-rl-runtime-image-"));
  fs.writeFileSync(path.join(buildDir, "Dockerfile"), sanitizedDockerfile());
  fs.writeFileSync(path.join(buildDir, "trainer_stub.py"), trainerStubSource());

  try {
    const login = await executeWithExitCode(
      `aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${registry}`,
      { env, timeout: 60000 }
    );
    if (!login.success) {
      throw new Error(`Failed to log in to ECR: ${login.output}`);
    }

    const build = await executeWithExitCode(
      `docker build --pull=false -t ${shellQuote(imageUri)} ${shellQuote(buildDir)}`,
      { env, timeout: 20 * 60 * 1000 }
    );
    if (!build.success) {
      throw new Error(`Failed to build sanitized runtime image: ${build.output.slice(-4000)}`);
    }

    const push = await executeWithExitCode(
      `docker push ${shellQuote(imageUri)}`,
      { env, timeout: 20 * 60 * 1000 }
    );
    if (!push.success) {
      throw new Error(`Failed to push sanitized runtime image: ${push.output.slice(-4000)}`);
    }
  } finally {
    fs.rmSync(buildDir, { recursive: true, force: true });
  }
}

function configureWorkspaceExample(imageUri: string): void {
  const [repository, tag] = imageUri.split(":");
  const valuesPath = path.join(WORKSPACE_PATH, "k8s", "prime-rl", "examples", "reverse-text.yaml");
  const content = fs.readFileSync(valuesPath, "utf-8");
  const next = content.replace(
    /image:\n(?:  .*\n)*?  tag: "[^"]+"/,
    `image:\n  repository: ${repository}\n  tag: "${tag}"`
  );

  if (next === content) {
    throw new Error(`Failed to configure sanitized runtime image in ${valuesPath}`);
  }

  fs.writeFileSync(valuesPath, next);
}

export async function prepareSanitizedRuntimeImage(logger: Logger): Promise<void> {
  const accountId = await getAwsAccountId();
  const imageUri = getEvalRuntimeImageUri(accountId);

  logger.info(`[Setup] Preparing sanitized Prime-RL runtime image: ${imageUri}`);
  await ensureEcrRepository(logger);

  if (await ecrImageExists()) {
    logger.info("[Setup] Sanitized runtime image already exists in ECR");
  } else {
    await buildAndPushImage(logger, imageUri, accountId);
    logger.info("[Setup] Built and pushed sanitized runtime image");
  }

  configureWorkspaceExample(imageUri);
  logger.info("[Setup] Configured workspace Helm example to use sanitized runtime image");
}
