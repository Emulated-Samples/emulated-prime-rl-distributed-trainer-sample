import { Logger, execute, executeWithExitCode } from "@hyperfocal/env-base";
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function getEnvWithAws(): Record<string, string> {
  const envPath = path.join(__dirname, "..", ".env");
  const envContent = fs.readFileSync(envPath, "utf-8");
  const env: Record<string, string> = { ...process.env } as Record<string, string>;

  for (const line of envContent.split("\n")) {
    if (line.startsWith("#") || !line.includes("=")) continue;
    const [key, ...valueParts] = line.split("=");
    const value = valueParts.join("=").trim();
    env[key.trim()] = value;
  }

  return env;
}

export async function commandExists(cmd: string): Promise<boolean> {
  const result = await executeWithExitCode(`which ${cmd}`, { silent: true });
  return result.success;
}

export async function ensureKubectl(logger: Logger): Promise<void> {
  if (await commandExists("kubectl")) {
    logger.info("kubectl already installed");
    return;
  }

  logger.info("Installing kubectl");
  await execute(
    `curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
     chmod +x kubectl && \
     sudo mv kubectl /usr/local/bin/`
  );
  logger.info("kubectl installed successfully");
}

export async function ensureHelm(logger: Logger): Promise<void> {
  if (await commandExists("helm")) {
    logger.info("helm already installed");
    return;
  }

  logger.info("Installing helm");
  await execute(
    `curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash`
  );
  logger.info("helm installed successfully");
}

export async function ensureEksctl(logger: Logger): Promise<void> {
  if (await commandExists("eksctl")) {
    logger.info("eksctl already installed");
    return;
  }

  logger.info("Installing eksctl");
  await execute(
    `curl -sLO "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_Linux_amd64.tar.gz" && \
     tar -xzf eksctl_Linux_amd64.tar.gz && \
     sudo mv eksctl /usr/local/bin/ && \
     rm eksctl_Linux_amd64.tar.gz`
  );
  logger.info("eksctl installed successfully");
}

export interface PollCheckResult {
  done: boolean;
  message?: string;
}

export interface PollOptions {
  timeout: number;
  interval: number;
  description: string;
  logger: Logger;
}

export interface PollResult {
  success: boolean;
  error?: string;
}

export async function pollUntil(
  check: () => Promise<PollCheckResult>,
  options: PollOptions
): Promise<PollResult> {
  const startTime = Date.now();
  const { timeout, interval, description, logger } = options;
  let lastMessage: string | undefined;

  while (Date.now() - startTime < timeout) {
    try {
      const result = await check();
      if (result.done) {
        return { success: true };
      }
      if (result.message) {
        lastMessage = result.message;
        logger.info(`${description}: ${result.message}`);
      }
    } catch (e) {
      lastMessage = String(e);
      logger.debug(`Poll check failed: ${e}`);
    }
    await sleep(interval);
  }

  const detail = lastMessage ? `; last observed state: ${lastMessage}` : "";
  return { success: false, error: `Did not complete ${description} before deadline${detail}` };
}

export interface KubernetesCondition {
  type: string;
  status: string;
}

export interface KubernetesNode {
  metadata: {
    name: string;
    labels?: Record<string, string>;
  };
  status: {
    conditions: KubernetesCondition[];
    capacity?: Record<string, string>;
    allocatable?: Record<string, string>;
  };
}

export interface KubernetesNodeList {
  items: KubernetesNode[];
}

export interface KubernetesPod {
  metadata: {
    name: string;
  };
  status: {
    phase: string;
  };
}

export interface KubernetesPodList {
  items: KubernetesPod[];
}

export interface KubernetesPVC {
  spec?: {
    storageClassName?: string;
    volumeName?: string;
  };
  status?: {
    phase?: string;
  };
}

// ---------------------------------------------------------------------------
// Failure attribution
//
// When kubectl-driven tests fail because an upstream resource is missing or
// misconfigured, many tests cascade with near-identical errors. The classifier
// below recognises the common cascade signatures and swaps the per-test error
// for a short, model-attributed message ("Agent didn't deploy ..."), without
// any cross-references to other tests.
//
// The first time we see each category in a rollout, we also write a single
// diagnostic info event with the full original error so debug context survives.
// Subsequent occurrences emit only the short message; the original kubectl
// noise is not repeated.
//
// Module-level state lives for the duration of one orchestrator process
// invocation (one test phase = one fresh node process), so the map naturally
// resets between rollouts.
// ---------------------------------------------------------------------------

type FailureCategory =
  | "trainer-pod-missing"
  | "configmap-missing"
  | "helm-release-missing"
  | "cluster-api-unreachable"
  | "image-pull-failed"
  | "sourceoverride-disabled";

const CATEGORY_MESSAGES: Record<FailureCategory, string> = {
  "trainer-pod-missing": "Agent didn't deploy the trainer pod",
  "configmap-missing": "Agent didn't create the trainer source ConfigMap",
  "helm-release-missing": "Agent didn't install the prime-rl Helm release",
  "cluster-api-unreachable": "Kubernetes API was unreachable",
  "image-pull-failed": "Trainer container image could not be pulled",
  "sourceoverride-disabled":
    "Agent ran helm install without enabling trainer.sourceOverride; trainer pod is running image-baked source instead of workspace source",
};

// Patterns matched directly against raw error output. Order matters; the first
// match wins. Keep these high-precision (avoid false positives that would
// over-attribute to a single root cause).
const PRIMARY_PATTERNS: Array<[RegExp, FailureCategory]> = [
  [/configmap "prime-rl-trainer-source" not found/i, "configmap-missing"],
  [/pods?\s+"[^"]*-trainer-0"\s+not found/i, "trainer-pod-missing"],
  [/error from server \(notfound\):\s+pods?\s+"[^"]+"/i, "trainer-pod-missing"],
  [/release: not found|Helm release [^ ]+ not found/i, "helm-release-missing"],
  [/imagepullbackoff|errimagepull|manifest unknown/i, "image-pull-failed"],
  [/unable to connect to the server|connection refused/i, "cluster-api-unreachable"],
];

// Patterns that only attribute when the corresponding root category has
// already been registered in this rollout. Avoids over-attribution (e.g. a
// hash mismatch could legitimately mean a stale ConfigMap rather than a
// missing sourceOverride; we only re-attribute when the root cause was
// detected upstream).
const SECONDARY_PATTERNS: Array<[RegExp, FailureCategory]> = [
  [/does not match workspace source/i, "sourceoverride-disabled"],
  [/statefulset(?:s\.apps)?\s+"[^"]+-trainer"\s+not found/i, "helm-release-missing"],
];

const seenFailureCategories = new Map<FailureCategory, { firstTestId: string }>();

function recordFirstSighting(
  category: FailureCategory,
  testId: string,
  logger: Logger,
  rawError?: string
): void {
  if (seenFailureCategories.has(category)) return;
  seenFailureCategories.set(category, { firstTestId: testId });
  const tail = rawError ? `\nOriginal kubectl error:\n${rawError}` : "";
  logger.info(
    `[diagnostic] ${CATEGORY_MESSAGES[category]} (first detected in test "${testId}").${tail}`
  );
}

/**
 * Classify a raw kubectl/test error and return a short, model-attributed
 * message when the error matches a known cascade signature. The first
 * sighting of each category in a rollout is logged once with the full raw
 * error for debug context; subsequent occurrences return only the short
 * attribution and the raw error is not repeated.
 *
 * If no pattern matches, the original error is returned unchanged.
 */
export function classifyAndAttributeError(
  rawError: string,
  testId: string,
  logger: Logger
): string {
  for (const [pattern, category] of PRIMARY_PATTERNS) {
    if (!pattern.test(rawError)) continue;
    recordFirstSighting(category, testId, logger, rawError);
    return CATEGORY_MESSAGES[category];
  }

  for (const [pattern, category] of SECONDARY_PATTERNS) {
    if (!pattern.test(rawError)) continue;
    if (!seenFailureCategories.has(category)) continue;
    return CATEGORY_MESSAGES[category];
  }

  return rawError;
}

/**
 * Register a failure category that a test detected directly (not via parsing
 * a kubectl error). Used by tests that inspect StatefulSet config, ConfigMap
 * contents, etc. and want to seed an attribution that downstream cascade
 * tests can pick up. Returns the short attribution message so the caller can
 * use it as the test's error.
 */
export function registerFailureCategory(
  category: FailureCategory,
  testId: string,
  logger: Logger
): string {
  recordFirstSighting(category, testId, logger);
  return CATEGORY_MESSAGES[category];
}
