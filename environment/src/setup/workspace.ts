import { Logger, executeWithExitCode } from "@hyperfocal/env-base";
import { execSync } from "child_process";
import * as fs from "fs";
import * as path from "path";

import { AWS_REGION, WORKSPACE_PATH } from "../config.js";

const AGENT_USER = "hyperfocal-agent";
const PRESERVED_HYPERFOCAL_FILES = new Set(["credentials.env"]);
const REQUIRED_AWS_CREDENTIAL_KEYS = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"] as const;
const OPTIONAL_AWS_CREDENTIAL_KEYS = [
  "AWS_SESSION_TOKEN",
  "AWS_ACCOUNT_ID",
  "AWS_CREDENTIALS_EXPIRATION",
] as const;

function cleanupHyperfocalArtifacts(logger: Logger): void {
  const hyperfocalDir = path.join(WORKSPACE_PATH, ".hyperfocal");

  if (!fs.existsSync(hyperfocalDir)) {
    return;
  }

  for (const entry of fs.readdirSync(hyperfocalDir)) {
    if (PRESERVED_HYPERFOCAL_FILES.has(entry)) {
      continue;
    }

    fs.rmSync(path.join(hyperfocalDir, entry), { recursive: true, force: true });
    logger.info(`[Setup] Removed stale workspace artifact: .hyperfocal/${entry}`);
  }
}

function shellQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`;
}

function agentUserExists(): boolean {
  try {
    execSync(`id ${AGENT_USER}`, { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

function applyHyperfocalPermissions(hyperfocalDir: string, credentialsPath: string): void {
  if (agentUserExists()) {
    execSync(`chown -R ${AGENT_USER}:${AGENT_USER} ${shellQuote(hyperfocalDir)}`);
  }

  fs.chmodSync(hyperfocalDir, 0o770);
  fs.chmodSync(credentialsPath, 0o640);
}

async function resetWorkspaceGitState(logger: Logger): Promise<boolean> {
  const gitCheck = await executeWithExitCode(
    `git -C "${WORKSPACE_PATH}" rev-parse --is-inside-work-tree`,
    { silent: true }
  );

  if (!gitCheck.success || gitCheck.output.trim() !== "true") {
    return false;
  }

  const rootResult = await executeWithExitCode(
    `git -C "${WORKSPACE_PATH}" rev-parse --show-toplevel`,
    { silent: true }
  );

  if (!rootResult.success || !rootResult.output.trim()) {
    return false;
  }

  const gitRoot = rootResult.output.trim();
  const workspaceRelative = path.relative(gitRoot, WORKSPACE_PATH).replace(/\\/g, "/") || ".";
  if (workspaceRelative.startsWith("..") || path.isAbsolute(workspaceRelative)) {
    throw new Error(
      `Workspace path ${WORKSPACE_PATH} is outside git root ${gitRoot}; refusing to reset`
    );
  }

  const quotedRoot = shellQuote(gitRoot);
  const quotedWorkspace = shellQuote(workspaceRelative);
  const quotedCredentials = shellQuote(
    path.posix.join(workspaceRelative, ".hyperfocal", "credentials.env")
  );

  const resetResult = await executeWithExitCode(
    `git -C ${quotedRoot} restore --source=HEAD --staged --worktree -- ${quotedWorkspace}`,
    { silent: true, timeout: 120000 }
  );

  if (!resetResult.success) {
    throw new Error(`Failed to restore workspace from HEAD: ${resetResult.output}`);
  }

  const cleanResult = await executeWithExitCode(
    `git -C ${quotedRoot} clean -fdx -e ${quotedCredentials} -- ${quotedWorkspace}`,
    { silent: true, timeout: 120000 }
  );

  if (!cleanResult.success) {
    throw new Error(`Failed to clean workspace artifacts: ${cleanResult.output}`);
  }

  logger.info("[Setup] Workspace reset to HEAD and untracked artifacts were removed");
  return true;
}

function writeWorkspaceCredentials(logger: Logger): void {
  const missing = REQUIRED_AWS_CREDENTIAL_KEYS.filter((key) => {
    const value = process.env[key];
    return !value || value.trim().length === 0;
  });

  if (missing.length > 0) {
    throw new Error(
      `Missing AWS credential environment variables for workspace refresh: ${missing.join(", ")}`
    );
  }

  const hyperfocalDir = path.join(WORKSPACE_PATH, ".hyperfocal");
  const credentialsPath = path.join(hyperfocalDir, "credentials.env");
  const region = process.env.AWS_REGION?.trim() || AWS_REGION;
  const defaultRegion = process.env.AWS_DEFAULT_REGION?.trim() || region;
  const lines = [
    "# AWS Sandbox Credentials",
    "# Refreshed by setupProblem before each rollout",
    `AWS_ACCESS_KEY_ID=${shellQuote(process.env.AWS_ACCESS_KEY_ID!.trim())}`,
    `AWS_SECRET_ACCESS_KEY=${shellQuote(process.env.AWS_SECRET_ACCESS_KEY!.trim())}`,
    `AWS_REGION=${shellQuote(region)}`,
    `AWS_DEFAULT_REGION=${shellQuote(defaultRegion)}`,
  ];

  for (const key of OPTIONAL_AWS_CREDENTIAL_KEYS) {
    const value = process.env[key]?.trim();
    if (value) {
      lines.push(`${key}=${shellQuote(value)}`);
    }
  }

  fs.mkdirSync(hyperfocalDir, { recursive: true });
  fs.writeFileSync(credentialsPath, `${lines.join("\n")}\n`);
  applyHyperfocalPermissions(hyperfocalDir, credentialsPath);
  logger.info("[Setup] Refreshed workspace AWS credentials at .hyperfocal/credentials.env");
}

export async function prepareWorkspaceForRollout(logger: Logger): Promise<void> {
  if (!fs.existsSync(WORKSPACE_PATH)) {
    throw new Error(`Workspace not found at ${WORKSPACE_PATH}`);
  }

  logger.info("[Setup] Resetting workspace to a clean pre-rollout state...");

  const resetWithGit = await resetWorkspaceGitState(logger);
  cleanupHyperfocalArtifacts(logger);
  writeWorkspaceCredentials(logger);

  if (!resetWithGit) {
    logger.warn(
      "[Setup] Workspace is not a git checkout; only .hyperfocal artifacts were cleaned"
    );
  }
}
