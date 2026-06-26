import { Logger, executeWithExitCode } from "@hyperfocal/env-base";
import * as fs from "fs";

import { AWS_REGION, CLUSTER_NAME, RELEASE_NAME } from "../config.js";
import { getEnvWithAws, sleep } from "../helpers.js";
import { clusterExists } from "./cluster.js";

const TEMP_SETUP_FILES = [
  "/tmp/efs-id.txt",
  "/tmp/efs-ap-id.txt",
  "/tmp/efs-storage-class.yaml",
  "/tmp/efs-pv.yaml",
];

async function uninstallHelmRelease(
  logger: Logger,
  release: string,
  namespace?: string
): Promise<void> {
  const env = getEnvWithAws();
  const namespaceArgs = namespace ? ` -n ${namespace}` : "";
  const result = await executeWithExitCode(
    `helm uninstall ${release}${namespaceArgs}`,
    { env, timeout: 120000 }
  );

  if (result.success) {
    logger.info(`[Setup] Removed Helm release: ${release}`);
  }
}

async function removeEfsCsiDriver(logger: Logger): Promise<void> {
  const env = getEnvWithAws();

  await executeWithExitCode(
    `aws eks delete-addon --cluster-name ${CLUSTER_NAME} --addon-name aws-efs-csi-driver --region ${AWS_REGION}`,
    { env, timeout: 120000 }
  );
  await uninstallHelmRelease(logger, "aws-efs-csi-driver", "kube-system");
}

async function deleteKubernetesRolloutResources(logger: Logger): Promise<void> {
  const env = getEnvWithAws();

  await executeWithExitCode(
    `kubectl delete all -l app.kubernetes.io/instance=${RELEASE_NAME} --ignore-not-found --wait=false`,
    { env, timeout: 120000 }
  );
  await executeWithExitCode(
    `kubectl delete pod -l app.kubernetes.io/instance=${RELEASE_NAME} --ignore-not-found --force --grace-period=0 --wait=false`,
    { env, timeout: 60000 }
  );
  await executeWithExitCode(
    "kubectl delete pod debug-prime-rl --ignore-not-found --force --grace-period=0 --wait=false",
    { env, timeout: 60000 }
  );
  await executeWithExitCode(
    `kubectl delete pvc ${RELEASE_NAME}-shared-data --ignore-not-found`,
    { env, timeout: 60000 }
  );
  await executeWithExitCode(
    "kubectl delete pv prime-rl-efs-pv --ignore-not-found",
    { env, timeout: 60000 }
  );
  await executeWithExitCode(
    "kubectl delete storageclass efs-sc --ignore-not-found",
    { env, timeout: 60000 }
  );

  logger.info("[Setup] Cleared prior Kubernetes rollout resources");
}

async function deleteEfsResources(logger: Logger): Promise<void> {
  const env = getEnvWithAws();
  const efsResult = await executeWithExitCode(
    `aws efs describe-file-systems --region ${AWS_REGION} --query "FileSystems[?Tags[?Key=='cluster' && Value=='${CLUSTER_NAME}']].FileSystemId" --output text`,
    { env, silent: true }
  );

  if (!efsResult.success) {
    throw new Error(`Failed to query EFS filesystems: ${efsResult.output}`);
  }

  const efsIds = efsResult.output
    .trim()
    .split(/\s+/)
    .filter((value) => value && value !== "None");

  if (efsIds.length === 0) {
    logger.info("[Setup] No prior EFS rollout storage found");
    return;
  }

  for (const efsId of efsIds) {
    logger.info(`[Setup] Removing prior EFS filesystem: ${efsId}`);

    const apResult = await executeWithExitCode(
      `aws efs describe-access-points --file-system-id ${efsId} --region ${AWS_REGION} --query 'AccessPoints[*].AccessPointId' --output text`,
      { env, silent: true }
    );

    if (apResult.success) {
      for (const apId of apResult.output.trim().split(/\s+/).filter(Boolean)) {
        await executeWithExitCode(
          `aws efs delete-access-point --access-point-id ${apId} --region ${AWS_REGION}`,
          { env }
        );
      }
    }

    const mtResult = await executeWithExitCode(
      `aws efs describe-mount-targets --file-system-id ${efsId} --region ${AWS_REGION} --query 'MountTargets[*].MountTargetId' --output text`,
      { env, silent: true }
    );

    const mountTargetIds = mtResult.success
      ? mtResult.output.trim().split(/\s+/).filter(Boolean)
      : [];

    for (const mtId of mountTargetIds) {
      await executeWithExitCode(
        `aws efs delete-mount-target --mount-target-id ${mtId} --region ${AWS_REGION}`,
        { env }
      );
    }

    if (mountTargetIds.length > 0) {
      logger.info("[Setup] Waiting for EFS mount targets to delete...");
      await sleep(60000);
    }

    await executeWithExitCode(
      `aws efs delete-file-system --file-system-id ${efsId} --region ${AWS_REGION}`,
      { env }
    );
  }

  logger.info("[Setup] Removed prior EFS rollout storage");
}

async function deleteEfsSecurityGroups(logger: Logger): Promise<void> {
  const env = getEnvWithAws();
  const result = await executeWithExitCode(
    `aws ec2 describe-security-groups --region ${AWS_REGION} --filters Name=group-name,Values=prime-rl-efs-sg --query 'SecurityGroups[*].GroupId' --output text`,
    { env, silent: true }
  );

  if (!result.success) {
    logger.warn(`[Setup] Could not query prior EFS security groups: ${result.output}`);
    return;
  }

  const securityGroupIds = result.output.trim().split(/\s+/).filter(Boolean);
  if (securityGroupIds.length === 0) {
    return;
  }

  for (const securityGroupId of securityGroupIds) {
    logger.info(`[Setup] Removing prior EFS security group: ${securityGroupId}`);
    await executeWithExitCode(
      `aws ec2 delete-security-group --region ${AWS_REGION} --group-id ${securityGroupId}`,
      { env }
    );
  }
}

function cleanupTempSetupFiles(): void {
  for (const filePath of TEMP_SETUP_FILES) {
    try {
      fs.rmSync(filePath, { force: true });
    } catch {
      // Ignore cleanup failures for temp files.
    }
  }
}

export async function prepareCloudForRollout(logger: Logger): Promise<void> {
  logger.info(
    "[Setup] Resetting cloud rollout state while preserving the EKS cluster, nodegroups, and GPU operator..."
  );

  if (!(await clusterExists())) {
    cleanupTempSetupFiles();
    logger.info("[Setup] No active EKS cluster found; skipping cloud rollout reset");
    return;
  }

  const env = getEnvWithAws();
  const kubeconfigResult = await executeWithExitCode(
    `aws eks update-kubeconfig --name ${CLUSTER_NAME} --region ${AWS_REGION}`,
    { env, silent: true, timeout: 60000 }
  );

  if (!kubeconfigResult.success) {
    throw new Error(
      `Failed to update kubeconfig for cloud rollout reset: ${kubeconfigResult.output}`
    );
  }

  await uninstallHelmRelease(logger, RELEASE_NAME);
  await deleteKubernetesRolloutResources(logger);
  await removeEfsCsiDriver(logger);
  await deleteEfsResources(logger);
  await deleteEfsSecurityGroups(logger);

  cleanupTempSetupFiles();
  logger.info("[Setup] Cloud rollout state reset complete");
}
