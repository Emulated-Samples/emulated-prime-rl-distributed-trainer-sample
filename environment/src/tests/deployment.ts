import { Logger, SimpleTest, executeWithExitCode } from "@hyperfocal/env-base";

import {
  AWS_REGION,
  RELEASE_NAME,
  CHECKPOINT_TIMEOUT,
  INFERENCE_STARTUP_TIMEOUT,
  POD_READY_TIMEOUT,
  PVC_BIND_TIMEOUT,
  TRAINING_COMPLETION_TIMEOUT,
  TRAINING_PROGRESS_TIMEOUT,
  getEvalRuntimeImageUri,
} from "../config.js";

import {
  classifyAndAttributeError,
  getEnvWithAws,
  pollUntil,
  registerFailureCategory,
  KubernetesPodList,
  KubernetesPVC,
} from "../helpers.js";

function stripAnsi(value: string): string {
  return value.replace(/\x1b\[[0-9;]*m/g, "");
}

function getMaxTrainerStep(logs: string): number | null {
  const normalized = stripAnsi(logs);
  const matches = Array.from(
    normalized.matchAll(/\bStep\s+(\d+)(?:\/\d+)?\s*\|/g)
  );
  if (matches.length === 0) return null;
  return Math.max(...matches.map((match) => Number.parseInt(match[1], 10)));
}

function getTrainerCompletionEvidence(logs: string): string | null {
  const normalized = stripAnsi(logs);
  if (normalized.includes("RL trainer finished")) {
    return "trainer log showed RL trainer finished";
  }
  if (
    normalized.includes("Writing final checkpoint") ||
    normalized.includes("Writing final weight checkpoint")
  ) {
    return "trainer log showed the final checkpoint was written";
  }
  const maxStep = getMaxTrainerStep(normalized);
  if (maxStep !== null && maxStep >= 19) {
    return `trainer reached step ${maxStep}`;
  }
  return null;
}

function getTrainerProgressEvidence(logs: string): string | null {
  const completion = getTrainerCompletionEvidence(logs);
  if (completion) return completion;

  const maxStep = getMaxTrainerStep(logs);
  if (maxStep !== null && maxStep >= 3) {
    return `trainer reached step ${maxStep}`;
  }
  return null;
}

interface KubernetesPV {
  metadata?: {
    name?: string;
  };
  spec?: {
    storageClassName?: string;
    csi?: {
      driver?: string;
      volumeHandle?: string;
    };
  };
  status?: {
    phase?: string;
  };
}

interface KubernetesStorageClass {
  provisioner?: string;
}

interface SharedStorageDetails {
  pvcName: string;
  pvcPhase?: string;
  pvName: string;
  pvPhase?: string;
  storageClassName?: string;
  driver?: string;
  volumeHandle?: string;
  fileSystemId?: string;
  accessPointId?: string;
}

type StorageLookupResult =
  | { success: true; details: SharedStorageDetails }
  | { success: false; error: string; errored?: boolean };

function parseEfsIds(volumeHandle?: string): {
  fileSystemId?: string;
  accessPointId?: string;
} {
  const fileSystemId = volumeHandle?.match(/\b(fs-[A-Za-z0-9]+)\b/)?.[1];
  const accessPointId = volumeHandle?.match(/\b(fsap-[A-Za-z0-9]+)\b/)?.[1];
  return { fileSystemId, accessPointId };
}

async function getSharedStorageDetails(
  env: Record<string, string>
): Promise<StorageLookupResult> {
  const pvcName = `${RELEASE_NAME}-shared-data`;
  const pvcResult = await executeWithExitCode(
    `kubectl get pvc ${pvcName} -o json`,
    { env, silent: true }
  );

  if (!pvcResult.success) {
    return {
      success: false,
      error: `PVC ${pvcName} was not found or could not be read`,
    };
  }

  let pvc: KubernetesPVC;
  try {
    pvc = JSON.parse(pvcResult.output);
  } catch {
    return {
      success: false,
      error: `Failed to parse PVC ${pvcName}`,
      errored: true,
    };
  }

  const pvcPhase = pvc.status?.phase;
  const pvName = pvc.spec?.volumeName;
  const storageClassName = pvc.spec?.storageClassName;

  if (pvcPhase !== "Bound") {
    return {
      success: false,
      error: `PVC ${pvcName} phase was ${pvcPhase ?? "unknown"}, expected Bound`,
    };
  }

  if (!pvName) {
    return {
      success: false,
      error: `PVC ${pvcName} was Bound but did not report a backing PersistentVolume`,
    };
  }

  const pvResult = await executeWithExitCode(`kubectl get pv ${pvName} -o json`, {
    env,
    silent: true,
  });

  if (!pvResult.success) {
    return {
      success: false,
      error: `Backing PersistentVolume ${pvName} for PVC ${pvcName} was not found`,
    };
  }

  let pv: KubernetesPV;
  try {
    pv = JSON.parse(pvResult.output);
  } catch {
    return {
      success: false,
      error: `Failed to parse PersistentVolume ${pvName}`,
      errored: true,
    };
  }

  const volumeHandle = pv.spec?.csi?.volumeHandle;
  const { fileSystemId, accessPointId } = parseEfsIds(volumeHandle);

  return {
    success: true,
    details: {
      pvcName,
      pvcPhase,
      pvName,
      pvPhase: pv.status?.phase,
      storageClassName: storageClassName ?? pv.spec?.storageClassName,
      driver: pv.spec?.csi?.driver,
      volumeHandle,
      fileSystemId,
      accessPointId,
    },
  };
}

async function getCompletionEvidenceFromCheckpoint(
  env: Record<string, string>
): Promise<string | null> {
  const checkpointResult = await executeWithExitCode(
    `kubectl exec ${RELEASE_NAME}-trainer-0 -- sh -c 'test -f /data/outputs/run_default/broadcasts/step_20/STABLE && echo stable || find /data/outputs -name STABLE -type f 2>/dev/null | head -1'`,
    { env, silent: true, timeout: 30000 }
  );

  if (!checkpointResult.success) {
    return null;
  }

  const evidence = checkpointResult.output.trim();
  if (!evidence) {
    return null;
  }

  return evidence === "stable"
    ? "final checkpoint STABLE marker existed at step_20"
    : `final checkpoint STABLE marker existed at ${evidence}`;
}

/**
 * Verify EFS storage is set up correctly.
 */
export function createStorageTests(): SimpleTest[] {
  return [
    {
      id: "efs-exists",
      name: "EFS Filesystem Exists",
      description: "Verify EFS filesystem exists and is available",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();
        const storage = await getSharedStorageDetails(env);

        if (!storage.success) {
          return storage;
        }

        const { details } = storage;
        if (!details.fileSystemId) {
          return {
            success: false,
            error: `Bound PV ${details.pvName} did not expose an EFS filesystem ID in volumeHandle ${details.volumeHandle ?? "missing"}`,
          };
        }

        const result = await executeWithExitCode(
          `aws efs describe-file-systems --file-system-id ${details.fileSystemId} --region ${AWS_REGION} --query "FileSystems[0].{Id:FileSystemId,State:LifeCycleState}" --output json`,
          { env }
        );

        if (!result.success) {
          return {
            success: false,
            error: `EFS filesystem ${details.fileSystemId} from bound PV ${details.pvName} could not be described`,
            output: result.output,
          };
        }

        try {
          const efs = JSON.parse(result.output);
          if (!efs?.Id) {
            return {
              success: false,
              error: `EFS filesystem ${details.fileSystemId} was not found`,
            };
          }
          if (efs.State !== "available") {
            return {
              success: false,
              error: `EFS filesystem ${efs.Id} state was ${efs.State}, expected available`,
            };
          }

          const message = `PVC ${details.pvcName} was backed by available EFS filesystem ${efs.Id}`;
          logger.info(message);
          return { success: true, output: message };
        } catch {
          return {
            success: false,
            error: `Failed to parse EFS response for filesystem ${details.fileSystemId}`,
            errored: true,
          };
        }
      },
    },
    {
      id: "efs-access-point-exists",
      name: "EFS Access Point Exists",
      description: "Verify EFS access point with correct UID exists",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();
        const storage = await getSharedStorageDetails(env);

        if (!storage.success) {
          return storage;
        }

        const { details } = storage;
        if (!details.accessPointId) {
          return {
            success: false,
            error: `Bound PV ${details.pvName} did not use an EFS access point in volumeHandle ${details.volumeHandle ?? "missing"}`,
          };
        }

        const result = await executeWithExitCode(
          `aws efs describe-access-points --access-point-id ${details.accessPointId} --region ${AWS_REGION} --query 'AccessPoints[0].{Id:AccessPointId,FileSystemId:FileSystemId,Uid:PosixUser.Uid,Gid:PosixUser.Gid,Perms:RootDirectory.CreationInfo.Permissions}' --output json`,
          { env }
        );

        if (!result.success) {
          return {
            success: false,
            error: `EFS access point ${details.accessPointId} from bound PV ${details.pvName} could not be described`,
            output: result.output,
          };
        }

        try {
          const ap = JSON.parse(result.output);
          if (!ap || !ap.Id) {
            return {
              success: false,
              error: "No access point found for EFS",
            };
          }

          if (details.fileSystemId && ap.FileSystemId !== details.fileSystemId) {
            return {
              success: false,
              error: `Access point ${ap.Id} belonged to filesystem ${ap.FileSystemId}, expected ${details.fileSystemId}`,
            };
          }

          const uid = String(ap.Uid ?? "");
          const gid = String(ap.Gid ?? "");
          if (uid !== "1000" || gid !== "1000") {
            return {
              success: false,
              error: `Access point ${ap.Id} POSIX identity was ${uid || "unset"}:${gid || "unset"}, expected 1000:1000`,
            };
          }

          const message = `Bound PV ${details.pvName} used EFS access point ${ap.Id} with UID:GID ${uid}:${gid} and permissions ${ap.Perms ?? "unknown"}`;
          logger.info(message);
          return { success: true, output: message };
        } catch {
          return {
            success: false,
            error: `Failed to parse access point response for ${details.accessPointId}`,
            errored: true,
          };
        }
      },
    },
    {
      id: "efs-csi-driver-installed",
      name: "EFS CSI Driver Installed",
      description: "Verify AWS EFS CSI driver is installed",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();

        const result = await executeWithExitCode(
          "kubectl get csidriver efs.csi.aws.com -o json",
          { env, silent: true }
        );

        if (!result.success) {
          return {
            success: false,
            error: "EFS CSI driver efs.csi.aws.com was not registered in Kubernetes",
          };
        }

        const podsResult = await executeWithExitCode(
          "kubectl get pods -n kube-system -l app.kubernetes.io/name=aws-efs-csi-driver -o json",
          { env, silent: true }
        );

        if (podsResult.success) {
          try {
            const pods: KubernetesPodList = JSON.parse(podsResult.output);
            const running = pods.items.filter(
              (p) => p.status.phase === "Running"
            );
            logger.info(`EFS CSI driver pods: ${running.length} running`);
            return {
              success: true,
              output: `CSIDriver efs.csi.aws.com existed with ${running.length} EFS CSI driver pod(s) Running`,
            };
          } catch {}
        }

        logger.info("CSIDriver efs.csi.aws.com is registered");
        return { success: true, output: "CSIDriver efs.csi.aws.com was registered" };
      },
    },
    {
      id: "storage-class-exists",
      name: "StorageClass Exists",
      description: "Verify the shared-storage StorageClass uses the EFS CSI provisioner",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();
        const storage = await getSharedStorageDetails(env);

        if (!storage.success) {
          return storage;
        }

        const { details } = storage;
        if (!details.storageClassName) {
          return {
            success: false,
            error: `PVC ${details.pvcName} did not report a StorageClass`,
          };
        }

        const result = await executeWithExitCode(
          `kubectl get storageclass ${details.storageClassName} -o json`,
          { env }
        );

        if (!result.success) {
          return {
            success: false,
            error: `StorageClass ${details.storageClassName} referenced by PVC ${details.pvcName} was not found`,
          };
        }

        try {
          const storageClass: KubernetesStorageClass = JSON.parse(result.output);
          if (storageClass.provisioner !== "efs.csi.aws.com") {
            return {
              success: false,
              error: `StorageClass ${details.storageClassName} used provisioner ${storageClass.provisioner}, expected efs.csi.aws.com`,
            };
          }
        } catch {
          return {
            success: false,
            error: `Failed to parse StorageClass ${details.storageClassName}`,
            errored: true,
          };
        }

        const message = `StorageClass ${details.storageClassName} for PVC ${details.pvcName} used provisioner efs.csi.aws.com`;
        logger.info(message);
        return { success: true, output: message };
      },
    },
    {
      id: "pv-exists",
      name: "PersistentVolume Exists",
      description: "Verify the shared-storage PVC is backed by an EFS PersistentVolume",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();
        const storage = await getSharedStorageDetails(env);

        if (!storage.success) {
          return storage;
        }

        const { details } = storage;
        if (details.driver !== "efs.csi.aws.com") {
          return {
            success: false,
            error: `PersistentVolume ${details.pvName} used CSI driver ${details.driver ?? "missing"}, expected efs.csi.aws.com`,
          };
        }
        if (!details.fileSystemId) {
          return {
            success: false,
            error: `PersistentVolume ${details.pvName} did not include an EFS filesystem ID in volumeHandle ${details.volumeHandle ?? "missing"}`,
          };
        }
        if (!details.accessPointId) {
          return {
            success: false,
            error: `PersistentVolume ${details.pvName} did not include an EFS access point in volumeHandle ${details.volumeHandle ?? "missing"}`,
          };
        }
        const message = `PVC ${details.pvcName} was bound to PV ${details.pvName} using efs.csi.aws.com and EFS access point ${details.accessPointId}`;
        logger.info(message);
        return { success: true, output: message };
      },
    },
  ];
}

/**
 * Verify prime-rl Helm deployment.
 */
export function createHelmDeploymentTests(): SimpleTest[] {
  return [
    {
      id: "helm-release-exists",
      name: "Helm Release Exists",
      description: "Verify prime-rl Helm release is deployed",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();

        const result = await executeWithExitCode(
          `helm list -o json`,
          { env }
        );

        if (!result.success || !result.output.includes(RELEASE_NAME)) {
          return {
            success: false,
            error: registerFailureCategory("helm-release-missing", "helm-release-exists", logger),
          };
        }

        const message = `Helm release ${RELEASE_NAME} existed`;
        logger.info(message);
        return { success: true, output: message };
      },
    },
    {
      id: "sanitized-runtime-image-used",
      name: "Sanitized Runtime Image Used",
      description: "Verify all Prime-RL pods use the sanitized evaluation runtime image",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();
        const accountId = env.AWS_ACCOUNT_ID?.trim();
        if (!accountId) {
          return {
            success: false,
            error: "AWS_ACCOUNT_ID is required to verify sanitized runtime image",
            errored: true,
          };
        }

        const expectedImage = getEvalRuntimeImageUri(accountId);
        const result = await executeWithExitCode(
          `kubectl get statefulsets -l app.kubernetes.io/instance=${RELEASE_NAME} -o json`,
          { env, timeout: 30000 }
        );

        if (!result.success) {
          return {
            success: false,
            error: `Failed to inspect StatefulSets: ${result.output}`,
          };
        }

        try {
          const parsed = JSON.parse(result.output);
          const images: string[] = [];
          for (const item of parsed.items ?? []) {
            for (const container of item.spec?.template?.spec?.containers ?? []) {
              images.push(container.image);
            }
          }

          const wrongImages = images.filter((image) => image !== expectedImage);
          if (images.length < 3 || wrongImages.length > 0) {
            return {
              success: false,
              error: `Expected all Prime-RL containers to use ${expectedImage}; found ${images.join(", ")}`,
            };
          }

          const message = `${images.length} Prime-RL containers used sanitized runtime image ${expectedImage}`;
          logger.info(message);
          return { success: true, output: message };
        } catch (error) {
          return {
            success: false,
            error: `Failed to parse StatefulSet images: ${error}`,
            errored: true,
          };
        }
      },
    },
    {
      id: "pods-running",
      name: "All Pods Running",
      description: "Verify all prime-rl pods are running",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();

        const result = await pollUntil(
          async () => {
            const podsResult = await executeWithExitCode(
              `kubectl get pods -l app.kubernetes.io/instance=${RELEASE_NAME} -o json`,
              { env, silent: true }
            );

            if (!podsResult.success) {
              return { done: false, message: "Cannot get pods" };
            }

            try {
              const pods: KubernetesPodList = JSON.parse(podsResult.output);
              const runningPods = pods.items.filter(
                (pod) => pod.status.phase === "Running"
              );

              const total = pods.items.length;
              const running = runningPods.length;

              // Need 3 pods: orchestrator, inference, trainer
              if (total >= 3 && running >= 3) {
                return { done: true };
              }

              const statuses = pods.items.map(
                (pod) => `${pod.metadata.name}: ${pod.status.phase}`
              );

              return {
                done: false,
                message: `Pods: ${running}/${total}. ${statuses.join(", ")}`,
              };
            } catch {
              return { done: false, message: "Failed to parse pod status" };
            }
          },
          {
            timeout: POD_READY_TIMEOUT,
            interval: 10000,
            description: "Checking pods",
            logger,
          }
        );

        if (!result.success) {
          logger.info("Pods failed to reach Running state - gathering diagnostics");

          const podsOverview = await runDiagnosticCommand(
            `kubectl get pods -l app.kubernetes.io/instance=${RELEASE_NAME} -o wide 2>&1`,
            env
          );
          logger.info(`Pod status:\n${podsOverview}`);

          const events = await runDiagnosticCommand(
            `kubectl get events --sort-by='.lastTimestamp' 2>&1 | tail -30`,
            env
          );
          logger.info(`Recent events:\n${events}`);

          const podDetails = await runDiagnosticCommand(
            `kubectl describe pods -l app.kubernetes.io/instance=${RELEASE_NAME} 2>&1 | tail -100`,
            env
          );
          logger.info(`Pod details:\n${podDetails}`);

          const nodeStatus = await runDiagnosticCommand(
            `kubectl get nodes -o wide 2>&1`,
            env
          );
          logger.info(`Node status:\n${nodeStatus}`);

          return { success: false, error: result.error };
        }

        const podsOutput = await executeWithExitCode(
          `kubectl get pods -l app.kubernetes.io/instance=${RELEASE_NAME} -o wide`,
          { env }
        );
        logger.info(`Running pods:\n${podsOutput.output}`);
        const runningCount = Math.max(0, podsOutput.output.trim().split("\n").length - 1);
        return {
          success: true,
          output: `${runningCount} Prime-RL pods were Running`,
        };
      },
    },
    {
      id: "services-exist",
      name: "Services Exist",
      description: "Verify Kubernetes services are created",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();

        const svcResult = await executeWithExitCode(
          `kubectl get svc -l app.kubernetes.io/instance=${RELEASE_NAME}`,
          { env }
        );

        if (!svcResult.success) {
          return {
            success: false,
            error: `Failed to get services: ${svcResult.output}`,
          };
        }

        const expectedServices = [
          `${RELEASE_NAME}-orchestrator`,
          `${RELEASE_NAME}-inference`,
          `${RELEASE_NAME}-trainer`,
        ];

        const missing: string[] = [];
        for (const svc of expectedServices) {
          if (!svcResult.output.includes(svc)) {
            missing.push(svc);
          }
        }

        if (missing.length > 0) {
          return {
            success: false,
            error: `Missing services: ${missing.join(", ")}`,
          };
        }

        const message = `required services existed: ${expectedServices.join(", ")}`;
        logger.info(message);
        return { success: true, output: message };
      },
    },
    {
      id: "pvc-bound",
      name: "PVC Bound",
      description: "Verify shared storage PVC is bound",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();

        const result = await pollUntil(
          async () => {
            const pvcResult = await executeWithExitCode(
              `kubectl get pvc ${RELEASE_NAME}-shared-data -o json`,
              { env, silent: true }
            );

            if (!pvcResult.success) {
              return { done: false, message: "PVC not found" };
            }

            try {
              const pvc: KubernetesPVC = JSON.parse(pvcResult.output);
              const phase = pvc.status?.phase;
              const storageClassName = pvc.spec?.storageClassName;
              const volumeName = pvc.spec?.volumeName;

              if (phase === "Bound" && volumeName) {
                return { done: true };
              }

              return {
                done: false,
                message: `PVC status: ${phase}, storageClassName: ${storageClassName}, volumeName: ${volumeName}`,
              };
            } catch {
              return { done: false, message: "Failed to parse PVC status" };
            }
          },
          {
            timeout: PVC_BIND_TIMEOUT,
            interval: 5000,
            description: "Checking PVC",
            logger,
          }
        );

        if (!result.success) {
          return { success: false, error: result.error };
        }

        const storage = await getSharedStorageDetails(env);
        if (!storage.success) {
          return storage;
        }

        const message = `${storage.details.pvcName} PVC was Bound to PV ${storage.details.pvName} with storageClassName ${storage.details.storageClassName ?? "unset"}`;
        logger.info(message);
        return { success: true, output: message };
      },
    },
  ];
}

/**
 * Verify training is working.
 */
export function createTrainingVerificationTests(): SimpleTest[] {
  return [
    {
      id: "inference-health",
      name: "Inference Server Health",
      description: "Verify inference server (vLLM) is responding",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();

        const result = await pollUntil(
          async () => {
            const logsResult = await executeWithExitCode(
              `kubectl logs ${RELEASE_NAME}-inference-0 --tail=100`,
              { env, silent: true }
            );

            if (!logsResult.success) {
              return { done: false, message: "Cannot get inference logs" };
            }

            if (
              logsResult.output.includes("Uvicorn running") ||
              logsResult.output.includes("Application startup complete") ||
              logsResult.output.includes("Started server process") ||
              logsResult.output.includes("Starting vLLM API server") ||
              logsResult.output.includes("vLLM API server version") ||
              logsResult.output.includes("Engine 000:") ||
              logsResult.output.includes("/v1/chat/completions") ||
              logsResult.output.includes("Model loaded")
            ) {
              return { done: true };
            }

            if (logsResult.output.trim().length === 0) {
              return { done: false, message: "Logs empty - container may still be starting" };
            }

            return {
              done: false,
              message: "Waiting for inference server",
            };
          },
          {
            timeout: INFERENCE_STARTUP_TIMEOUT,
            interval: 15000,
            description: "Checking inference",
            logger,
          }
        );

        if (!result.success) {
          logger.info("Inference health check failed - gathering diagnostics");

          const logs = await runDiagnosticCommand(
            `kubectl logs ${RELEASE_NAME}-inference-0 --tail=150 2>&1`,
            env
          );

          if (logs.trim().length === 0 || logs === "[no output]") {
            logger.info("Pod logs are empty. Container may have crashed before producing output");
          } else {
            logger.info(`Inference logs:\n${logs}`);
          }

          const events = await runDiagnosticCommand(
            `kubectl get events --field-selector involvedObject.name=${RELEASE_NAME}-inference-0 --sort-by='.lastTimestamp' 2>&1 | tail -20`,
            env
          );
          logger.info(`Pod events:\n${events}`);

          const describe = await runDiagnosticCommand(
            `kubectl describe pod ${RELEASE_NAME}-inference-0 2>&1 | tail -50`,
            env
          );
          logger.info(`Pod describe:\n${describe}`);

          const podStatus = await runDiagnosticCommand(
            `kubectl get pod ${RELEASE_NAME}-inference-0 -o jsonpath='{.status.phase} - {.status.containerStatuses[0].state}' 2>&1`,
            env
          );
          logger.info(`Pod status: ${podStatus}`);

          return { success: false, error: result.error };
        }

        const message = "inference logs showed the server reached a ready state";
        logger.info(message);
        return { success: true, output: message };
      },
    },
    {
      id: "training-progress",
      name: "Training Makes Progress",
      description: "Verify training steps are completing (step >= 3)",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();
        let progressEvidence: string | null = null;

        const result = await pollUntil(
          async () => {
            const logsResult = await executeWithExitCode(
              `kubectl logs ${RELEASE_NAME}-trainer-0 --tail=2000`,
              { env, silent: true }
            );

            if (!logsResult.success) {
              return { done: false, message: "Cannot get trainer logs" };
            }

            const evidence = getTrainerProgressEvidence(logsResult.output);
            if (evidence) {
              progressEvidence = evidence;
              return { done: true };
            }

            const maxStep = getMaxTrainerStep(logsResult.output);
            if (maxStep !== null) {
              return { done: false, message: `At step ${maxStep}` };
            }
            if (
              logsResult.output.includes("Starting training") ||
              logsResult.output.includes("Initializing")
            ) {
              return { done: false, message: "Training initializing" };
            }

            return { done: false, message: "Waiting for training" };
          },
          {
            timeout: TRAINING_PROGRESS_TIMEOUT,
            interval: 20000,
            description: "Checking training progress",
            logger,
          }
        );

        if (!result.success) {
          const fullLogs = await executeWithExitCode(
            `kubectl logs ${RELEASE_NAME}-trainer-0 --tail=-1`,
            { env, silent: true, timeout: 30000 }
          );
          if (fullLogs.success) {
            const lateEvidence = getTrainerProgressEvidence(fullLogs.output);
            if (lateEvidence) {
              const message = `${lateEvidence} before final progress verdict`;
              logger.info(message);
              return { success: true, output: message };
            }
          }

          const checkpointEvidence = await getCompletionEvidenceFromCheckpoint(env);
          if (checkpointEvidence) {
            logger.info(checkpointEvidence);
            return { success: true, output: checkpointEvidence };
          }

          logger.info("Training progress check failed - gathering diagnostics");

          const logs = await runDiagnosticCommand(
            `kubectl logs ${RELEASE_NAME}-trainer-0 --tail=500 2>&1`,
            env
          );

          if (logs.trim().length === 0 || logs === "[no output]") {
            logger.info("WARNING: Trainer logs are EMPTY - container may have crashed");
          } else {
            logger.info(`Trainer logs:\n${logs}`);
          }

          const events = await runDiagnosticCommand(
            `kubectl get events --field-selector involvedObject.name=${RELEASE_NAME}-trainer-0 --sort-by='.lastTimestamp' 2>&1 | tail -20`,
            env
          );
          logger.info(`Trainer pod events:\n${events}`);

          const podStatus = await runDiagnosticCommand(
            `kubectl get pod ${RELEASE_NAME}-trainer-0 -o jsonpath='{.status.phase} - {.status.containerStatuses[0].state}' 2>&1`,
            env
          );
          logger.info(`Trainer pod status: ${podStatus}`);

          return { success: false, error: result.error };
        }

        const message = progressEvidence ?? "trainer reached at least step 3";
        logger.info(message);
        return { success: true, output: message };
      },
    },
    {
      id: "training-completes",
      name: "Training Completes",
      description: "Wait for training to complete all 20 steps",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();

        const result = await pollUntil(
          async () => {
            const logsResult = await executeWithExitCode(
              `kubectl logs ${RELEASE_NAME}-trainer-0 --tail=300`,
              { env, silent: true }
            );

            if (!logsResult.success) {
              return { done: false, message: "Cannot get trainer logs" };
            }

            if (
              logsResult.output.includes("Step 19 |") ||
              logsResult.output.includes("RL trainer finished") ||
              logsResult.output.includes("Writing final checkpoint") ||
              logsResult.output.includes("Writing final weight checkpoint")
            ) {
              return { done: true };
            }

            const stepMatches = logsResult.output.match(/Step\s+(\d+)\s+\|/g);
            if (stepMatches && stepMatches.length > 0) {
              const steps = stepMatches.map((m) => {
                const num = m.match(/\d+/);
                return num ? parseInt(num[0]) : 0;
              });
              const maxStep = Math.max(...steps);
              return { done: false, message: `Step ${maxStep}/20` };
            }

            return { done: false, message: "Training in progress" };
          },
          {
            timeout: TRAINING_COMPLETION_TIMEOUT,
            interval: 30000,
            description: "Waiting for completion",
            logger,
          }
        );

        if (!result.success) {
          const fullLogs = await executeWithExitCode(
            `kubectl logs ${RELEASE_NAME}-trainer-0 --tail=-1`,
            { env, silent: true, timeout: 30000 }
          );
          if (fullLogs.success) {
            const lateEvidence = getTrainerCompletionEvidence(fullLogs.output);
            if (lateEvidence) {
              const message = `${lateEvidence} before final completion verdict`;
              logger.info(message);
              return { success: true, output: message };
            }
          }

          const checkpointEvidence = await getCompletionEvidenceFromCheckpoint(env);
          if (checkpointEvidence) {
            logger.info(checkpointEvidence);
            return { success: true, output: checkpointEvidence };
          }

          return { success: false, error: result.error };
        }

        const message = "trainer reached the required final training step";
        logger.info(message);
        return { success: true, output: message };
      },
    },
    {
      id: "checkpoint-saved",
      name: "Checkpoint Artifacts Present",
      description: "Verify final checkpoint marker and model weights were saved to shared storage",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();

        const result = await pollUntil(
          async () => {
            const checkResult = await executeWithExitCode(
              `kubectl exec ${RELEASE_NAME}-trainer-0 -- ls -la /data/outputs/ 2>/dev/null || echo "dir_not_found"`,
              { env, silent: true }
            );

            if (
              !checkResult.success ||
              checkResult.output.includes("dir_not_found")
            ) {
              return { done: false, message: "Outputs directory not found" };
            }

            const weightsCheck = await executeWithExitCode(
              `kubectl exec ${RELEASE_NAME}-trainer-0 -- ls /data/outputs/weights/ 2>/dev/null || echo "weights_not_found"`,
              { env, silent: true }
            );

            if (
              weightsCheck.success &&
              !weightsCheck.output.includes("weights_not_found") &&
              weightsCheck.output.includes("step_")
            ) {
              return { done: true };
            }

            const ckptCheck = await executeWithExitCode(
              `kubectl exec ${RELEASE_NAME}-trainer-0 -- find /data/outputs -name "*.pt" -o -name "*.safetensors" 2>/dev/null | head -5`,
              { env, silent: true }
            );

            if (ckptCheck.success && ckptCheck.output.trim()) {
              return { done: true };
            }

            return { done: false, message: "Waiting for checkpoint" };
          },
          {
            timeout: CHECKPOINT_TIMEOUT,
            interval: 15000,
            description: "Checking checkpoint",
            logger,
          }
        );

        if (!result.success) {
          return { success: false, error: result.error };
        }

        const lsResult = await executeWithExitCode(
          `kubectl exec ${RELEASE_NAME}-trainer-0 -- sh -c 'find /data/outputs \\( -name STABLE -o -name "*.safetensors" -o -path "*/weights/step_*/*" \\) -type f 2>/dev/null | head -30'`,
          { env }
        );
        logger.info(`Checkpoint artifacts:\n${lsResult.output}`);
        const checkpointArtifacts = lsResult.output
          .split("\n")
          .map((line) => line.trim())
          .filter(Boolean).length;

        return {
          success: true,
          output: `found ${checkpointArtifacts} final checkpoint marker and model-weight artifact(s) under /data/outputs`,
        };
      },
    },
  ];
}

const DIAGNOSTIC_COMMAND_TIMEOUT = 15 * 1000;
const DIAGNOSTIC_KUBECTL_REQUEST_TIMEOUT = "15s";

async function runDiagnosticCommand(
  command: string,
  env: NodeJS.ProcessEnv
): Promise<string> {
  const boundedCommand = command.startsWith("kubectl ")
    ? command.replace(
        /^kubectl /,
        `kubectl --request-timeout=${DIAGNOSTIC_KUBECTL_REQUEST_TIMEOUT} `
      )
    : command;

  const result = await executeWithExitCode(boundedCommand, {
    env,
    silent: true,
    timeout: DIAGNOSTIC_COMMAND_TIMEOUT,
  });

  if (!result.success && result.exitCode === 124) {
    return `[diagnostic timed out after ${DIAGNOSTIC_COMMAND_TIMEOUT}ms]\n${result.output}`;
  }

  if (!result.output.trim()) {
    return result.success
      ? "[no output]"
      : `[diagnostic failed with exit code ${result.exitCode}]`;
  }

  return result.output;
}
