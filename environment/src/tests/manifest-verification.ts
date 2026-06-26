import {
  Logger,
  SimpleTest,
  executeWithExitCode,
  readManifest,
  loadSchema,
  validateManifest,
} from "@hyperfocal/env-base";
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";

import {
  WORKSPACE_PATH,
  AWS_REGION,
  CLUSTER_NAME,
  RELEASE_NAME,
} from "../config.js";

import {
  getEnvWithAws,
  KubernetesPodList,
} from "../helpers.js";

interface Manifest {
  cluster?: { name?: string; region?: string };
  deployment?: { helmRelease?: string; namespace?: string };
  paths?: { finalCheckpoint?: string };
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const MANIFEST_SCHEMA_PATH = path.join(__dirname, "..", "..", "manifest-schema.json");

let manifestSchema: Record<string, unknown> | null = null;

function getManifest(): Manifest | null {
  const raw = readManifest(WORKSPACE_PATH);
  return raw as Manifest | null;
}

function getManifestSchema(): Record<string, unknown> {
  if (!manifestSchema) {
    manifestSchema = loadSchema(MANIFEST_SCHEMA_PATH);
  }

  return manifestSchema;
}

function formatValidationErrors(errors: { path: string; message: string }[]): string {
  return errors
    .map((error) => `${error.path || "/"} ${error.message}`.trim())
    .join("; ");
}

export function createManifestVerificationTests(): SimpleTest[] {
  return [
    {
      id: "output-manifest-exists",
      name: "Output Manifest Exists",
      description: "Agent created .hyperfocal/manifest.json declaring deployment details",
      run: async (logger: Logger) => {
        const manifestPath = path.join(WORKSPACE_PATH, ".hyperfocal", "manifest.json");

        if (!fs.existsSync(manifestPath)) {
          return {
            success: false,
            error: "No manifest found at .hyperfocal/manifest.json. " +
                   "You must declare deployment details for verification."
          };
        }

        try {
          const content = fs.readFileSync(manifestPath, "utf-8");
          JSON.parse(content);
        } catch (e) {
          return {
            success: false,
            error: `Manifest is not valid JSON: ${e}`
          };
        }

        const message = "output manifest existed at .hyperfocal/manifest.json";
        logger.info(`${message} (${manifestPath})`);
        return { success: true, output: message };
      }
    },
    {
      id: "output-manifest-valid",
      name: "Output Manifest Schema Valid",
      description: "Manifest conforms to required JSON schema structure",
      run: async (logger: Logger) => {
        const manifest = getManifest();

        if (!manifest) {
          return {
            success: false,
            error: "Cannot read manifest - run output-manifest-exists first"
          };
        }

        const validation = validateManifest(
          manifest as Record<string, unknown>,
          getManifestSchema()
        );

        if (!validation.valid) {
          return {
            success: false,
            error: `Manifest failed schema validation: ${formatValidationErrors(validation.errors || [])}`
          };
        }

        const message = "output manifest passed schema validation";
        logger.info(message);
        return { success: true, output: message };
      }
    },
    {
      id: "gpu-cluster-reachable",
      name: "GPU Cluster Reachable",
      description: "Can connect to the EKS cluster with GPU nodes",
      run: async (logger: Logger) => {
        const manifest = getManifest();
        const clusterName = manifest?.cluster?.name || CLUSTER_NAME;
        const region = manifest?.cluster?.region || AWS_REGION;
        const env = getEnvWithAws();

        logger.info(`Connecting to cluster ${clusterName} in ${region}...`);
        const updateResult = await executeWithExitCode(
          `aws eks update-kubeconfig --name ${clusterName} --region ${region}`,
          { env, timeout: 30000 }
        );

        if (!updateResult.success) {
          return {
            success: false,
            error: `Cannot reach cluster '${clusterName}' in ${region}: ${updateResult.output}`
          };
        }

        const testResult = await executeWithExitCode("kubectl cluster-info", { env });
        if (!testResult.success) {
          return {
            success: false,
            error: `kubectl cannot connect to cluster: ${testResult.output}`
          };
        }

        const gpuNodesResult = await executeWithExitCode(
          "kubectl get nodes -l nvidia.com/gpu.present=true -o name",
          { env }
        );

        if (!gpuNodesResult.success || !gpuNodesResult.output.trim()) {
          return { success: false, error: "No GPU nodes found in cluster" };
        }

        const gpuNodeCount = gpuNodesResult.output.trim().split("\n").length;
        const message = `cluster ${clusterName} in ${region} was reachable with ${gpuNodeCount} GPU nodes`;
        logger.info(message);
        return { success: true, output: message };
      }
    },
    {
      id: "deployment-pods-healthy",
      name: "Deployment Pods Healthy",
      description: "The training deployment has running pods for all components",
      run: async (logger: Logger) => {
        const manifest = getManifest();
        const releaseName = manifest?.deployment?.helmRelease || RELEASE_NAME;
        const namespace = manifest?.deployment?.namespace || "default";
        const env = getEnvWithAws();

        logger.info(`Checking pods for release ${releaseName}...`);
        const podsResult = await executeWithExitCode(
          `kubectl get pods -n ${namespace} -l "app.kubernetes.io/instance=${releaseName}" -o json`,
          { env }
        );

        if (!podsResult.success) {
          return {
            success: false,
            error: `Cannot list pods for release '${releaseName}': ${podsResult.output}`
          };
        }

        let pods: KubernetesPodList;
        try {
          pods = JSON.parse(podsResult.output);
        } catch {
          return { success: false, error: "Failed to parse pod status" };
        }

        const runningPods = pods.items.filter((p) => p.status.phase === "Running");

        if (runningPods.length < 3) {
          const statuses = pods.items.map((p) =>
            `${p.metadata.name}: ${p.status.phase}`
          ).join(", ");
          return {
            success: false,
            error: `Expected 3+ running pods for release '${releaseName}', found ${runningPods.length}. Status: ${statuses}`
          };
        }

        const components = ["orchestrator", "trainer", "inference"];
        const missing = components.filter(
          (c) => !pods.items.some((p) => p.metadata.name.includes(c))
        );

        if (missing.length > 0) {
          return { success: false, error: `Missing components: ${missing.join(", ")}` };
        }

        const message = `release ${releaseName} had ${runningPods.length} running pods`;
        logger.info(message);
        return { success: true, output: message };
      }
    },
    {
      id: "training-checkpoint-valid",
      name: "Training Checkpoint Valid",
      description: "The checkpoint path contains valid model weights",
      run: async (logger: Logger) => {
        const manifest = getManifest();
        const checkpointPath = manifest?.paths?.finalCheckpoint ||
                              "/data/outputs/run_default/broadcasts/step_20";
        const releaseName = manifest?.deployment?.helmRelease || RELEASE_NAME;
        const env = getEnvWithAws();

        logger.info(`Checking checkpoint at ${checkpointPath}...`);
        const stableCheck = await executeWithExitCode(
          `kubectl exec ${releaseName}-orchestrator-0 -- test -f ${checkpointPath}/STABLE && echo "exists"`,
          { env, timeout: 30000 }
        );

        if (!stableCheck.success || !stableCheck.output.includes("exists")) {
          return {
            success: false,
            error: `No STABLE marker at ${checkpointPath}/STABLE. Training may not have completed.`
          };
        }

        const filesCheck = await executeWithExitCode(
          `kubectl exec ${releaseName}-orchestrator-0 -- bash -c "ls ${checkpointPath}/*.safetensors 2>/dev/null | wc -l"`,
          { env }
        );

        const fileCount = parseInt(filesCheck.output.trim()) || 0;
        if (fileCount === 0) {
          return {
            success: false,
            error: `No safetensors files at ${checkpointPath}. Expected model weights.`
          };
        }

        const sizeCheck = await executeWithExitCode(
          `kubectl exec ${releaseName}-orchestrator-0 -- du -sb ${checkpointPath}/ | cut -f1`,
          { env }
        );

        const sizeBytes = parseInt(sizeCheck.output.trim()) || 0;
        const sizeGB = sizeBytes / (1024 * 1024 * 1024);

        if (sizeGB < 1.0) {
          return {
            success: false,
            error: `Checkpoint too small (${sizeGB.toFixed(2)}GB). Expected > 1GB for real weights.`
          };
        }

        const message = `checkpoint had ${fileCount} safetensors files and was ${sizeGB.toFixed(2)}GB`;
        logger.info(`Checkpoint valid: ${fileCount} files, ${sizeGB.toFixed(2)}GB`);
        return { success: true, output: message };
      }
    }
  ];
}
