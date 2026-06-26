import {
  Logger,
  SimpleTest,
  executeWithExitCode,
  readManifest,
} from "@hyperfocal/env-base";
import { getEnvWithAws, sleep } from "../helpers.js";
import {
  RELEASE_NAME,
  VF_EVAL_TIMEOUT,
  VF_EVAL_REWARD_THRESHOLD,
  WORKSPACE_PATH,
} from "../config.js";

interface Manifest {
  deployment?: { helmRelease?: string };
  paths?: { finalCheckpoint?: string };
  services?: { inference?: { modelName?: string } };
}

function getManifest(): Manifest | null {
  return readManifest(WORKSPACE_PATH) as Manifest | null;
}

export function createTrainingQualityTests(): SimpleTest[] {
  return [
    {
      id: "model-benchmark",
      name: "Model Passes Benchmark (vf-eval)",
      description: `Run vf-eval on trained model, verify avg reward >= ${VF_EVAL_REWARD_THRESHOLD}`,
      run: async (logger: Logger) => {
        const env = getEnvWithAws();

        const manifest = getManifest();
        const releaseName = manifest?.deployment?.helmRelease || RELEASE_NAME;
        const weightsPath = manifest?.paths?.finalCheckpoint ||
          "/data/outputs/run_default/broadcasts/step_20";
        const modelName = manifest?.services?.inference?.modelName ||
          "PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT";

        logger.info("Updating inference server to serve trained weights");
        const reloadResult = await executeWithExitCode(
          `kubectl exec ${releaseName}-orchestrator-0 -- curl -s -X POST ` +
          `"http://${releaseName}-inference:8000/update_weights" ` +
          `-H "Content-Type: application/json" ` +
          `-d '{"weight_dir": "${weightsPath}"}'`,
          { env, timeout: 60000 }
        );

        if (!reloadResult.success || !reloadResult.output.includes("ok")) {
          return {
            success: false,
            error: "vf-eval could not load the agent's trained checkpoint into the inference server; the reload request was rejected",
            output: reloadResult.output,
          };
        }

        logger.info(`Weight reload response: ${reloadResult.output.trim()}`);
        await sleep(15000);

        logger.info("Running vf-eval benchmark (this may take 2-5 minutes)");
        const evalResult = await executeWithExitCode(
          `kubectl exec ${releaseName}-orchestrator-0 -- ` +
          `uv run --no-sync vf-eval reverse-text ` +
          `-m "${modelName}" ` +
          `-b "http://${releaseName}-inference:8000/v1" ` +
          `-n 20 --max-tokens 256 2>&1`,
          { env, timeout: VF_EVAL_TIMEOUT }
        );

        if (!evalResult.success) {
          return {
            success: false,
            error: "vf-eval could not run against the deployed trained model",
            output: evalResult.output,
          };
        }
        logger.info(`vf-eval output:\n${evalResult.output}`);
        
        let avgReward: number | null = null;
        
        const avgDashMatch = evalResult.output.match(/reward:\s*avg\s*-\s*(\d+\.\d+)/);
        if (avgDashMatch) {
          avgReward = parseFloat(avgDashMatch[1]);
        }
        
        if (avgReward === null) {
          const avgAtMatch = evalResult.output.match(/Avg@\d+=(\d+\.\d+)/);
          if (avgAtMatch) {
            avgReward = parseFloat(avgAtMatch[1]);
          }
        }
        
        if (avgReward === null) {
          return {
            success: false,
            error: `Could not parse average reward from vf-eval output. ` +
                   `Expected format like "reward: avg - 0.831" or "Avg@3=0.7832"`
          };
        }

        logger.info(`Parsed average reward: ${avgReward.toFixed(4)}`);

        if (avgReward < VF_EVAL_REWARD_THRESHOLD) {
          return {
            success: false,
            error: `Model benchmark failed: avg reward ${avgReward.toFixed(4)} < ${VF_EVAL_REWARD_THRESHOLD} threshold. ` +
                   `This indicates training did not produce a model that learned the task. ` +
                   `The base model (untrained) typically scores ~0.05.`
          };
        }

        const message = `avg reward ${avgReward.toFixed(4)} met threshold ${VF_EVAL_REWARD_THRESHOLD}`;
        logger.info(`Model benchmark PASSED: ${message}`);
        return { success: true, output: message };
      }
    }
  ];
}
