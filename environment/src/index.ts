import {
  EnvironmentDefinition,
  Logger,
  ConsoleLogger,
  TestResult,
  runSimpleTests,
  loadProblemsFromDirectory,
} from "@hyperfocal/env-base";
import * as path from "path";
import { fileURLToPath } from "url";

import { ensureKubectl, ensureHelm, ensureEksctl } from "./helpers.js";
import { runSetup } from "./setup/index.js";
import { prepareCloudForRollout } from "./setup/cloud-reset.js";
import { prepareWorkspaceForRollout } from "./setup/workspace.js";
import { prepareSanitizedRuntimeImage } from "./setup/runtime-image.js";
import {
  createPrerequisiteTests,
  createEksClusterTests,
  createGpuOperatorTests,
  createStorageTests,
  createHelmDeploymentTests,
  createTrainerUnitTests,
  createWeightSyncUnitTests,
  createCodeRubricTests,
  createProcessRubricTests,
  createTrainingVerificationTests,
  createTrainingQualityTests,
  createManifestVerificationTests,
} from "./tests/index.js";
import { attachOutcomeNarratives } from "./tests/narratives.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const problems = loadProblemsFromDirectory(path.join(__dirname, ".."));

class Environment implements EnvironmentDefinition {
  async listProblems() {
    return problems;
  }

  async setupProblem(problemId: string, logger?: Logger): Promise<void> {
    const log = logger ?? new ConsoleLogger();
    log.info(`Setting up problem: ${problemId}`);

    await prepareWorkspaceForRollout(log);

    await ensureKubectl(log);
    await ensureHelm(log);
    await ensureEksctl(log);

    await prepareSanitizedRuntimeImage(log);
    await prepareCloudForRollout(log);
    await runSetup(log);

    log.info("Problem setup completed successfully");
  }

  async runTests(problemId: string, logger: Logger): Promise<TestResult[]> {
    const tests = attachOutcomeNarratives([
      ...createPrerequisiteTests(),
      ...createEksClusterTests(),
      ...createGpuOperatorTests(),
      ...createStorageTests(),
      ...createHelmDeploymentTests(),
      ...createTrainerUnitTests(),
      ...createWeightSyncUnitTests(),
      ...createCodeRubricTests(),
      ...createProcessRubricTests(problemId),
      ...createTrainingVerificationTests(),
      ...createTrainingQualityTests(),
      ...createManifestVerificationTests(),
    ]);
    return runSimpleTests(tests, logger);
  }

  async cleanup(logger?: Logger): Promise<void> {
    const log = logger ?? new ConsoleLogger();

    log.info("Starting rollout cleanup while preserving reusable EKS/GPU infrastructure");
    await prepareCloudForRollout(log);
    log.info("Cleanup complete; preserved EKS cluster, nodegroups, and GPU operator");
  }
}

export default new Environment();
