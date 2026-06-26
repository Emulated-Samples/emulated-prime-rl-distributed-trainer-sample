import type {
  SimpleTest,
  TestOutcomeContext,
  TestOutcomeNarrative,
} from "@hyperfocal/env-base";

function pct(score: number): number {
  return Math.round(score * 100);
}

function clean(value: string | undefined, max = 220): string | null {
  if (!value) return null;
  const normalized = value
    .replace(/\x1b\[[0-9;]*m/g, "")
    .replace(/[a-f0-9]{64}/gi, (hash) => hash.slice(0, 12))
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^(?:PASS|FAIL|INFO|PARTIAL|SKIP)[:\s]+/i, "");
  if (!normalized) return null;
  return normalized.length > max ? `${normalized.slice(0, max - 3).trim()}...` : normalized;
}

function outputOr(fallback: string): (context: TestOutcomeContext) => string {
  return (context) => clean(context.output) ?? fallback;
}

function errorOr(fallback: string): (context: TestOutcomeContext) => string {
  return (context) => clean(context.error) ?? fallback;
}

function afterPrefix(
  prefix: string,
  fallback: string
): (context: TestOutcomeContext) => string {
  return (context) => {
    const error = context.error ?? "";
    const runtimeImageReason = runtimeImageImportFailure(error);
    if (runtimeImageReason) return runtimeImageReason;

    const index = error.indexOf(prefix);
    if (index >= 0) {
      return clean(error.slice(index + prefix.length)) ?? fallback;
    }
    return clean(error) ?? fallback;
  };
}

function pollFailure(fallback: string): (context: TestOutcomeContext) => string {
  return (context) => {
    const error = context.error ?? "";
    const match = error.match(/Did not complete ([^;]+) before deadline(?:;\s*last observed state:\s*(.+))?/i);
    if (!match) return clean(error) ?? fallback;

    const description = match[1]?.trim();
    const lastObserved = match[2]?.trim();
    const checkName = description
      ?.replace(/^Checking\s+/i, "")
      .replace(/^Waiting for\s+/i, "")
      .trim();
    if (lastObserved) {
      return clean(`${lastObserved} when the ${checkName || "poll"} check reached its deadline`) ?? fallback;
    }
    return clean(`${checkName || description} check did not complete before the deadline`) ?? fallback;
  };
}

function runtimeImageImportFailure(error: string): string | null {
  const missingPrimeModule =
    /ModuleNotFoundError:\s*No module named ['"]prime_rl(?:\.|['"])/.test(error) ||
    /ImportError:\s*cannot import name .* from ['"]prime_rl/.test(error) ||
    (
      /ModuleNotFoundError:\s*No module named ['"]pydantic_settings['"]/.test(error) &&
      /prime_rl[\/\\]utils[\/\\]pydantic_config\.py/.test(error)
    );
  if (!missingPrimeModule) return null;

  return "unit probe could not run because the deployed runtime did not expose the expected sanitized Prime-RL modules";
}

function probeFailure(fallback: string): (context: TestOutcomeContext) => string {
  return (context) => {
    const error = context.error ?? "";
    const runtimeImageReason = runtimeImageImportFailure(error);
    if (runtimeImageReason) return runtimeImageReason;

    const failLine = error.match(/FAIL:\s*([^\n\r]+)/);
    if (failLine?.[1]) {
      return clean(failLine[1]) ?? fallback;
    }
    const exceptionLine = error.match(/(?:^|\n)\s*((?:[A-Za-z_][\w.]*Error|AssertionError|Exception):\s*[^\n\r]+)/g)?.pop();
    if (exceptionLine) {
      return clean(`test raised ${exceptionLine.trim()}`) ?? fallback;
    }
    return clean(error) ?? fallback;
  };
}

function computeLossProbeFailure(fallback: string): (context: TestOutcomeContext) => string {
  return (context) => {
    const error = context.error ?? "";
    const runtimeImageReason = runtimeImageImportFailure(error);
    if (runtimeImageReason) return runtimeImageReason;

    if (/AttributeError:\s*'list' object has no attribute '(?:float|to|shape|dim|device|dtype)'/.test(error)) {
      return "compute_loss treated packed per-sequence input lists as a single tensor";
    }
    if (/TypeError:.*(?:list|Tensor).*not iterable|cannot unpack/i.test(error)) {
      return "compute_loss did not support the public loss, metrics return contract";
    }
    return probeFailure(fallback)(context);
  };
}

function benchmarkFailure(context: TestOutcomeContext): string {
  const error = context.error ?? "";
  const thresholdMatch = error.match(/avg reward ([0-9.]+) < ([0-9.]+) threshold/i);
  if (thresholdMatch) {
    return `avg reward ${thresholdMatch[1]} was below required threshold ${thresholdMatch[2]}`;
  }
  return clean(error) ?? "trained checkpoint did not pass the vf-eval benchmark";
}

function checkpointFailure(context: TestOutcomeContext): string {
  const error = context.error ?? "";
  const sizeMatch = error.match(/Checkpoint too small \(([0-9.]+)GB\)\. Expected > 1GB/i);
  if (sizeMatch) {
    return `checkpoint was ${sizeMatch[1]}GB, below the required 1.00GB minimum`;
  }
  return clean(error) ?? "reported final checkpoint was missing, unstable, or too small";
}

const NARRATIVES: Record<string, TestOutcomeNarrative> = {
  "aws-credentials": {
    pass: "AWS credentials were valid",
    fail: errorOr("AWS credentials were not valid for the required EKS and EFS workflow"),
  },
  "cli-tools-installed": {
    pass: "required AWS, Kubernetes, Helm, and EKS CLI tools were installed",
    fail: errorOr("one or more required AWS, Kubernetes, Helm, or EKS CLI tools were missing"),
  },
  "workspace-exists": {
    pass: "Prime-RL workspace and Helm chart were present",
    fail: errorOr("Prime-RL workspace or Helm chart was missing from the expected location"),
  },
  "cluster-active": {
    pass: "reusable EKS cluster was active",
    fail: errorOr("reusable EKS cluster was not active or accessible"),
  },
  "kubeconfig-valid": {
    pass: "kubectl could connect to the reusable EKS cluster",
    fail: errorOr("kubectl could not connect to the reusable EKS cluster"),
  },
  "cpu-nodegroup-active": {
    pass: "CPU nodegroup was active",
    fail: errorOr("CPU nodegroup was not active, so orchestration pods may not schedule"),
  },
  "gpu-nodegroup-active": {
    pass: "GPU nodegroup was active",
    fail: errorOr("GPU nodegroup was not active, so training or inference pods may not schedule"),
  },
  "all-nodes-ready": {
    pass: "all cluster nodes were Ready",
    fail: pollFailure("one or more cluster nodes were not Ready"),
  },
  "gpu-operator-installed": {
    pass: "NVIDIA GPU Operator was installed",
    fail: errorOr("NVIDIA GPU Operator was not installed"),
  },
  "gpu-operator-pods-running": {
    pass: "GPU Operator pods were running",
    fail: pollFailure("GPU Operator pods were not running"),
  },
  "gpus-allocatable": {
    pass: "cluster reported allocatable GPUs",
    fail: pollFailure("cluster did not report allocatable GPUs for Prime-RL pods"),
  },
  "efs-exists": {
    pass: outputOr("EFS filesystem existed for shared Prime-RL storage"),
    fail: errorOr("EFS filesystem for shared Prime-RL storage was not found"),
  },
  "efs-access-point-exists": {
    pass: outputOr("EFS access point existed with the expected POSIX identity"),
    fail: errorOr("EFS access point was missing or had the wrong POSIX identity"),
  },
  "efs-csi-driver-installed": {
    pass: outputOr("EFS CSI driver was installed"),
    fail: errorOr("EFS CSI driver was not installed, so shared volume mounts cannot work"),
  },
  "storage-class-exists": {
    pass: outputOr("EFS StorageClass existed with the expected provisioner"),
    fail: errorOr("EFS StorageClass was missing or used the wrong provisioner"),
  },
  "pv-exists": {
    pass: outputOr("Prime-RL EFS PersistentVolume existed"),
    fail: errorOr("Prime-RL EFS PersistentVolume was missing or misconfigured"),
  },
  "helm-release-exists": {
    pass: outputOr("Prime-RL Helm release existed"),
    fail: errorOr("Prime-RL Helm release was not found"),
  },
  "sanitized-runtime-image-used": {
    pass: outputOr("Prime-RL pods used the sanitized runtime image"),
    fail: errorOr("one or more Prime-RL pods did not use the sanitized runtime image"),
  },
  "pods-running": {
    pass: outputOr("all Prime-RL pods were Running"),
    fail: pollFailure("not all Prime-RL pods reached Running state"),
  },
  "services-exist": {
    pass: outputOr("required Prime-RL Kubernetes Services existed"),
    fail: errorOr("one or more required Prime-RL Kubernetes Services were missing"),
  },
  "pvc-bound": {
    pass: outputOr("shared-data PVC was Bound"),
    fail: pollFailure("shared-data PVC was not Bound to shared storage"),
  },
  "trainer-source-override-configured": {
    pass: outputOr("trainer StatefulSet mounted the agent-edited source ConfigMap"),
    fail: errorOr("trainer StatefulSet did not mount the agent-edited source ConfigMap"),
  },
  "deployed-trainer-source-loss-matches-workspace": {
    pass: outputOr("deployed loss.py matched the workspace implementation"),
    fail: errorOr("deployed loss.py did not match the workspace implementation"),
  },
  "deployed-trainer-source-data-matches-workspace": {
    pass: outputOr("deployed data.py matched the workspace implementation"),
    fail: errorOr("deployed data.py did not match the workspace implementation"),
  },
  "deployed-trainer-source-packer-matches-workspace": {
    pass: outputOr("deployed packer.py matched the workspace implementation"),
    fail: errorOr("deployed packer.py did not match the workspace implementation"),
  },
  "deployed-trainer-source-train-matches-workspace": {
    pass: outputOr("deployed train.py matched the workspace implementation"),
    fail: errorOr("deployed train.py did not match the workspace implementation"),
  },
  "deployed-transport-source-matches-workspace": {
    pass: outputOr("deployed transport source matched the workspace implementation"),
    fail: errorOr("deployed transport source did not match the workspace implementation"),
  },
  "deployed-broadcast-source-matches-workspace": {
    pass: outputOr("deployed weight-broadcast source matched the workspace implementation"),
    fail: errorOr("deployed weight-broadcast source did not match the workspace implementation"),
  },
  "deployed-sync-support-source-matches-workspace": {
    pass: outputOr("deployed parallel-dims and checkpoint source matched the workspace implementation"),
    fail: errorOr("deployed parallel-dims or checkpoint source did not match the workspace implementation"),
  },
  "deployed-trainer-source-not-stub": {
    pass: outputOr("deployed task source contained implemented concrete paths"),
    fail: afterPrefix(
      "Concrete task source still contains implementation stubs:",
      "deployed task source still contained concrete implementation stubs"
    ),
  },
  "loss-imports": {
    pass: outputOr("loss module imported with the required public functions"),
    fail: afterPrefix("Loss module import failed:", "loss module did not import with the required public functions"),
  },
  "shift-tensor-left": {
    pass: outputOr("shift_tensor_left produced the expected token alignment"),
    fail: probeFailure("shift_tensor_left produced incorrect token alignment"),
  },
  "shift-tensor-right": {
    pass: outputOr("shift_tensor_right produced the expected label alignment"),
    fail: probeFailure("shift_tensor_right produced incorrect label alignment"),
  },
  "selective-log-softmax-shape": {
    pass: outputOr("selective_log_softmax returned the expected tensor shape"),
    fail: probeFailure("selective_log_softmax returned an unexpected tensor shape"),
  },
  "compute-entropy-shape": {
    pass: outputOr("compute_entropy returned the expected tensor shape"),
    fail: probeFailure("compute_entropy returned an unexpected tensor shape"),
  },
  "data-loader-imports": {
    pass: outputOr("data loader module imported successfully"),
    fail: afterPrefix("Data module import failed:", "data loader module did not import successfully"),
  },
  "packer-imports": {
    pass: outputOr("packer module imported successfully"),
    fail: afterPrefix("Packer module import failed:", "packer module did not import successfully"),
  },
  "compute-loss-returns-tuple": {
    pass: outputOr("compute_loss accepted packed per-sequence tensors and returned scalar loss plus metrics"),
    fail: computeLossProbeFailure("compute_loss did not satisfy the packed sequence API contract"),
  },
  "loss-sum-not-mean": {
    pass: outputOr("GRPO loss aggregated eligible-token contributions"),
    fail: computeLossProbeFailure("GRPO loss did not aggregate eligible-token contributions correctly"),
  },
  "selective-log-softmax-accuracy": {
    pass: outputOr("selective_log_softmax matched the token log-probability test oracle"),
    fail: probeFailure("selective_log_softmax did not match the token log-probability test oracle"),
  },
  "shift-right-pad-value": {
    pass: outputOr("shift_tensor_right preserved the configured pad value"),
    fail: probeFailure("shift_tensor_right did not preserve the configured pad value"),
  },
  "temperature-before-logsoftmax": {
    pass: outputOr("temperature-aware log-softmax matched the expected token probabilities"),
    fail: probeFailure("temperature-aware log-softmax did not match expected token probabilities"),
  },
  "per-sequence-loss": {
    pass: outputOr("packed rollout sequences preserved their sequence boundaries"),
    fail: computeLossProbeFailure("packed rollout sequences did not preserve sequence-boundary loss behavior"),
  },
  "loss-scale-unmasked-tokens": {
    pass: outputOr("loss_scale normalized the total loss as expected"),
    fail: computeLossProbeFailure("loss_scale did not normalize the total loss correctly"),
  },
  "entropy-accuracy": {
    pass: outputOr("compute_entropy matched expected entropy values"),
    fail: probeFailure("compute_entropy did not match expected entropy values"),
  },
  "pg-loss-masking": {
    pass: outputOr("importance-ratio masking excluded extreme-ratio tokens"),
    fail: computeLossProbeFailure("importance-ratio masking did not exclude extreme-ratio tokens"),
  },
  "compute-loss-reference": {
    pass: outputOr("GRPO loss formula matched the test oracle values"),
    fail: computeLossProbeFailure("GRPO loss formula did not match the test oracle values"),
  },
  "weight-sync-import-surface": {
    pass: outputOr("weight-sync modules exported the required public symbols"),
    fail: probeFailure("weight-sync modules did not export the required public symbols"),
  },
  "weight-sync-factory-dispatch": {
    pass: outputOr("factories selected filesystem, ZMQ, and NCCL implementations by config type"),
    fail: probeFailure("weight-sync factories did not dispatch to the expected backend implementations"),
  },
  "filesystem-transport-roundtrip": {
    pass: outputOr("filesystem transport preserved training-batch and micro-batch contents"),
    fail: probeFailure("filesystem transport did not preserve training-batch or micro-batch contents"),
  },
  "zmq-transport-roundtrip": {
    pass: outputOr("ZMQ transport preserved training-batch and micro-batch contents"),
    fail: probeFailure("ZMQ transport did not preserve training-batch or micro-batch contents"),
  },
  "parallel-dims-validation": {
    pass: outputOr("parallel dims accepted valid dimensions and rejected invalid sequence partitioning"),
    fail: probeFailure("parallel dims validation did not match the expected distributed-shape rules"),
  },
  "checkpoint-broadcast-smoke": {
    pass: outputOr("multi-run checkpoint manager initialized and filesystem broadcast produced a stable checkpoint"),
    fail: probeFailure("checkpoint manager or filesystem broadcast did not produce a stable checkpoint"),
  },
  "rl-trainer-correctness": {
    pass: ({ score }) => `trainer algorithm rubric scored ${pct(score)}%, meeting required threshold 70%`,
    fail: ({ score }) => `trainer algorithm rubric scored ${pct(score)}%, below required threshold 70%`,
  },
  "weight-sync-implementation-quality": {
    pass: ({ score }) => `weight-sync implementation rubric scored ${pct(score)}%, meeting required threshold 70%`,
    fail: ({ score }) => `weight-sync implementation rubric scored ${pct(score)}%, below required threshold 70%`,
  },
  "process-quality": {
    pass: ({ score }) => `agent process quality scored ${pct(score)}%, meeting required threshold 60%`,
    fail: ({ score }) => `agent process quality scored ${pct(score)}%, below required threshold 60%`,
  },
  "inference-health": {
    pass: outputOr("inference service became healthy"),
    fail: pollFailure("inference service did not become healthy"),
  },
  "training-progress": {
    pass: outputOr("training showed step progress"),
    fail: pollFailure("training did not show step progress"),
  },
  "training-completes": {
    pass: outputOr("training completed the required 20 steps"),
    fail: pollFailure("training did not complete the required 20 steps"),
  },
  "checkpoint-saved": {
    pass: outputOr("final checkpoint marker and model-weight artifacts were present"),
    fail: pollFailure("final checkpoint marker or model-weight artifacts were not present"),
  },
  "model-benchmark": {
    pass: outputOr("trained checkpoint passed the vf-eval benchmark"),
    fail: benchmarkFailure,
  },
  "output-manifest-exists": {
    pass: outputOr("output manifest existed"),
    fail: errorOr("output manifest was not written"),
  },
  "output-manifest-valid": {
    pass: outputOr("output manifest matched the required schema"),
    fail: errorOr("output manifest did not match the required schema"),
  },
  "gpu-cluster-reachable": {
    pass: outputOr("manifest cluster was reachable"),
    fail: errorOr("manifest cluster was not reachable"),
  },
  "deployment-pods-healthy": {
    pass: outputOr("manifest deployment pods were healthy"),
    fail: errorOr("manifest deployment pods were not healthy"),
  },
  "training-checkpoint-valid": {
    pass: outputOr("reported final checkpoint was valid"),
    fail: checkpointFailure,
  },
};

export function attachOutcomeNarratives(tests: SimpleTest[]): SimpleTest[] {
  return tests.map((test) => ({
    ...test,
    outcomeNarrative: NARRATIVES[test.id],
  }));
}

export function missingOutcomeNarratives(tests: SimpleTest[]): string[] {
  return tests
    .map((test) => test.id)
    .filter((id) => !NARRATIVES[id]);
}

export function narrativeIds(): string[] {
  return Object.keys(NARRATIVES);
}
