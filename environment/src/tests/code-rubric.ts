import {
  Logger,
  SimpleTest,
  createRubricTest,
  createOpenRouterGenerateFn,
} from "@hyperfocal/env-base";
import { WORKSPACE_PATH } from "../config.js";
import * as fs from "fs";
import * as path from "path";

const SRC_DIR = path.join("src", "prime_rl", "trainer", "rl");
const CODE_FILES = ["loss.py", "data.py", "packer.py", "train.py"];
const WEIGHT_SYNC_CODE_FILES = [
  path.join("src", "prime_rl", "transport", "__init__.py"),
  path.join("src", "prime_rl", "transport", "config.py"),
  path.join("src", "prime_rl", "transport", "types.py"),
  path.join("src", "prime_rl", "transport", "filesystem.py"),
  path.join("src", "prime_rl", "transport", "zmq.py"),
  path.join("src", "prime_rl", "trainer", "rl", "broadcast", "__init__.py"),
  path.join("src", "prime_rl", "trainer", "rl", "broadcast", "filesystem.py"),
  path.join("src", "prime_rl", "trainer", "rl", "broadcast", "nccl.py"),
  path.join("src", "prime_rl", "trainer", "parallel_dims.py"),
  path.join("src", "prime_rl", "trainer", "multi_ckpt.py"),
];

function readWorkspaceCode(): string {
  return CODE_FILES.map((f) => {
    const p = path.join(WORKSPACE_PATH, SRC_DIR, f);
    try {
      return `=== ${f} ===\n${fs.readFileSync(p, "utf-8")}`;
    } catch {
      return `=== ${f} ===\n(file not found)`;
    }
  }).join("\n\n");
}

function readWeightSyncCode(): string {
  return WEIGHT_SYNC_CODE_FILES.map((file) => {
    const p = path.join(WORKSPACE_PATH, file);
    try {
      return `=== ${file} ===\n${fs.readFileSync(p, "utf-8")}`;
    } catch {
      return `=== ${file} ===\n(file not found)`;
    }
  }).join("\n\n");
}

export function createCodeRubricTests(): SimpleTest[] {
  return [
    createRubricTest({
      id: "rl-trainer-correctness",
      name: "RL Trainer Algorithmic Correctness",
      description: "Evaluates structural patterns in the GRPO trainer implementation that deterministic tests cannot verify",
      criteria: [
        {
          weight: 10,
          requirement: `The GRPO loss implementation handles packed variable-length
sequences correctly. compute_loss should iterate over per-sequence tensors,
preserve the sequence boundaries for trainer_logprobs, inference_logprobs,
teacher_logprobs, advantages, and loss_mask, and aggregate the loss across
eligible tokens without assuming a fixed rectangular batch layout.`,
          context: ["code"],
        },
        {
          weight: 8,
          requirement: `The GRPO policy-gradient loss correctly uses .detach() for gradient
routing: the loss coefficient (importance_ratio * advantages) is detached so
that only trainer_logprobs carries gradients. Concretely, the pattern should be
coeff.detach() * trainer_logprobs or equivalent stop-gradient, NOT
-log_ratio * advantages (which is a simplified policy gradient that computes
different gradients). A naive implementation without .detach() is UNMET even
if the forward loss values appear similar.`,
          context: ["code"],
        },
        {
          weight: 6,
          requirement: `The training loop in train.py follows correct lifecycle ordering:
forward pass, backward pass (loss.backward()), gradient clipping
(clip_grad_norm_), then optimizer.step(). Gradient clipping must happen AFTER
loss.backward() accumulates across all micro-batches and BEFORE
optimizer.step() updates weights. Gradients must be zeroed between batches
via optimizer.zero_grad() — either before the forward pass or after
optimizer.step() (both are standard PyTorch patterns).`,
          context: ["code"],
        },
        {
          weight: 5,
          requirement: `DataLoader and Packer preserve the rollout data contract from
the orchestrator. They should move input_ids, position_ids, advantages,
inference_logprobs, optional teacher_logprobs, loss_mask, temperature, and
lora_num_tokens into TensorMicroBatch objects without dropping fields, changing
sequence boundaries, or falling back to fake data in the real transport path.`,
          context: ["code"],
        },
        {
          weight: 5,
          requirement: `Teacher/KL terms are handled using the configured LossConfig
weights. Optional teacher_logprobs should only affect the loss when present and
enabled by teacher_tau, teacher signals should not receive gradients, and kl_tau
should penalize trainer/inference log-ratio drift without breaking the base
GRPO policy-gradient term.`,
          context: ["code"],
        },
        {
          weight: -4,
          requirement: `The loss function applies masking in probability space
rather than importance ratio space. For example, computing masks based on raw
probabilities or logprobs directly (mask = logprobs > threshold) instead of
computing the importance ratio first (ratio = exp(trainer_logprobs -
inference_logprobs)) and masking based on that ratio. The correct approach
computes token_importance_ratio = exp(log_importance_ratio) and then applies
thresholds from LossConfig (token_mask_high, token_mask_low, geo_mask_high,
geo_mask_low) to the ratio values.`,
          context: ["code"],
        },
      ],
      getContext: async (logger: Logger) => {
        logger.info("Reading agent's source files");
        const code = readWorkspaceCode();
        return { code };
      },
      generateFn: createOpenRouterGenerateFn({
        model: "openai/gpt-5.5",
        maxAttempts: 5,
      }),
      passThreshold: 0.7,
    }),
    createRubricTest({
      id: "weight-sync-implementation-quality",
      name: "Weight Sync Implementation Quality",
      description: "Evaluates whether transport, broadcast, and checkpoint logic forms a coherent distributed update path",
      criteria: [
        {
          weight: 8,
          requirement: `The transport factories dispatch correctly by config type
and the code makes it clear how filesystem and ZMQ backends are selected rather
than hardcoding one backend path.`,
          context: ["code"],
        },
        {
          weight: 8,
          requirement: `Filesystem and ZMQ transport paths preserve training-batch
and micro-batch boundaries, ordering semantics, and run identity. Good
implementations show clear serialization, receive-side ordering or buffering,
and independent per-run progress rather than ambiguous blob passing.`,
          context: ["code"],
        },
        {
          weight: 7,
          requirement: `Weight broadcast logic covers filesystem checkpoint
handoff and NCCL setup coherently. The trainer should produce stable
per-run checkpoint directories and mark them ready for inference without
silently skipping ready runs.`,
          context: ["code"],
        },
        {
          weight: 6,
          requirement: `Parallel dimension validation and multi-run checkpointing
are aware of world size, sequence divisibility, run identity, and per-run
state. The implementation should not collapse everything into a single global
happy path.`,
          context: ["code"],
        },
        {
          weight: -8,
          requirement: `The implementation only supports one narrow happy path or
preserves placeholder behavior. Examples include dispatching every config to
the same backend, ignoring run identity, leaving NotImplementedError in
task-owned concrete paths, or returning dummy objects that satisfy imports but
cannot move batches or mark weight checkpoints stable.`,
          context: ["code"],
        },
      ],
      getContext: async (logger: Logger) => {
        logger.info("Reading agent's weight-sync source files");
        return { code: readWeightSyncCode() };
      },
      generateFn: createOpenRouterGenerateFn({
        model: "openai/gpt-5.5",
        maxAttempts: 5,
      }),
      passThreshold: 0.7,
    }),
  ];
}
