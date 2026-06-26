import { Logger, SimpleTest, executeWithExitCode } from "@hyperfocal/env-base";
import {
  classifyAndAttributeError,
  getEnvWithAws,
  registerFailureCategory,
} from "../helpers.js";
import { RELEASE_NAME, WORKSPACE_PATH } from "../config.js";
import * as path from "path";
import * as fs from "fs";
import { createHash } from "crypto";

const TRAINER_SOURCE_CONFIGMAP = "prime-rl-trainer-source";
const TRAINER_SOURCE_DIR = "/app/src/prime_rl/trainer/rl";

interface SourceOverrideFile {
  key: string;
  workspaceRel: string;
  deployedPath: string;
  label: string;
}

const TRAINER_SOURCE_FILES: SourceOverrideFile[] = [
  {
    key: "loss.py",
    workspaceRel: path.join("src", "prime_rl", "trainer", "rl", "loss.py"),
    deployedPath: `${TRAINER_SOURCE_DIR}/loss.py`,
    label: "loss.py",
  },
  {
    key: "data.py",
    workspaceRel: path.join("src", "prime_rl", "trainer", "rl", "data.py"),
    deployedPath: `${TRAINER_SOURCE_DIR}/data.py`,
    label: "data.py",
  },
  {
    key: "packer.py",
    workspaceRel: path.join("src", "prime_rl", "trainer", "rl", "packer.py"),
    deployedPath: `${TRAINER_SOURCE_DIR}/packer.py`,
    label: "packer.py",
  },
  {
    key: "train.py",
    workspaceRel: path.join("src", "prime_rl", "trainer", "rl", "train.py"),
    deployedPath: `${TRAINER_SOURCE_DIR}/train.py`,
    label: "train.py",
  },
];

const TRANSPORT_SOURCE_FILES: SourceOverrideFile[] = [
  ["transport-init.py", "src/prime_rl/transport/__init__.py", "/app/src/prime_rl/transport/__init__.py"],
  ["transport-base.py", "src/prime_rl/transport/base.py", "/app/src/prime_rl/transport/base.py"],
  ["transport-config.py", "src/prime_rl/transport/config.py", "/app/src/prime_rl/transport/config.py"],
  ["transport-types.py", "src/prime_rl/transport/types.py", "/app/src/prime_rl/transport/types.py"],
  ["transport-filesystem.py", "src/prime_rl/transport/filesystem.py", "/app/src/prime_rl/transport/filesystem.py"],
  ["transport-zmq.py", "src/prime_rl/transport/zmq.py", "/app/src/prime_rl/transport/zmq.py"],
].map(([key, workspaceRel, deployedPath]) => ({
  key,
  workspaceRel: path.join(...workspaceRel.split("/")),
  deployedPath,
  label: workspaceRel,
}));

const BROADCAST_SOURCE_FILES: SourceOverrideFile[] = [
  ["broadcast-init.py", "src/prime_rl/trainer/rl/broadcast/__init__.py", "/app/src/prime_rl/trainer/rl/broadcast/__init__.py"],
  ["broadcast-base.py", "src/prime_rl/trainer/rl/broadcast/base.py", "/app/src/prime_rl/trainer/rl/broadcast/base.py"],
  ["broadcast-filesystem.py", "src/prime_rl/trainer/rl/broadcast/filesystem.py", "/app/src/prime_rl/trainer/rl/broadcast/filesystem.py"],
  ["broadcast-nccl.py", "src/prime_rl/trainer/rl/broadcast/nccl.py", "/app/src/prime_rl/trainer/rl/broadcast/nccl.py"],
].map(([key, workspaceRel, deployedPath]) => ({
  key,
  workspaceRel: path.join(...workspaceRel.split("/")),
  deployedPath,
  label: workspaceRel,
}));

const SUPPORT_SOURCE_FILES: SourceOverrideFile[] = [
  ["parallel-dims.py", "src/prime_rl/trainer/parallel_dims.py", "/app/src/prime_rl/trainer/parallel_dims.py"],
  ["multi-ckpt.py", "src/prime_rl/trainer/multi_ckpt.py", "/app/src/prime_rl/trainer/multi_ckpt.py"],
].map(([key, workspaceRel, deployedPath]) => ({
  key,
  workspaceRel: path.join(...workspaceRel.split("/")),
  deployedPath,
  label: workspaceRel,
}));

const SOURCE_OVERRIDE_FILES = [
  ...TRAINER_SOURCE_FILES,
  ...TRANSPORT_SOURCE_FILES,
  ...BROADCAST_SOURCE_FILES,
  ...SUPPORT_SOURCE_FILES,
];

const STUB_CHECK_FILES = [
  ...TRAINER_SOURCE_FILES,
  ...TRANSPORT_SOURCE_FILES.filter((file) => !file.workspaceRel.endsWith(`${path.sep}base.py`)),
  ...BROADCAST_SOURCE_FILES.filter((file) => !file.workspaceRel.endsWith(`${path.sep}base.py`)),
  ...SUPPORT_SOURCE_FILES,
];

function workspaceFilePath(file: SourceOverrideFile): string {
  return path.join(WORKSPACE_PATH, file.workspaceRel);
}

function sha256File(filePath: string): string {
  return createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

async function deployedSourceFileMatchesWorkspace(
  file: SourceOverrideFile,
  testId: string,
  logger: Logger
): Promise<{ success: boolean; error?: string; output?: string }> {
  const env = getEnvWithAws();
  const srcFile = workspaceFilePath(file);
  if (!fs.existsSync(srcFile)) {
    return {
      success: false,
      error: `Workspace task source file is missing: ${srcFile}`,
    };
  }

  const localHash = sha256File(srcFile);
  const podResult = await executeWithExitCode(
    `kubectl exec ${RELEASE_NAME}-trainer-0 -- sha256sum ${file.deployedPath}`,
    { env, timeout: 30000 }
  );

  if (!podResult.success) {
    return {
      success: false,
      error: classifyAndAttributeError(
        `Could not hash deployed task source ${file.label}: ${podResult.output}`,
        testId,
        logger
      ),
    };
  }

  const deployedHash = podResult.output.trim().split(/\s+/)[0];
  if (deployedHash !== localHash) {
    return {
      success: false,
      error: classifyAndAttributeError(
        `Trainer pod ${file.label} does not match workspace source: workspace ${localHash} != deployed ${deployedHash}`,
        testId,
        logger
      ),
    };
  }

  const message = `trainer pod ${file.label} matched workspace source hash ${localHash.slice(0, 12)}`;
  logger.info(message);
  return { success: true, output: message };
}

async function deployedSourceGroupMatchesWorkspace(
  files: SourceOverrideFile[],
  testId: string,
  logger: Logger
): Promise<{ success: boolean; error?: string; output?: string }> {
  const matched: string[] = [];
  for (const file of files) {
    const result = await deployedSourceFileMatchesWorkspace(file, testId, logger);
    if (!result.success) {
      return result;
    }
    matched.push(file.label);
  }
  return {
    success: true,
    output: `${matched.length} deployed task source file(s) matched workspace: ${matched.join(", ")}`,
  };
}

export function createTrainerUnitTests(): SimpleTest[] {
  return [
    {
      id: "trainer-source-override-configured",
      name: "Trainer Source Override Configured",
      description: "Verify the trainer StatefulSet mounts the agent-edited source ConfigMap",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();
        const testId = "trainer-source-override-configured";
        const result = await executeWithExitCode(
          `kubectl get statefulset ${RELEASE_NAME}-trainer -o json`,
          { env, timeout: 30000 }
        );

        if (!result.success) {
          return {
            success: false,
            error: classifyAndAttributeError(
              `Trainer StatefulSet not found: ${result.output}`,
              testId,
              logger
            ),
          };
        }

        try {
          const statefulSet = JSON.parse(result.output);
          const volumes = statefulSet.spec?.template?.spec?.volumes ?? [];
          const containers = statefulSet.spec?.template?.spec?.containers ?? [];
          const sourceVolume = volumes.find(
            (volume: { configMap?: { name?: string } }) =>
              volume.configMap?.name === TRAINER_SOURCE_CONFIGMAP
          );
          const mounted = containers.some((container: { volumeMounts?: { name?: string; mountPath?: string }[] }) =>
            (container.volumeMounts ?? []).some(
              (mount) => mount.name === sourceVolume?.name && mount.mountPath === "/workspace-source"
            )
          );

          if (!sourceVolume || !mounted) {
            return {
              success: false,
              error: registerFailureCategory("sourceoverride-disabled", testId, logger),
            };
          }

          const configMapResult = await executeWithExitCode(
            `kubectl get configmap ${TRAINER_SOURCE_CONFIGMAP} -o json`,
            { env, timeout: 30000 }
          );

          if (!configMapResult.success) {
            return {
              success: false,
              error: classifyAndAttributeError(
                `Trainer source ConfigMap ${TRAINER_SOURCE_CONFIGMAP} was not readable: ${configMapResult.output}`,
                testId,
                logger
              ),
            };
          }

          const configMap = JSON.parse(configMapResult.output);
          const keys = new Set(Object.keys(configMap.data ?? {}));
          const missingKeys = SOURCE_OVERRIDE_FILES
            .map((file) => file.key)
            .filter((key) => !keys.has(key));

          if (missingKeys.length > 0) {
            return {
              success: false,
              error: `Trainer source ConfigMap is missing task-owned file key(s): ${missingKeys.join(", ")}`,
            };
          }

          const message = `trainer mounted ConfigMap ${TRAINER_SOURCE_CONFIGMAP} with ${SOURCE_OVERRIDE_FILES.length} task-owned source files`;
          logger.info(message);
          return { success: true, output: message };
        } catch (error) {
          return {
            success: false,
            error: `Failed to inspect trainer StatefulSet: ${error}`,
          };
        }
      },
    },
    ...TRAINER_SOURCE_FILES.map((file) => {
      const testId = `deployed-trainer-source-${path.basename(file.key, ".py")}-matches-workspace`;
      return {
        id: testId,
        name: `Deployed ${file.key} Matches Workspace`,
        description: `Verify the trainer pod is running the exact ${file.key} source edited in the workspace`,
        run: async (logger: Logger) => {
          return deployedSourceFileMatchesWorkspace(file, testId, logger);
        },
      };
    }),
    {
      id: "deployed-transport-source-matches-workspace",
      name: "Deployed Transport Source Matches Workspace",
      description: "Verify the trainer pod is running the workspace transport source",
      run: async (logger: Logger) =>
        deployedSourceGroupMatchesWorkspace(
          TRANSPORT_SOURCE_FILES,
          "deployed-transport-source-matches-workspace",
          logger
        ),
    },
    {
      id: "deployed-broadcast-source-matches-workspace",
      name: "Deployed Broadcast Source Matches Workspace",
      description: "Verify the trainer pod is running the workspace weight-broadcast source",
      run: async (logger: Logger) =>
        deployedSourceGroupMatchesWorkspace(
          BROADCAST_SOURCE_FILES,
          "deployed-broadcast-source-matches-workspace",
          logger
        ),
    },
    {
      id: "deployed-sync-support-source-matches-workspace",
      name: "Deployed Sync Support Source Matches Workspace",
      description: "Verify the trainer pod is running workspace parallel-dims and checkpoint source",
      run: async (logger: Logger) =>
        deployedSourceGroupMatchesWorkspace(
          SUPPORT_SOURCE_FILES,
          "deployed-sync-support-source-matches-workspace",
          logger
        ),
    },
    {
      id: "deployed-trainer-source-not-stub",
      name: "Deployed Trainer Source Is Implemented",
      description: "Verify deployed trainer source no longer contains concrete implementation stubs",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();
        const files = JSON.stringify(STUB_CHECK_FILES.map((file) => file.deployedPath));
        const result = await executeWithExitCode(
          `kubectl exec ${RELEASE_NAME}-trainer-0 -- python3 - <<'PY'
import ast
from pathlib import Path

files = ${files}


def decorator_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = decorator_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Call):
        return decorator_name(node.func)
    return ""


def is_abstract_function(node):
    return any(decorator_name(decorator).endswith("abstractmethod") for decorator in node.decorator_list)


def non_docstring_statements(body):
    return [
        stmt for stmt in body
        if not (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )
    ]


def is_not_implemented_raise(stmt):
    if not isinstance(stmt, ast.Raise) or stmt.exc is None:
        return False
    exc = stmt.exc
    if isinstance(exc, ast.Call):
        exc = exc.func
    if isinstance(exc, ast.Name):
        return exc.id == "NotImplementedError"
    if isinstance(exc, ast.Attribute):
        return exc.attr == "NotImplementedError"
    return False


def is_stub_body(body):
    statements = non_docstring_statements(body)
    if not statements:
        return True
    if len(statements) != 1:
        return False
    stmt = statements[0]
    if isinstance(stmt, ast.Pass):
        return True
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is Ellipsis:
        return True
    return is_not_implemented_raise(stmt)


problems = []
for file in files:
    source = Path(file).read_text()
    tree = ast.parse(source, filename=file)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if is_abstract_function(node):
            continue
        if is_stub_body(node.body):
            problems.append(f"{file}:{node.lineno}: {node.name}")

if problems:
    print("FAIL: concrete task stubs remain: " + "; ".join(problems[:12]))
    raise SystemExit(1)

print("PASS: deployed task source has no concrete implementation stubs")
PY`,
          { env, timeout: 30000 }
        );

        if (!result.success) {
          return {
            success: false,
            error: classifyAndAttributeError(
              `Concrete task source still contains implementation stubs:\n${result.output}`,
              "deployed-trainer-source-not-stub",
              logger
            ),
          };
        }

        const message = "deployed task source had no concrete implementation stubs";
        logger.info(message);
        return { success: true, output: message };
      },
    },
    {
      id: "loss-imports",
      name: "Loss Module Imports",
      description: "Verify loss.py exports required functions",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();

        const result = await executeWithExitCode(
          `kubectl exec ${RELEASE_NAME}-trainer-0 -- python3 -c "
from prime_rl.trainer.rl.loss import (
    selective_log_softmax,
    compute_entropy,
    shift_tensor_left,
    shift_tensor_right,
    compute_loss,
)
print('All imports successful')
"`,
          { env, timeout: 30000 }
        );

        if (!result.success || !result.output.includes("All imports successful")) {
          return {
            success: false,
            error: classifyAndAttributeError(
              `Loss module import failed: ${result.output}`,
              "loss-imports",
              logger
            ),
          };
        }

        const message = "loss module exported selective_log_softmax, compute_entropy, shifts, and compute_loss";
        logger.info(message);
        return { success: true, output: message };
      },
    },
    {
      id: "shift-tensor-left",
      name: "Shift Tensor Left Correctness",
      description: "Verify shift_tensor_left shifts data correctly",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();

        const result = await executeWithExitCode(
          `kubectl exec ${RELEASE_NAME}-trainer-0 -- python3 -c "
import torch
from prime_rl.trainer.rl.loss import shift_tensor_left

# Test: [1, 2, 3, 4] -> [2, 3, 4, 0]
t = torch.tensor([[1, 2, 3, 4]])
shifted = shift_tensor_left(t)
expected = torch.tensor([[2, 3, 4, 0]])

if not torch.equal(shifted, expected):
    print(f'FAIL: Expected {expected.tolist()}, got {shifted.tolist()}')
    exit(1)

# Test batch dimension preserved
t2 = torch.tensor([[1, 2], [3, 4]])
shifted2 = shift_tensor_left(t2)
if shifted2.shape != (2, 2):
    print(f'FAIL: batch shape should stay (2, 2), got {tuple(shifted2.shape)}')
    exit(1)

print('PASS')
"`,
          { env, timeout: 30000 }
        );

        if (!result.success || !result.output.includes("PASS")) {
          return {
            success: false,
            error: classifyAndAttributeError(
              `shift_tensor_left test failed: ${result.output}`,
              "shift-tensor-left",
              logger
            ),
          };
        }

        const message = "shift_tensor_left transformed [1,2,3,4] into [2,3,4,0]";
        logger.info(message);
        return { success: true, output: message };
      },
    },
    {
      id: "shift-tensor-right",
      name: "Shift Tensor Right Correctness",
      description: "Verify shift_tensor_right shifts data correctly",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();

        const result = await executeWithExitCode(
          `kubectl exec ${RELEASE_NAME}-trainer-0 -- python3 -c "
import torch
from prime_rl.trainer.rl.loss import shift_tensor_right

# Test: [1, 2, 3, 4] -> [0, 1, 2, 3]
t = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
shifted = shift_tensor_right(t)
expected = torch.tensor([[0.0, 1.0, 2.0, 3.0]])

if not torch.allclose(shifted, expected):
    print(f'FAIL: Expected {expected.tolist()}, got {shifted.tolist()}')
    exit(1)

# Test with custom pad value
shifted_pad = shift_tensor_right(t, pad_value=-1.0)
expected_pad = torch.tensor([[-1.0, 1.0, 2.0, 3.0]])

if not torch.allclose(shifted_pad, expected_pad):
    print(f'FAIL with pad_value: Expected {expected_pad.tolist()}, got {shifted_pad.tolist()}')
    exit(1)

print('PASS')
"`,
          { env, timeout: 30000 }
        );

        if (!result.success || !result.output.includes("PASS")) {
          return {
            success: false,
            error: classifyAndAttributeError(
              `shift_tensor_right test failed: ${result.output}`,
              "shift-tensor-right",
              logger
            ),
          };
        }

        const message = "shift_tensor_right shifted values and honored the custom pad value";
        logger.info(message);
        return { success: true, output: message };
      },
    },
    {
      id: "selective-log-softmax-shape",
      name: "Selective Log Softmax Shape",
      description: "Verify selective_log_softmax output shape",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();

        const result = await executeWithExitCode(
          `kubectl exec ${RELEASE_NAME}-trainer-0 -- python3 -c "
import torch
from prime_rl.trainer.rl.loss import selective_log_softmax

# Input: (batch=2, seq=4, vocab=100)
logits = torch.randn(2, 4, 100)
indices = torch.randint(0, 100, (2, 4))

output = selective_log_softmax(logits, indices)

# Output should be (batch=2, seq=4) - one logprob per position
if output.shape != (2, 4):
    print(f'FAIL: Expected shape (2, 4), got {output.shape}')
    exit(1)

# Values should be negative (log probabilities)
if (output > 0).any():
    print(f'FAIL: Log probabilities should be <= 0, got max {output.max().item()}')
    exit(1)

print('PASS')
"`,
          { env, timeout: 30000 }
        );

        if (!result.success || !result.output.includes("PASS")) {
          return {
            success: false,
            error: classifyAndAttributeError(
              `selective_log_softmax test failed: ${result.output}`,
              "selective-log-softmax-shape",
              logger
            ),
          };
        }

        const message = "selective_log_softmax returned shape (2,4) with non-positive log probabilities";
        logger.info(message);
        return { success: true, output: message };
      },
    },
    {
      id: "compute-entropy-shape",
      name: "Compute Entropy Shape",
      description: "Verify compute_entropy output shape and range",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();

        const result = await executeWithExitCode(
          `kubectl exec ${RELEASE_NAME}-trainer-0 -- python3 -c "
import torch
from prime_rl.trainer.rl.loss import compute_entropy

# Input: (batch=2, seq=4, vocab=100)
logits = torch.randn(2, 4, 100)

entropy = compute_entropy(logits)

# Output should be (batch=2, seq=4)
if entropy.shape != (2, 4):
    print(f'FAIL: Expected shape (2, 4), got {entropy.shape}')
    exit(1)

# Entropy should be non-negative
if (entropy < 0).any():
    print(f'FAIL: Entropy should be >= 0, got min {entropy.min().item()}')
    exit(1)

print('PASS')
"`,
          { env, timeout: 30000 }
        );

        if (!result.success || !result.output.includes("PASS")) {
          return {
            success: false,
            error: classifyAndAttributeError(
              `compute_entropy test failed: ${result.output}`,
              "compute-entropy-shape",
              logger
            ),
          };
        }

        const message = "compute_entropy returned shape (2,4) with non-negative values";
        logger.info(message);
        return { success: true, output: message };
      },
    },
    {
      id: "data-loader-imports",
      name: "Data Loader Module Imports",
      description: "Verify data.py exports required classes",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();

        const result = await executeWithExitCode(
          `kubectl exec ${RELEASE_NAME}-trainer-0 -- python3 -c "
from prime_rl.trainer.rl.data import (
    TensorMicroBatch,
    FakeDataLoader,
    DataLoader,
)
print('All imports successful')
"`,
          { env, timeout: 30000 }
        );

        if (!result.success || !result.output.includes("All imports successful")) {
          return {
            success: false,
            error: classifyAndAttributeError(
              `Data module import failed: ${result.output}`,
              "data-loader-imports",
              logger
            ),
          };
        }

        const message = "data module exported TensorMicroBatch, FakeDataLoader, and DataLoader";
        logger.info(message);
        return { success: true, output: message };
      },
    },
    {
      id: "packer-imports",
      name: "Packer Module Imports",
      description: "Verify packer.py exports required classes",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();

        const result = await executeWithExitCode(
          `kubectl exec ${RELEASE_NAME}-trainer-0 -- python3 -c "
from prime_rl.trainer.rl.packer import (
    BasePacker,
    SinglePacker,
    MultiPacker,
    setup_packer,
)
print('All imports successful')
"`,
          { env, timeout: 30000 }
        );

        if (!result.success || !result.output.includes("All imports successful")) {
          return {
            success: false,
            error: classifyAndAttributeError(
              `Packer module import failed: ${result.output}`,
              "packer-imports",
              logger
            ),
          };
        }

        const message = "packer module exported BasePacker, SinglePacker, MultiPacker, and setup_packer";
        logger.info(message);
        return { success: true, output: message };
      },
    },
    {
      id: "compute-loss-returns-tuple",
      name: "Compute Loss Packed Sequence API",
      description: "Verify compute_loss accepts packed per-sequence tensor lists and returns loss plus metrics",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();

        const result = await executeWithExitCode(
          `kubectl exec ${RELEASE_NAME}-trainer-0 -- python3 -c "
import torch
from prime_rl.trainer.rl.loss import compute_loss
from prime_rl.trainer.rl.config import LossConfig

# Create simple test data as lists of tensors (packed sequences)
trainer_logprobs = [torch.randn(10)]  # 1 sequence of 10 tokens
inference_logprobs = [torch.randn(10)]
advantages = [torch.randn(10)]
loss_mask = [torch.ones(10, dtype=torch.bool)]

loss_config = LossConfig()

result = compute_loss(
    trainer_logprobs=trainer_logprobs,
    inference_logprobs=inference_logprobs,
    teacher_logprobs=None,
    advantages=advantages,
    loss_mask=loss_mask,
    loss_config=loss_config,
    loss_scale=1,
)

# Should support the public trainer API: loss, metrics = compute_loss(...)
try:
    loss, metrics = result
except Exception as exc:
    print(f'FAIL: compute_loss should return two values: scalar loss and metrics dict; got {type(result).__name__}: {exc}')
    exit(1)

if not isinstance(loss, torch.Tensor):
    print(f'FAIL: compute_loss returned loss as {type(loss).__name__}, expected torch.Tensor')
    exit(1)

if not isinstance(metrics, dict):
    print(f'FAIL: compute_loss returned metrics as {type(metrics).__name__}, expected dict')
    exit(1)

# Loss should be scalar
if loss.dim() != 0:
    print(f'FAIL: compute_loss returned non-scalar loss with shape {tuple(loss.shape)}')
    exit(1)

print('PASS')
"`,
          { env, timeout: 60000 }
        );

        if (!result.success || !result.output.includes("PASS")) {
          return {
            success: false,
            error: classifyAndAttributeError(
              `compute_loss test failed: ${result.output}`,
              "compute-loss-returns-tuple",
              logger
            ),
          };
        }

        const message = "compute_loss accepted packed per-sequence tensors and returned scalar loss plus metrics";
        logger.info(message);
        return { success: true, output: message };
      },
    },
    {
      id: "loss-sum-not-mean",
      name: "Loss Aggregates Eligible Tokens",
      description: "Verify duplicated eligible-token contributions double the total loss",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();
        const result = await executeWithExitCode(
          `kubectl exec ${RELEASE_NAME}-trainer-0 -- python3 -c "
import torch
from prime_rl.trainer.rl.loss import compute_loss
from prime_rl.trainer.rl.config import LossConfig

torch.manual_seed(42)
t_lp = torch.randn(10) * 0.5
i_lp = torch.randn(10) * 0.5
adv = torch.ones(10)
mask = torch.ones(10, dtype=torch.bool)
cfg = LossConfig()

loss1, _ = compute_loss([t_lp], [i_lp], None, [adv], [mask], cfg, 1)
loss2, _ = compute_loss([t_lp, t_lp], [i_lp, i_lp], None, [adv, adv], [mask, mask], cfg, 1)

ratio = loss2.item() / loss1.item() if abs(loss1.item()) > 1e-10 else float('inf')
if not (1.9 < ratio < 2.1):
    print(f'FAIL: duplicating the same sequence should double total loss; observed ratio {ratio:.4f} instead of 2.0')
    exit(1)
print('PASS')
"`,
          { env, timeout: 60000 }
        );

        if (!result.success || !result.output.includes("PASS")) {
          return {
            success: false,
            error: classifyAndAttributeError(
              `loss-sum-not-mean failed: ${result.output}`,
              "loss-sum-not-mean",
              logger
            ),
          };
        }
        const message = "two identical sequences produced approximately 2x loss";
        logger.info(message);
        return { success: true, output: message };
      },
    },
    {
      id: "selective-log-softmax-accuracy",
      name: "Selective Log Softmax Accuracy",
      description: "Verify selective_log_softmax matches the torch log-softmax test oracle",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();
        const result = await executeWithExitCode(
          `kubectl exec ${RELEASE_NAME}-trainer-0 -- python3 -c "
import torch
import torch.nn.functional as F
from prime_rl.trainer.rl.loss import selective_log_softmax

torch.manual_seed(42)
logits = torch.randn(2, 8, 50)
indices = torch.randint(0, 50, (2, 8))

result = selective_log_softmax(logits, indices)
expected = F.log_softmax(logits, dim=-1).gather(-1, indices.unsqueeze(-1)).squeeze(-1)

if not torch.allclose(result, expected, atol=1e-5):
    max_diff = (result - expected).abs().max().item()
    print(f'FAIL: max diff from the torch log-softmax test oracle was {max_diff}')
    exit(1)
print('PASS')
"`,
          { env, timeout: 30000 }
        );

        if (!result.success || !result.output.includes("PASS")) {
          return {
            success: false,
            error: classifyAndAttributeError(
              `selective-log-softmax-accuracy failed: ${result.output}`,
              "selective-log-softmax-accuracy",
              logger
            ),
          };
        }
        const message = "selective_log_softmax matched the torch log-softmax test oracle within tolerance";
        logger.info(message);
        return { success: true, output: message };
      },
    },
    {
      id: "shift-right-pad-value",
      name: "Shift Right Pad Value",
      description: "Verify shift_tensor_right handles pad_value correctly",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();
        const result = await executeWithExitCode(
          `kubectl exec ${RELEASE_NAME}-trainer-0 -- python3 -c "
import torch, math
from prime_rl.trainer.rl.loss import shift_tensor_right

t = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
shifted = shift_tensor_right(t)
if shifted[0, 0].item() != 0.0:
    print(f'FAIL: default pad should be 0.0, got {shifted[0,0].item()}')
    exit(1)

pad_val = math.log(1.0 / 32000)
shifted2 = shift_tensor_right(t, pad_value=pad_val)
if abs(shifted2[0, 0].item() - pad_val) > 1e-5:
    print(f'FAIL: pad_value not applied correctly')
    exit(1)

if not torch.allclose(shifted[0, 1:], t[0, :-1]):
    print('FAIL: shifted values incorrect')
    exit(1)
print('PASS')
"`,
          { env, timeout: 30000 }
        );

        if (!result.success || !result.output.includes("PASS")) {
          return {
            success: false,
            error: classifyAndAttributeError(
              `shift-right-pad-value failed: ${result.output}`,
              "shift-right-pad-value",
              logger
            ),
          };
        }
        const message = "shift_tensor_right preserved default and custom pad values";
        logger.info(message);
        return { success: true, output: message };
      },
    },
    {
      id: "temperature-before-logsoftmax",
      name: "Temperature-Aware Log Softmax",
      description: "Verify selective_log_softmax preserves pre-applied temperature scaling",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();
        const result = await executeWithExitCode(
          `kubectl exec ${RELEASE_NAME}-trainer-0 -- python3 -c "
import torch, torch.nn.functional as F
from prime_rl.trainer.rl.loss import selective_log_softmax

torch.manual_seed(42)
logits = torch.randn(1, 4, 20)
indices = torch.randint(0, 20, (1, 4))
T = 2.0

correct = selective_log_softmax(logits / T, indices)
wrong = selective_log_softmax(logits, indices) / T

if torch.allclose(correct, wrong, atol=1e-3):
    print('FAIL: temperature-scaled logits behaved like post-hoc scaling instead of pre-softmax scaling')
    exit(1)

ref = F.log_softmax(logits / T, dim=-1).gather(-1, indices.unsqueeze(-1)).squeeze(-1)
if not torch.allclose(correct, ref, atol=1e-5):
    print('FAIL: log probabilities did not match the test oracle when logits were pre-scaled by temperature')
    exit(1)
print('PASS')
"`,
          { env, timeout: 30000 }
        );

        if (!result.success || !result.output.includes("PASS")) {
          return {
            success: false,
            error: classifyAndAttributeError(
              `temperature-before-logsoftmax failed: ${result.output}`,
              "temperature-before-logsoftmax",
              logger
            ),
          };
        }
        const message = "pre-scaled logits matched the temperature-aware test oracle";
        logger.info(message);
        return { success: true, output: message };
      },
    },
    {
      id: "per-sequence-loss",
      name: "Packed Sequence Boundary Preservation",
      description: "Verify packed sequences preserve boundaries when loss is aggregated",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();
        const result = await executeWithExitCode(
          `kubectl exec ${RELEASE_NAME}-trainer-0 -- python3 -c "
import torch
from prime_rl.trainer.rl.loss import compute_loss
from prime_rl.trainer.rl.config import LossConfig

torch.manual_seed(42)
cfg = LossConfig()

t_a = torch.randn(5) * 0.3
i_a = torch.randn(5) * 0.3
adv_a = torch.ones(5)
mask_a = torch.ones(5, dtype=torch.bool)

t_b = torch.randn(5) * 0.5
i_b = torch.randn(5) * 0.5
adv_b = torch.ones(5) * 2.0
mask_b = torch.ones(5, dtype=torch.bool)

loss_a, _ = compute_loss([t_a], [i_a], None, [adv_a], [mask_a], cfg, 1)
loss_b, _ = compute_loss([t_b], [i_b], None, [adv_b], [mask_b], cfg, 1)
loss_ab, _ = compute_loss([t_a, t_b], [i_a, i_b], None, [adv_a, adv_b], [mask_a, mask_b], cfg, 1)

expected = loss_a.item() + loss_b.item()
actual = loss_ab.item()
if abs(expected - actual) > 1e-4:
    print(f'FAIL: combined packed loss was {actual:.6f}, but separate sequence losses summed to {expected:.6f}')
    exit(1)
print('PASS')
"`,
          { env, timeout: 60000 }
        );

        if (!result.success || !result.output.includes("PASS")) {
          return {
            success: false,
            error: classifyAndAttributeError(
              `per-sequence-loss failed: ${result.output}`,
              "per-sequence-loss",
              logger
            ),
          };
        }
        const message = "combined packed-sequence loss equaled the sum of individual sequence losses";
        logger.info(message);
        return { success: true, output: message };
      },
    },
    {
      id: "loss-scale-unmasked-tokens",
      name: "Loss Scale Normalization",
      description: "Verify loss_scale normalizes the total loss",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();
        const result = await executeWithExitCode(
          `kubectl exec ${RELEASE_NAME}-trainer-0 -- python3 -c "
import torch
from prime_rl.trainer.rl.loss import compute_loss
from prime_rl.trainer.rl.config import LossConfig

torch.manual_seed(42)
t_lp = torch.randn(10) * 0.5
i_lp = torch.randn(10) * 0.5
adv = torch.ones(10)
mask = torch.ones(10, dtype=torch.bool)
cfg = LossConfig()

loss1, _ = compute_loss([t_lp], [i_lp], None, [adv], [mask], cfg, 1)
loss2, _ = compute_loss([t_lp], [i_lp], None, [adv], [mask], cfg, 2)

ratio = loss2.item() / loss1.item() if abs(loss1.item()) > 1e-10 else float('inf')
if abs(ratio - 0.5) > 0.01:
    print(f'FAIL: loss_scale=2 should halve the loss; observed ratio {ratio:.4f}')
    exit(1)
print('PASS')
"`,
          { env, timeout: 60000 }
        );

        if (!result.success || !result.output.includes("PASS")) {
          return {
            success: false,
            error: classifyAndAttributeError(
              `loss-scale-unmasked-tokens failed: ${result.output}`,
              "loss-scale-unmasked-tokens",
              logger
            ),
          };
        }
        const message = "loss_scale=2 approximately halved the computed loss";
        logger.info(message);
        return { success: true, output: message };
      },
    },
    {
      id: "entropy-accuracy",
      name: "Compute Entropy Accuracy",
      description: "Verify entropy values for known distributions (uniform and peaked)",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();
        const result = await executeWithExitCode(
          `kubectl exec ${RELEASE_NAME}-trainer-0 -- python3 -c "
import torch, math
from prime_rl.trainer.rl.loss import compute_entropy

vocab = 100
logits = torch.zeros(1, 1, vocab)
entropy = compute_entropy(logits)
expected = math.log(vocab)
if abs(entropy[0, 0].item() - expected) > 0.01:
    print(f'FAIL: uniform entropy should be {expected:.4f}, got {entropy[0,0].item():.4f}')
    exit(1)

peaked = torch.full((1, 1, vocab), -100.0)
peaked[0, 0, 0] = 100.0
ent_peaked = compute_entropy(peaked)
if ent_peaked[0, 0].item() > 0.1:
    print(f'FAIL: peaked entropy should be ~0, got {ent_peaked[0,0].item():.4f}')
    exit(1)
print('PASS')
"`,
          { env, timeout: 30000 }
        );

        if (!result.success || !result.output.includes("PASS")) {
          return {
            success: false,
            error: classifyAndAttributeError(
              `entropy-accuracy failed: ${result.output}`,
              "entropy-accuracy",
              logger
            ),
          };
        }
        const message = "entropy matched uniform and peaked distribution expectations";
        logger.info(message);
        return { success: true, output: message };
      },
    },
    {
      id: "pg-loss-masking",
      name: "Importance Ratio Masking",
      description: "Verify importance-ratio masking excludes extreme-ratio tokens from loss",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();
        const result = await executeWithExitCode(
          `kubectl exec ${RELEASE_NAME}-trainer-0 -- python3 -c "
import torch
from prime_rl.trainer.rl.loss import compute_loss
from prime_rl.trainer.rl.config import LossConfig

cfg = LossConfig()
# Token 0: log_ratio=2.2, ratio=exp(2.2)=9.03 > token_mask_high(8.0) -> masked
# Token 1: log_ratio=0.1, ratio=exp(0.1)=1.105 -> between 0.125 and 8.0, not masked
# geo_seq_ratio = exp(mean([2.2, 0.1])) = exp(1.15) = 3.16 -> between 0.1 and 10.0, OK
# seq_max = 9.03 < 100, seq_min = 1.105 > 0 -> no sequence masking
t_lp = torch.tensor([0.0, -1.0])
i_lp = torch.tensor([-2.2, -1.1])

loss1, _ = compute_loss([t_lp], [i_lp], None, [torch.tensor([100.0, 1.0])], [torch.ones(2, dtype=torch.bool)], cfg, 1)
loss2, _ = compute_loss([t_lp], [i_lp], None, [torch.tensor([0.0, 1.0])], [torch.ones(2, dtype=torch.bool)], cfg, 1)

diff = abs(loss1.item() - loss2.item())
if diff > 0.01:
    print(f'FAIL: changing a masked token changed the loss by {diff:.4f}, so extreme-ratio tokens were not excluded')
    exit(1)
print('PASS')
"`,
          { env, timeout: 60000 }
        );

        if (!result.success || !result.output.includes("PASS")) {
          return {
            success: false,
            error: classifyAndAttributeError(
              `pg-loss-masking failed: ${result.output}`,
              "pg-loss-masking",
              logger
            ),
          };
        }
        const message = "changing a masked token advantage did not change the policy-gradient loss";
        logger.info(message);
        return { success: true, output: message };
      },
    },
    {
      id: "compute-loss-reference",
      name: "GRPO Loss Formula",
      description: "Verify compute_loss matches the hand-computed GRPO formula",
      run: async (logger: Logger) => {
        const env = getEnvWithAws();
        const result = await executeWithExitCode(
          `kubectl exec ${RELEASE_NAME}-trainer-0 -- python3 -c "
import torch
from prime_rl.trainer.rl.loss import compute_loss
from prime_rl.trainer.rl.config import LossConfig

cfg = LossConfig()
# Small log-ratio differences so no masking triggers
# token_mask bounds: [0.125, 8.0], ratios ~[0.905, 1.105]
t_lp = torch.tensor([-1.0, -2.0, -0.5, -1.5])
i_lp = torch.tensor([-1.1, -1.9, -0.6, -1.4])
adv = torch.tensor([1.0, -0.5, 2.0, 0.0])
mask = torch.tensor([True, True, True, False])

loss, metrics = compute_loss([t_lp], [i_lp], None, [adv], [mask], cfg, 1)

# Reference: REINFORCE with importance sampling
# With kl_tau=0.0 and adv_tau=1.0 (defaults):
# coeff = importance_ratio * advantages
# loss = -(coeff.detach() * trainer_logprobs)[keep_mask].sum()
log_ratio = t_lp - i_lp
ratio = torch.exp(log_ratio)
# All ratios between 0.125 and 8.0, no geo/seq masking -> keep_mask = mask
coeff = ratio * (cfg.adv_tau * adv - cfg.kl_tau * log_ratio)
expected = -(coeff.detach() * t_lp)[mask].sum().item()

if abs(loss.item() - expected) > 0.01:
    print(f'FAIL: computed loss was {loss.item():.6f}, expected {expected:.6f} from the test oracle GRPO formula')
    exit(1)
if not isinstance(metrics, dict):
    print(f'FAIL: compute_loss returned metrics as {type(metrics).__name__}, expected dict')
    exit(1)
print('PASS')
"`,
          { env, timeout: 60000 }
        );

        if (!result.success || !result.output.includes("PASS")) {
          return {
            success: false,
            error: classifyAndAttributeError(
              `compute-loss-reference failed: ${result.output}`,
              "compute-loss-reference",
              logger
            ),
          };
        }
        const message = "compute_loss matched hand-computed test oracle GRPO values";
        logger.info(message);
        return { success: true, output: message };
      },
    },
  ];
}
