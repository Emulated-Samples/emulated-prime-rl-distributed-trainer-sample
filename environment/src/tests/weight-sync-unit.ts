import { Logger, SimpleTest, executeWithExitCode } from "@hyperfocal/env-base";

import { RELEASE_NAME } from "../config.js";
import { classifyAndAttributeError, getEnvWithAws } from "../helpers.js";

const POD_NAME = `${RELEASE_NAME}-trainer-0`;

export function createWeightSyncUnitTests(): SimpleTest[] {
  return [
    {
      id: "weight-sync-import-surface",
      name: "Weight Sync Public Imports",
      description: "Verify the transport, broadcast, parallel-dims, and checkpoint surfaces import cleanly",
      run: async (logger: Logger) =>
        runPythonSnippet(logger, "weight-sync-import-surface", "Weight-sync import surface", IMPORT_SURFACE_SCRIPT),
    },
    {
      id: "weight-sync-factory-dispatch",
      name: "Weight Sync Config Factory Dispatch",
      description: "Verify config-driven dispatch across filesystem, ZMQ, and NCCL surfaces",
      run: async (logger: Logger) =>
        runPythonSnippet(logger, "weight-sync-factory-dispatch", "Weight-sync factory dispatch", FACTORY_DISPATCH_SCRIPT),
    },
    {
      id: "filesystem-transport-roundtrip",
      name: "Filesystem Transport Roundtrip",
      description: "Verify filesystem training-batch and micro-batch transport works on a tiny fixture",
      run: async (logger: Logger) =>
        runPythonSnippet(logger, "filesystem-transport-roundtrip", "Filesystem transport roundtrip", FILESYSTEM_ROUNDTRIP_SCRIPT),
    },
    {
      id: "zmq-transport-roundtrip",
      name: "ZMQ Transport Roundtrip",
      description: "Verify ZMQ training-batch and micro-batch transport works on localhost",
      run: async (logger: Logger) =>
        runPythonSnippet(logger, "zmq-transport-roundtrip", "ZMQ transport roundtrip", ZMQ_ROUNDTRIP_SCRIPT),
    },
    {
      id: "parallel-dims-validation",
      name: "Parallel Dims Validation",
      description: "Verify get_parallel_dims succeeds for valid inputs and fails clearly for invalid ones",
      run: async (logger: Logger) =>
        runPythonSnippet(logger, "parallel-dims-validation", "Parallel dims validation", PARALLEL_DIMS_SCRIPT),
    },
    {
      id: "checkpoint-broadcast-smoke",
      name: "Filesystem Broadcast Stable Checkpoint",
      description: "Verify multi-run checkpoint setup and filesystem broadcast notification semantics",
      run: async (logger: Logger) =>
        runPythonSnippet(logger, "checkpoint-broadcast-smoke", "Checkpoint and broadcast smoke", CHECKPOINT_BROADCAST_SCRIPT),
    },
  ];
}

async function runPythonSnippet(
  logger: Logger,
  testId: string,
  label: string,
  snippet: string
): Promise<{ success: boolean; error?: string; output?: string; errored?: boolean }> {
  const env = getEnvWithAws();
  const result = await executeWithExitCode(
    `kubectl exec -i ${POD_NAME} -- python3 - <<'PY'\n${snippet}\nPY`,
    { env, timeout: 120000 }
  );

  if (!result.success || !result.output.includes("PASS:")) {
    return {
      success: false,
      error: classifyAndAttributeError(`${label} failed: ${result.output}`, testId, logger),
    };
  }

  const passLine = result.output.split("\n").find((line) => line.startsWith("PASS:")) ?? `${label} passed`;
  logger.info(passLine);
  return { success: true, output: passLine.replace(/^PASS:\s*/, "") };
}

const IMPORT_SURFACE_SCRIPT = `
from prime_rl.transport import (
    setup_training_batch_sender,
    setup_training_batch_receiver,
    setup_micro_batch_sender,
    setup_micro_batch_receiver,
    TrainingSample,
    TrainingBatch,
    MicroBatch,
)
from prime_rl.trainer.rl.broadcast import setup_weight_broadcast
from prime_rl.trainer.parallel_dims import ParallelDims, get_parallel_dims
from prime_rl.trainer.multi_ckpt import MultiCheckpointManager, setup_multi_checkpoint_manager
print("PASS: weight-sync modules exported the required public symbols")
`;

const FACTORY_DISPATCH_SCRIPT = `
from pathlib import Path
from unittest.mock import patch

from prime_rl.transport import (
    setup_micro_batch_receiver,
    setup_micro_batch_sender,
    setup_training_batch_receiver,
    setup_training_batch_sender,
)
from prime_rl.transport.config import FileSystemTransportConfig, ZMQTransportConfig
from prime_rl.trainer.rl.broadcast import setup_weight_broadcast
from prime_rl.trainer.rl.config import FileSystemWeightBroadcastConfig, NCCLWeightBroadcastConfig


class FakeManager:
    used_idxs = [0]
    ready_to_update = [False]
    ready_to_update_idxs = []
    progress = {0: type("Progress", (), {"step": 0})()}
    idx_2_id = {0: "run0"}

    def get_run_dir(self, idx):
        return Path("/tmp/run0")


class FakeWorld:
    is_master = False
    rank = 0


def check_type(obj, expected_name, label):
    actual_name = type(obj).__name__
    if actual_name != expected_name:
        print(f"FAIL: {label} returned {actual_name}, expected {expected_name}")
        raise SystemExit(1)


with patch("prime_rl.trainer.runs.get_multi_run_manager", return_value=FakeManager(), create=True), patch(
    "prime_rl.transport.filesystem.get_multi_run_manager", return_value=FakeManager(), create=True
), patch("prime_rl.transport.zmq.get_multi_run_manager", return_value=FakeManager(), create=True):
    fs_config = FileSystemTransportConfig()
    zmq_config = ZMQTransportConfig(host="127.0.0.1", port=5561, hwm=10)

    check_type(setup_training_batch_sender(Path("/tmp/run0"), fs_config), "FileSystemTrainingBatchSender", "filesystem training-batch sender factory")
    check_type(setup_training_batch_receiver(fs_config), "FileSystemTrainingBatchReceiver", "filesystem training-batch receiver factory")

    zmq_receiver = setup_training_batch_receiver(zmq_config)
    zmq_sender = setup_training_batch_sender(Path("/tmp/run0"), zmq_config)
    check_type(zmq_sender, "ZMQTrainingBatchSender", "ZMQ training-batch sender factory")
    check_type(zmq_receiver, "ZMQTrainingBatchReceiver", "ZMQ training-batch receiver factory")
    zmq_sender.close()
    zmq_receiver.close()

    check_type(setup_micro_batch_sender(Path("/tmp/run0"), 1, 0, fs_config), "FileSystemMicroBatchSender", "filesystem micro-batch sender factory")
    check_type(setup_micro_batch_receiver(Path("/tmp/run0"), 0, 0, fs_config), "FileSystemMicroBatchReceiver", "filesystem micro-batch receiver factory")

with patch("prime_rl.trainer.runs.get_multi_run_manager", return_value=FakeManager(), create=True), patch(
    "prime_rl.trainer.world.get_world", return_value=FakeWorld(), create=True
), patch("prime_rl.trainer.rl.broadcast.filesystem.get_multi_run_manager", return_value=FakeManager(), create=True), patch(
    "prime_rl.trainer.rl.broadcast.filesystem.get_world", return_value=FakeWorld(), create=True
), patch("prime_rl.trainer.rl.broadcast.nccl.get_world", return_value=FakeWorld(), create=True), patch(
    "torch.cuda.current_device", return_value=0
):
    check_type(setup_weight_broadcast(Path("/tmp/run0"), FileSystemWeightBroadcastConfig()), "FileSystemWeightBroadcast", "filesystem weight-broadcast factory")
    check_type(setup_weight_broadcast(Path("/tmp/run0"), NCCLWeightBroadcastConfig()), "NCCLWeightBroadcast", "NCCL weight-broadcast factory")

print("PASS: factories selected filesystem, ZMQ, and NCCL implementations by config type")
`;

const FILESYSTEM_ROUNDTRIP_SCRIPT = `
from pathlib import Path
import tempfile
from unittest.mock import patch

from prime_rl.transport import (
    setup_micro_batch_receiver,
    setup_micro_batch_sender,
    setup_training_batch_receiver,
    setup_training_batch_sender,
)
from prime_rl.transport.config import FileSystemTransportConfig
from prime_rl.transport.types import MicroBatch, TrainingBatch, TrainingSample

run_dir = Path(tempfile.mkdtemp()) / "run0"
run_dir.mkdir(parents=True, exist_ok=True)


def check(condition, message):
    if not condition:
        print(f"FAIL: {message}")
        raise SystemExit(1)


class FakeManager:
    used_idxs = [0]
    ready_to_update = [False]
    progress = {0: type("Progress", (), {"step": 0})()}

    def get_run_dir(self, idx):
        return run_dir


with patch("prime_rl.trainer.runs.get_multi_run_manager", return_value=FakeManager(), create=True), patch(
    "prime_rl.transport.filesystem.get_multi_run_manager", return_value=FakeManager(), create=True
):
    transport = FileSystemTransportConfig()
    sender = setup_training_batch_sender(run_dir, transport)
    receiver = setup_training_batch_receiver(transport)

    batch = TrainingBatch(
        examples=[
            TrainingSample(
                prompt_ids=[1],
                prompt_mask=[True],
                completion_ids=[2],
                completion_mask=[True],
                completion_logprobs=[0.0],
                advantage=1.0,
                reward=1.0,
            )
        ],
        temperature=0.7,
        step=0,
    )
    sender.send(batch)
    check(receiver.can_receive(), "filesystem training-batch receiver was not ready after send")
    received = receiver.receive()
    check(len(received) == 1, f"filesystem training-batch receiver returned {len(received)} batch(es), expected 1")
    check(received[0].step == 0, f"filesystem training-batch step was {received[0].step}, expected 0")
    check(received[0].examples[0].prompt_ids == [1], f"filesystem training sample prompt_ids were {received[0].examples[0].prompt_ids}, expected [1]")

transport = FileSystemTransportConfig()
micro_sender = setup_micro_batch_sender(run_dir, 1, 0, transport)
micro_receiver = setup_micro_batch_receiver(run_dir, 0, 0, transport)
micro_batch = MicroBatch(
    input_ids=[1, 2],
    loss_mask=[True, True],
    advantages=[1.0, 0.5],
    inference_logprobs=[0.0, -0.1],
    position_ids=[0, 1],
    temperature=0.8,
)
micro_sender.send([[micro_batch]])
check(micro_receiver.can_receive(), "filesystem micro-batch receiver was not ready after send")
received_micro = micro_receiver.receive()
check(len(received_micro) == 1, f"filesystem micro-batch receiver returned {len(received_micro)} micro-batch(es), expected 1")
check(received_micro[0].input_ids == [1, 2], f"filesystem micro-batch input_ids were {received_micro[0].input_ids}, expected [1, 2]")
check(received_micro[0].temperature == 0.8, f"filesystem micro-batch temperature was {received_micro[0].temperature}, expected 0.8")

print("PASS: filesystem transport preserved training-batch and micro-batch contents")
`;

const ZMQ_ROUNDTRIP_SCRIPT = `
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import zmq

from prime_rl.transport import (
    setup_micro_batch_receiver,
    setup_micro_batch_sender,
    setup_training_batch_receiver,
    setup_training_batch_sender,
)
from prime_rl.transport.config import ZMQTransportConfig
from prime_rl.transport.types import MicroBatch, TrainingBatch, TrainingSample


def _pick_free_port() -> int:
    """Bind a probe socket to a kernel-assigned ephemeral port and return it.

    Avoids the TCP TIME_WAIT collision that hardcoded ports hit when the test
    is rerun within ~60s of a prior invocation on the same pod.
    """
    ctx = zmq.Context.instance()
    probe = ctx.socket(zmq.PUB)
    try:
        probe.bind("tcp://127.0.0.1:*")
        endpoint = probe.getsockopt(zmq.LAST_ENDPOINT).decode()
        return int(endpoint.rsplit(":", 1)[1])
    finally:
        probe.close(linger=0)


run_dir = Path(tempfile.mkdtemp()) / "run0"
run_dir.mkdir(parents=True, exist_ok=True)
training_port = _pick_free_port()
transport = ZMQTransportConfig(host="127.0.0.1", port=training_port, hwm=10)


def check(condition, message):
    if not condition:
        print(f"FAIL: {message}")
        raise SystemExit(1)


class FakeManager:
    used_idxs = [0]
    ready_to_update = [False]
    progress = {0: type("Progress", (), {"step": 0})()}
    idx_2_id = {0: "run0"}

    def get_run_dir(self, idx):
        return run_dir


with patch("prime_rl.trainer.runs.get_multi_run_manager", return_value=FakeManager(), create=True), patch(
    "prime_rl.transport.zmq.get_multi_run_manager", return_value=FakeManager(), create=True
):
    receiver = setup_training_batch_receiver(transport)
    sender = setup_training_batch_sender(run_dir, transport)
    batch = TrainingBatch(
        examples=[
            TrainingSample(
                prompt_ids=[1],
                prompt_mask=[True],
                completion_ids=[2],
                completion_mask=[True],
                completion_logprobs=[0.0],
                advantage=1.0,
                reward=1.0,
            )
        ],
        temperature=0.6,
        step=0,
    )
    sender.send(batch)

    deadline = time.time() + 2
    while not receiver.can_receive() and time.time() < deadline:
        time.sleep(0.05)
    check(receiver.can_receive(), "ZMQ training-batch receiver was not ready within 2 seconds")

    received = receiver.receive()
    check(len(received) == 1, f"ZMQ training-batch receiver returned {len(received)} batch(es), expected 1")
    check(received[0].examples[0].completion_ids == [2], f"ZMQ training sample completion_ids were {received[0].examples[0].completion_ids}, expected [2]")
    sender.close()
    receiver.close()

micro_receiver = setup_micro_batch_receiver(run_dir, 0, 0, transport)
micro_sender = setup_micro_batch_sender(run_dir, 1, 0, transport)
micro_batch = MicroBatch(
    input_ids=[4, 5],
    loss_mask=[True, False],
    advantages=[0.1, 0.2],
    inference_logprobs=[-0.2, -0.3],
    position_ids=[0, 1],
    temperature=0.5,
)
micro_sender.send([[micro_batch]])

deadline = time.time() + 2
while not micro_receiver.can_receive() and time.time() < deadline:
    time.sleep(0.05)
check(micro_receiver.can_receive(), "ZMQ micro-batch receiver was not ready within 2 seconds")

received_micro = micro_receiver.receive()
check(len(received_micro) == 1, f"ZMQ micro-batch receiver returned {len(received_micro)} micro-batch(es), expected 1")
check(received_micro[0].input_ids == [4, 5], f"ZMQ micro-batch input_ids were {received_micro[0].input_ids}, expected [4, 5]")
micro_sender.close()
micro_receiver.close()

print("PASS: ZMQ transport preserved training-batch and micro-batch contents")
`;

const PARALLEL_DIMS_SCRIPT = `
from unittest.mock import patch

from prime_rl.trainer.config import ModelConfig
from prime_rl.trainer.parallel_dims import get_parallel_dims

with patch("torch.distributed.get_world_size", return_value=1):
    dims = get_parallel_dims(ModelConfig(dp_replicate=1, cp=1, tp=1, ep=1), seq_len=128)
    if dims.world_size != 1:
        print(f"FAIL: valid parallel dims reported world_size {dims.world_size}, expected 1")
        raise SystemExit(1)
    if dims.dp_replicate != 1:
        print(f"FAIL: valid parallel dims reported dp_replicate {dims.dp_replicate}, expected 1")
        raise SystemExit(1)
    if dims.cp_enabled:
        print("FAIL: cp_enabled was true for cp=1")
        raise SystemExit(1)

with patch("torch.distributed.get_world_size", return_value=1):
    try:
        get_parallel_dims(ModelConfig(dp_replicate=1, cp=2, tp=1, ep=1), seq_len=3)
    except ValueError:
        pass
    else:
        print("FAIL: invalid sequence length with cp=2 did not raise ValueError")
        raise SystemExit(1)

print("PASS: parallel dims accepted valid dimensions and rejected invalid sequence partitioning")
`;

const CHECKPOINT_BROADCAST_SCRIPT = `
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch
import torch.nn as nn

from prime_rl.trainer.multi_ckpt import setup_multi_checkpoint_manager
from prime_rl.trainer.rl.broadcast import setup_weight_broadcast
from prime_rl.trainer.rl.config import FileSystemWeightBroadcastConfig


class FakeManager:
    max_runs = 2
    used_idxs = [0]
    ready_to_update = [True, False]
    ready_to_update_idxs = [0]
    progress = {0: type("Progress", (), {"step": 3})()}
    idx_2_id = {0: "run0"}
    config = {0: type("RunConfig", (), {"ckpt": None})()}

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def register_deletion_hook(self, hook):
        self.deletion_hook = hook

    def register_creation_hook(self, hook):
        self.creation_hook = hook

    def get_run_dir(self, idx):
        return self.output_dir / "run0"

    def get_orchestrator_config(self, run_id):
        return object()


class FakeWorld:
    is_master = True
    rank = 0


output_dir = Path(tempfile.mkdtemp())
fake_manager = FakeManager(output_dir)

with patch("prime_rl.trainer.runs.get_multi_run_manager", return_value=fake_manager, create=True), patch(
    "prime_rl.trainer.world.get_world", return_value=FakeWorld(), create=True
), patch("prime_rl.trainer.multi_ckpt.get_multi_run_manager", return_value=fake_manager, create=True), patch(
    "prime_rl.trainer.multi_ckpt.get_world", return_value=FakeWorld(), create=True
):
    manager, _ = setup_multi_checkpoint_manager(output_dir)

def fake_save_state_dict(state_dict, save_dir, save_format, save_sharded, adapter=False):
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "model.safetensors").write_bytes(b"ok")

with patch("prime_rl.trainer.runs.get_multi_run_manager", return_value=fake_manager, create=True), patch(
    "prime_rl.trainer.world.get_world", return_value=FakeWorld(), create=True
), patch("prime_rl.trainer.rl.broadcast.filesystem.get_multi_run_manager", return_value=fake_manager, create=True), patch(
    "prime_rl.trainer.rl.broadcast.filesystem.get_world", return_value=FakeWorld(), create=True
), patch("prime_rl.trainer.rl.broadcast.filesystem.gather_weights_on_master", return_value={"weight": torch.ones(1)}), patch(
    "prime_rl.trainer.rl.broadcast.filesystem.save_state_dict", side_effect=fake_save_state_dict
):
    broadcast = setup_weight_broadcast(output_dir, FileSystemWeightBroadcastConfig())
    broadcast.broadcast_weights(nn.Linear(1, 1), step=3)

save_dir = output_dir / "run0" / "broadcasts" / "step_3"
if manager is None:
    print("FAIL: setup_multi_checkpoint_manager returned no manager")
    raise SystemExit(1)
if not (save_dir / "STABLE").exists():
    print(f"FAIL: filesystem broadcast did not create stable marker at {save_dir / 'STABLE'}")
    raise SystemExit(1)
if not (save_dir / "model.safetensors").exists():
    print(f"FAIL: filesystem broadcast did not create model weights at {save_dir / 'model.safetensors'}")
    raise SystemExit(1)
print("PASS: multi-run checkpoint manager initialized and filesystem broadcast produced a stable checkpoint")
`;
