from pathlib import Path

from prime_rl.transport.base import MicroBatchReceiver, MicroBatchSender, TrainingBatchReceiver, TrainingBatchSender
from prime_rl.transport.config import TransportConfigType
from prime_rl.transport.filesystem import (
    FileSystemMicroBatchReceiver,
    FileSystemMicroBatchSender,
    FileSystemTrainingBatchReceiver,
    FileSystemTrainingBatchSender,
)
from prime_rl.transport.types import MicroBatch, TrainingBatch, TrainingSample
from prime_rl.transport.zmq import (
    ZMQMicroBatchReceiver,
    ZMQMicroBatchSender,
    ZMQTrainingBatchReceiver,
    ZMQTrainingBatchSender,
)


def setup_training_batch_sender(output_dir: Path, transport: TransportConfigType) -> TrainingBatchSender:
    """Create a training batch sender based on transport config."""
    raise NotImplementedError()


def setup_training_batch_receiver(transport: TransportConfigType) -> TrainingBatchReceiver:
    """Create a training batch receiver based on transport config."""
    raise NotImplementedError()


def setup_micro_batch_sender(
    output_dir: Path, data_world_size: int, current_step: int, transport: TransportConfigType
) -> MicroBatchSender:
    """Create a micro-batch sender based on transport config."""
    raise NotImplementedError()


def setup_micro_batch_receiver(
    output_dir: Path, data_rank: int, current_step: int, transport: TransportConfigType
) -> MicroBatchReceiver:
    """Create a micro-batch receiver based on transport config."""
    raise NotImplementedError()


__all__ = [
    "FileSystemTrainingBatchSender",
    "FileSystemTrainingBatchReceiver",
    "FileSystemMicroBatchSender",
    "FileSystemMicroBatchReceiver",
    "MicroBatchReceiver",
    "MicroBatchSender",
    "ZMQTrainingBatchSender",
    "ZMQTrainingBatchReceiver",
    "ZMQMicroBatchSender",
    "ZMQMicroBatchReceiver",
    "TrainingSample",
    "TrainingBatch",
    "MicroBatch",
    "setup_training_batch_sender",
    "setup_training_batch_receiver",
    "setup_micro_batch_sender",
    "setup_micro_batch_receiver",
]
