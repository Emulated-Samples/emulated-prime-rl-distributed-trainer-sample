from pathlib import Path

from prime_rl.transport.base import MicroBatchReceiver, MicroBatchSender, TrainingBatchReceiver, TrainingBatchSender
from prime_rl.transport.types import MicroBatch, TrainingBatch


class FileSystemTrainingBatchSender(TrainingBatchSender):
    """Filesystem-based training batch sender that writes batches to disk."""

    def __init__(self, output_dir: Path):
        raise NotImplementedError()

    def send(self, batch: TrainingBatch) -> None:
        raise NotImplementedError()


class FileSystemTrainingBatchReceiver(TrainingBatchReceiver):
    """Filesystem-based training batch receiver that reads batches from one or more run directories."""

    def __init__(self) -> None:
        raise NotImplementedError()

    def can_receive(self) -> bool:
        raise NotImplementedError()

    def receive(self) -> list[TrainingBatch]:
        raise NotImplementedError()


class FileSystemMicroBatchSender(MicroBatchSender):
    """Filesystem-based micro-batch sender that writes packed micro-batches to disk."""

    def __init__(self, output_dir: Path, data_world_size: int, current_step: int = 0):
        raise NotImplementedError()

    def send(self, micro_batch_grid: list[list[MicroBatch]]) -> None:
        raise NotImplementedError()


class FileSystemMicroBatchReceiver(MicroBatchReceiver):
    """Filesystem-based micro-batch receiver that reads packed micro-batches from disk."""

    def __init__(self, output_dir: Path, data_rank: int, current_step: int = 0):
        raise NotImplementedError()

    def wait(self) -> None:
        raise NotImplementedError()

    def can_receive(self) -> bool:
        raise NotImplementedError()

    def receive(self) -> list[MicroBatch]:
        raise NotImplementedError()
