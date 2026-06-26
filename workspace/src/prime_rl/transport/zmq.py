from pathlib import Path

from prime_rl.transport.base import MicroBatchReceiver, MicroBatchSender, TrainingBatchReceiver, TrainingBatchSender
from prime_rl.transport.config import ZMQTransportConfig
from prime_rl.transport.types import MicroBatch, TrainingBatch


class ZMQTrainingBatchSender(TrainingBatchSender):
    """ZMQ training-batch sender."""

    def __init__(self, output_dir: Path, transport: ZMQTransportConfig):
        raise NotImplementedError()

    def send(self, batch: TrainingBatch) -> None:
        raise NotImplementedError()


class ZMQTrainingBatchReceiver(TrainingBatchReceiver):
    """ZMQ training-batch receiver."""

    def __init__(self, transport: ZMQTransportConfig):
        raise NotImplementedError()

    def can_receive(self) -> bool:
        raise NotImplementedError()

    def receive(self) -> list[TrainingBatch]:
        raise NotImplementedError()


class ZMQMicroBatchSender(MicroBatchSender):
    """ZMQ micro-batch sender."""

    def __init__(self, output_dir: Path, data_world_size: int, current_step: int, transport: ZMQTransportConfig):
        raise NotImplementedError()

    def send(self, micro_batch_grid: list[list[MicroBatch]]) -> None:
        raise NotImplementedError()


class ZMQMicroBatchReceiver(MicroBatchReceiver):
    """ZMQ micro-batch receiver."""

    def __init__(self, output_dir: Path, data_rank: int, current_step: int, transport: ZMQTransportConfig):
        raise NotImplementedError()

    def wait(self) -> None:
        raise NotImplementedError()

    def can_receive(self) -> bool:
        raise NotImplementedError()

    def receive(self) -> list[MicroBatch]:
        raise NotImplementedError()
