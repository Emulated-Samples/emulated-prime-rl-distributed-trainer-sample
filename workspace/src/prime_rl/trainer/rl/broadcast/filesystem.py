from pathlib import Path

import torch.nn as nn

from prime_rl.trainer.config import LoRAConfig
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.rl.config import FileSystemWeightBroadcastConfig


class FileSystemWeightBroadcast(WeightBroadcast):
    """Broadcast weights to inference through shared filesystem checkpoints."""

    def __init__(
        self, output_dir: Path, config: FileSystemWeightBroadcastConfig, lora_config: LoRAConfig | None = None
    ):
        raise NotImplementedError()

    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        raise NotImplementedError()

    def maybe_clean(self, max_async_level: int, interval_to_keep: int | None):
        raise NotImplementedError()
