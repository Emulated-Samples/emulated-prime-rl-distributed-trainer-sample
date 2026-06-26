from pathlib import Path

from prime_rl.trainer.config import LoRAConfig
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.rl.broadcast.filesystem import FileSystemWeightBroadcast
from prime_rl.trainer.rl.broadcast.nccl import NCCLWeightBroadcast
from prime_rl.trainer.rl.config import WeightBroadcastConfigType


def setup_weight_broadcast(
    output_dir: Path, config: WeightBroadcastConfigType, lora_config: LoRAConfig | None = None
) -> WeightBroadcast:
    """Create a weight broadcast handler based on config type."""
    raise NotImplementedError()
