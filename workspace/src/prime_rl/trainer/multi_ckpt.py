from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prime_rl.trainer.ckpt import WeightCheckpointManager


class MultiCheckpointManager:
    """Owns per-run checkpoint managers and run-local state."""

    def __init__(self, output_dir: Path):
        raise NotImplementedError()


def setup_multi_checkpoint_manager(
    output_dir: Path,
) -> tuple[MultiCheckpointManager, "WeightCheckpointManager | None"]:
    raise NotImplementedError()
