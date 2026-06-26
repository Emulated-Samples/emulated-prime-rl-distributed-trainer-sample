"""Distributed RL trainer with GRPO loss."""

from prime_rl.trainer.rl.config import RLTrainerConfig
from prime_rl.utils.pydantic_config import parse_argv


def train(config: RLTrainerConfig):
    """Main training function."""
    raise NotImplementedError


def main():
    """Entry point for RL trainer. Run using `uv run trainer`."""
    train(parse_argv(RLTrainerConfig))


if __name__ == "__main__":
    main()
