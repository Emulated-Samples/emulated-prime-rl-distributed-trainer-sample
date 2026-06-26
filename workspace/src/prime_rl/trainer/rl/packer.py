"""Batch packing for distributed RL training."""

from abc import ABC, abstractmethod

from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.transport import TransportConfigType


class BasePacker(ABC):
    """Base class for batch packers."""

    def __init__(
        self,
        dp_world_size: int,
        seq_len: int,
        pad_to_multiple_of: int,
        tokenizer: PreTrainedTokenizer,
        config: TransportConfigType,
        start_step: int = 0,
    ):
        raise NotImplementedError

    @abstractmethod
    def pack(self) -> None:
        """Pack samples for the next training step."""
        pass


class SinglePacker(BasePacker):
    """Packer for single-run training."""

    def __init__(
        self,
        dp_world_size: int,
        seq_len: int,
        pad_to_multiple_of: int,
        tokenizer: PreTrainedTokenizer,
        config: TransportConfigType,
        start_step: int = 0,
    ):
        raise NotImplementedError

    def pack(self) -> None:
        raise NotImplementedError


class MultiPacker(BasePacker):
    """Packer for multi-run training with round-robin scheduling."""

    def __init__(
        self,
        dp_world_size: int,
        seq_len: int,
        pad_to_multiple_of: int,
        tokenizer: PreTrainedTokenizer,
        config: TransportConfigType,
        start_step: int = 0,
    ):
        raise NotImplementedError

    def pack(self) -> None:
        raise NotImplementedError


def setup_packer(
    dp_world_size: int,
    seq_len: int,
    pad_to_multiple_of: int,
    tokenizer: PreTrainedTokenizer,
    transport_config: TransportConfigType,
    start_step: int = 0,
) -> BasePacker:
    """Create appropriate packer based on configuration."""
    raise NotImplementedError
