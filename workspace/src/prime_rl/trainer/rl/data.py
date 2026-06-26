"""Training data loading and batch management."""

from pathlib import Path
from typing import TypedDict

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from transformers.tokenization_utils import PreTrainedTokenizer

from prime_rl.trainer.rl.config import FakeDataLoaderConfig
from prime_rl.transport import TransportConfigType


class TensorMicroBatch(TypedDict):
    """A micro batch of data for training."""

    input_ids: Int[Tensor, "batch seq"]
    position_ids: Int[Tensor, "batch seq"]
    advantages: Float[Tensor, "batch seq"]
    inference_logprobs: Float[Tensor, "batch seq"]
    teacher_logprobs: Float[Tensor, "batch seq"] | None
    loss_mask: Bool[Tensor, "batch seq"]
    temperature: float
    lora_num_tokens: Int[Tensor, "n_loras"]


class FakeDataLoader:
    """Fake data loader for debugging without orchestrator."""

    def __init__(self, config: FakeDataLoaderConfig, seq_len: int, dp_world_size: int):
        raise NotImplementedError

    def wait_for_batch(self) -> None:
        raise NotImplementedError

    def get_batch(self) -> list[TensorMicroBatch]:
        raise NotImplementedError


class DataLoader:
    """Loads training batches from the orchestrator via transport layer."""

    def __init__(
        self,
        output_dir: Path,
        start_step: int,
        dp_world_size: int,
        seq_len: int,
        pad_to_multiple_of: int,
        tokenizer: PreTrainedTokenizer,
        config: TransportConfigType,
    ):
        raise NotImplementedError

    def wait_for_batch(self) -> None:
        raise NotImplementedError

    def get_batch(self) -> list[TensorMicroBatch]:
        raise NotImplementedError
