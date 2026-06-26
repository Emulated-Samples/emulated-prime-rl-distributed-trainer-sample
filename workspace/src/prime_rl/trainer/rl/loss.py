"""GRPO loss computation with importance ratio masking."""

from typing import Any

import torch
from torch import Tensor

from prime_rl.trainer.rl.config import LossConfig


def selective_log_softmax(logits: Tensor, index: Tensor) -> Tensor:
    """Gather log probabilities at the specified token indices."""
    raise NotImplementedError


def compute_entropy(shifted_logits: Tensor) -> Tensor:
    """Compute entropy of the probability distribution."""
    raise NotImplementedError


def shift_tensor_left(t: Tensor) -> Tensor:
    """Shift tensor one position left, padding with zeros on the right."""
    raise NotImplementedError


def shift_tensor_right(t: Tensor, pad_value: float | None = None) -> Tensor:
    """Shift tensor one position right, prepending pad_value."""
    raise NotImplementedError


def compute_loss(
    trainer_logprobs: Any,
    inference_logprobs: Any,
    teacher_logprobs: Any | None,
    advantages: Any,
    loss_mask: Any,
    loss_config: LossConfig,
    loss_scale: int,
) -> tuple[Tensor, dict[str, Any]]:
    """Compute GRPO loss with importance ratio masking."""
    raise NotImplementedError
