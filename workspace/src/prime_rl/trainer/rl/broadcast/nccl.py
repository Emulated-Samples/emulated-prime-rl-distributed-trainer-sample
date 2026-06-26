from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.rl.config import NCCLWeightBroadcastConfig


def broadcast_integer(integer: int, communicator: PyNcclCommunicator) -> None:
    raise NotImplementedError()


def broadcast_state_dict(state_dict: dict[str, Tensor], communicator: PyNcclCommunicator) -> None:
    raise NotImplementedError()


class NCCLWeightBroadcastSender:
    def __init__(
        self,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        device: int | str | torch.device,
        timeout: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        raise NotImplementedError()

    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        raise NotImplementedError()


class NCCLWeightBroadcast(WeightBroadcast):
    """Broadcast weights to inference using NCCL."""

    def __init__(
        self,
        output_dir: Path,
        config: NCCLWeightBroadcastConfig,
        device: int | str | torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        raise NotImplementedError()

    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        raise NotImplementedError()
