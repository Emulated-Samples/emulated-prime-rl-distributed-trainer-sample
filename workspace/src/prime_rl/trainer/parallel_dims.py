from dataclasses import dataclass

from torch.distributed.device_mesh import DeviceMesh

from prime_rl.trainer.config import ModelConfig

__all__ = ["ParallelDims", "get_parallel_dims"]


@dataclass
class ParallelDims:
    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int
    pp: int
    ep: int
    world_size: int

    _world_mesh: DeviceMesh | None = None

    def __post_init__(self):
        raise NotImplementedError()

    def build_mesh(self) -> DeviceMesh:
        raise NotImplementedError()

    @property
    def world_mesh(self) -> DeviceMesh:
        raise NotImplementedError()

    @property
    def dp_enabled(self):
        raise NotImplementedError()

    @property
    def dp_replicate_enabled(self):
        raise NotImplementedError()

    @property
    def dp_shard_enabled(self):
        raise NotImplementedError()

    @property
    def cp_enabled(self):
        raise NotImplementedError()

    @property
    def dp_cp_enabled(self):
        raise NotImplementedError()

    @property
    def fsdp_enabled(self):
        raise NotImplementedError()

    @property
    def tp_enabled(self):
        raise NotImplementedError()

    @property
    def pp_enabled(self):
        raise NotImplementedError()

    @property
    def ep_enabled(self):
        raise NotImplementedError()

    @property
    def fsdp_gradient_divide_factor(self) -> int:
        raise NotImplementedError()

    @property
    def non_data_parallel_size(self):
        raise NotImplementedError()

    @property
    def seq_len_divisor(self):
        raise NotImplementedError()


def get_parallel_dims(config: ModelConfig, seq_len: int | None = None) -> ParallelDims:
    raise NotImplementedError()
