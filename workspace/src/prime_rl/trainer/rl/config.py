from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator

from prime_rl.trainer.config import (
    AdamWConfig,
    BenchConfig,
    CheckpointConfig,
    ConstantSchedulerConfig,
    ModelConfig,
    OptimizerConfigType,
    SchedulerConfigType,
    TokenizerConfig,
)
from prime_rl.transport.config import FileSystemTransportConfig, TransportConfigType
from prime_rl.utils.config import HeartbeatConfig, LogConfig, MetricsServerConfig, WandbConfig
from prime_rl.utils.pydantic_config import BaseConfig, BaseSettings


class LossConfig(BaseConfig):
    """Loss configuration."""

    ratio_type: Annotated[Literal["token", "sequence"], Field()] = "token"
    token_mask_high: Annotated[float, Field(ge=0)] = 8.0
    token_mask_low: Annotated[float, Field(ge=0)] = 0.125
    sequence_clip_high: Annotated[float, Field(ge=0)] = 10.0
    geo_mask_high: Annotated[float, Field(ge=0)] = 10.0
    geo_mask_low: Annotated[float, Field(ge=0)] = 0.1
    sequence_mask_low: Annotated[float, Field(ge=0)] = 0.0
    sequence_mask_high: Annotated[float, Field(ge=0)] = 100.0
    adv_tau: Annotated[float, Field(ge=0)] = 1.0
    teacher_tau: Annotated[float, Field(ge=0)] = 0.0
    kl_tau: Annotated[float, Field(ge=0)] = 0.0

    @model_validator(mode="after")
    def validate_mask_bounds(self):
        if self.token_mask_low >= self.token_mask_high:
            raise ValueError(
                f"token_mask_low ({self.token_mask_low}) must be less than token_mask_high ({self.token_mask_high})"
            )
        if self.geo_mask_low >= self.geo_mask_high:
            raise ValueError(
                f"geo_mask_low ({self.geo_mask_low}) must be less than geo_mask_high ({self.geo_mask_high})"
            )
        if self.sequence_mask_low >= self.sequence_mask_high:
            raise ValueError(
                f"sequence_mask_low ({self.sequence_mask_low}) must be less than sequence_mask_high ({self.sequence_mask_high})"
            )
        return self


class FakeDataLoaderConfig(BaseConfig):
    """Fake data loader configuration."""

    batch_size: Annotated[int, Field(ge=1)] = 2
    generate_samples: bool = False


class DataLoaderConfig(BaseConfig):
    """Data loader configuration."""

    fake: FakeDataLoaderConfig | None = None


class BaseWeightBroadcastConfig(BaseModel):
    """Base weight broadcast configuration."""

    pass


class FileSystemWeightBroadcastConfig(BaseWeightBroadcastConfig):
    """Filesystem weight broadcast configuration."""

    type: Literal["filesystem"] = "filesystem"
    save_sharded: bool = True
    save_format: Literal["safetensors", "torch"] = "safetensors"


class NCCLWeightBroadcastConfig(BaseWeightBroadcastConfig):
    """NCCL weight broadcast configuration."""

    type: Literal["nccl"] = "nccl"
    host: str = "localhost"
    port: int = 29501
    timeout: int = 1200
    inference_world_size: int = 1


WeightBroadcastConfigType: TypeAlias = FileSystemWeightBroadcastConfig | NCCLWeightBroadcastConfig


class RLTrainerConfig(BaseSettings):
    """RL trainer configuration."""

    model: ModelConfig = ModelConfig()
    tokenizer: TokenizerConfig = TokenizerConfig()
    data: DataLoaderConfig = DataLoaderConfig()
    loss: LossConfig = LossConfig()
    optim: Annotated[OptimizerConfigType, Field(discriminator="type")] = AdamWConfig()
    scheduler: Annotated[SchedulerConfigType, Field(discriminator="type")] = ConstantSchedulerConfig()
    ckpt: CheckpointConfig | None = None
    weight_broadcast: Annotated[WeightBroadcastConfigType, Field(discriminator="type")] = (
        FileSystemWeightBroadcastConfig()
    )
    rollout_transport: Annotated[TransportConfigType, Field(discriminator="type")] = FileSystemTransportConfig()
    log: LogConfig = LogConfig()
    wandb: WandbConfig | None = None
    output_dir: Path = Path("outputs")
    max_steps: int | None = None
    max_async_level: Annotated[int, Field(ge=0)] = 1
    memory_profiler_path: Path | None = None
    bench: BenchConfig | None = None
    trace_path: Path | None = None
    dist_timeout_seconds: int = 600
    heartbeat: HeartbeatConfig | None = None
    metrics_server: MetricsServerConfig | None = None
    max_concurrent_runs: Annotated[int, Field(ge=1)] = 1

    @model_validator(mode="after")
    def auto_setup_bench(self):
        if self.bench is not None:
            self.max_steps = 4
            if not self.data.fake:
                self.data.fake = FakeDataLoaderConfig()
            if self.ckpt:
                self.ckpt = None
        return self

    @model_validator(mode="after")
    def dont_do_massive_traces(self):
        if self.trace_path:
            if self.max_steps is None:
                raise ValueError("Must specify max_steps when tracing")
            if self.max_steps >= 10:
                raise ValueError("Tracing more than 10 steps is not recommended")
        return self

    @model_validator(mode="after")
    def validate_lora_adapter_saving(self):
        if self.ckpt and self.ckpt.weights and self.ckpt.weights.save_adapter_separately:
            lora_enabled = self.model and self.model.lora
            if not lora_enabled:
                raise ValueError("save_adapter_separately=True requires LoRA to be enabled")
        return self

    @model_validator(mode="after")
    def validate_weight_broadcast_type(self):
        if self.weight_broadcast.type == "nccl" and self.max_async_level != 1:
            raise ValueError("NCCL weight broadcast only works with async level 1")
        return self

    @model_validator(mode="after")
    def validate_opt_and_fsdp_offload(self):
        if self.optim.type == "muon" and self.model.fsdp_cpu_offload:
            raise ValueError("Muon optimizer does not support FSDP CPU offload")
        return self

    @model_validator(mode="after")
    def validate_lora_broadcast(self):
        if self.model.lora is not None and self.weight_broadcast.type == "nccl":
            raise ValueError("NCCL weight broadcast does not support LoRA yet.")
        return self

    @model_validator(mode="after")
    def auto_setup_tokenizer(self):
        if self.tokenizer.name is None:
            self.tokenizer.name = self.model.name
        if self.tokenizer.trust_remote_code is None:
            self.tokenizer.trust_remote_code = self.model.trust_remote_code
        return self

    @model_validator(mode="after")
    def auto_setup_fused_lm_head_chunk_size(self):
        if self.model.fused_lm_head_chunk_size == "auto":
            if self.model.impl == "liger_kernel":
                self.model.fused_lm_head_chunk_size = "disabled"
            else:
                self.model.fused_lm_head_chunk_size = 2048
        return self

    @model_validator(mode="after")
    def ep_only_with_custom_impl(self):
        if self.model.ep > 1 and self.model.impl not in ("custom", "auto"):
            raise ValueError("EP is only supported with the custom implementation or auto mode")
        return self
