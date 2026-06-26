import msgspec


class TrainingSample(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A single rollout sample sent from orchestrator to trainer."""

    # Public fields expected by the trainer/orchestrator contract:
    # prompt_ids, prompt_mask, completion_ids, completion_mask,
    # completion_logprobs, advantage, reward.
    pass


class TrainingBatch(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A batch of rollout samples plus transport metadata."""

    # Public fields expected by the transport contract:
    # examples, temperature, step, plus any run-identity metadata needed for
    # multi-run routing.
    pass


class MicroBatch(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A packed micro-batch sent from packer to trainer workers."""

    # Public fields expected by the trainer worker contract:
    # input_ids, loss_mask, advantages, inference_logprobs, position_ids,
    # temperature, plus optional trainer metadata.
    pass
