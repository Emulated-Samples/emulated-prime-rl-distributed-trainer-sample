# Prime-RL Distributed Trainer

> **Notice:** This repository is public for limited research review only. It is
> **not open source**. The data and tasks are proprietary IP of Emulated, Inc.
> and may not be copied, redistributed, used for model training, used to create
> derivative datasets/environments, or used commercially without written
> permission. See [`LICENSE`](./LICENSE). Contact: founders@emulated.so
>
> The `workspace/` tree is third-party prime-rl source under the Apache License 2.0 (`workspace/LICENSE`) and is not covered by the proprietary terms above.

This sample asks an agent to implement the trainer-side pieces of a distributed reinforcement learning system and deploy the result to an EKS cluster.

The agent works in the Prime-RL codebase, implements GRPO training logic, fills in weight-sync and transport surfaces, configures shared storage, deploys the three-pod training system, and leaves a manifest that the graders can verify. The environment runs on a regular GPU sandbox while simulating the distributed cluster shape through Kubernetes, EFS, trainer/inference/orchestrator pods, and source override mechanics.

## Quick start

- [Problem prompt](environment/problems.yaml): the exact agent-facing task, including trainer, weight-sync, EKS, EFS, and manifest requirements.
- [Process rubric](environment/src/tests/process-rubric.ts): criteria for codebase understanding, verification behavior, deployment checks, and cluster workflow.
- [Code rubric](environment/src/tests/code-rubric.ts): criteria for GRPO loss correctness, packed sequence handling, teacher/KL terms, transport, broadcast, and checkpoint behavior.
- [Deterministic graders](environment/src/index.ts): the test entrypoint showing the grader groups for prerequisites, cluster state, GPU operator, storage, Helm deployment, trainer units, weight sync, training verification, training quality, and manifest verification.

## Grader walkthrough

This environment combines ML training implementation with infrastructure deployment. It checks whether an agent can reason through a distributed RL training path, implement the missing trainer and weight-sync code, and prove that the deployed system is running the code it edited.

The graders look for:

- GRPO loss logic over packed variable-length sequences
- correct stop-gradient behavior for policy-gradient loss coefficients
- teacher/KL handling through the configured loss weights
- filesystem and ZMQ transport paths that preserve ordering, run identity, and batch boundaries
- broadcast/checkpoint logic that gives inference stable per-run checkpoints
- EFS-backed shared storage, Helm deployment, pod health checks, and manifest output
- verification that the trainer pod is running the workspace source via the chart's source override path
