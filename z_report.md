# Prime-RL QA Report

## Archive

- Archived the prior Opus rollout to `/hyperfocal/z-qa-report/medium-distributed-trainer/claude-opus-4-7/rollout-01-63pct`.
- Pushed the archive to `Hyperfocal-AI-Miscellaneous/qa-report` at commit `f47e21d`.
- The archived run remains useful as a diagnostic trace, but not as a clean VC-facing datapoint because it exposed a harness false failure and a runtime-image inspection bypass.

## Confirmed Issues

- `Training Makes Progress` was a harness issue: it failed from a limited trainer-log tail, while the immediately following completion, checkpoint, and benchmark tests proved the run finished.
- `Checkpoint Saved` used imprecise wording: it counted general output files under `/data/outputs`, including rollout `.bin` files, while the stronger checkpoint-valid test checked real safetensors artifacts.
- The agent could still create custom Kubernetes pods/jobs via manifest files and read their logs, which allowed runtime-image/source inspection despite direct `kubectl run`, `exec`, and `cp` being blocked.

## Fixes Applied

- Updated the training-progress test to look at a larger log window, accept completion markers, and perform a full-log fallback before failing.
- Updated checkpoint logging to report checkpoint artifacts (`STABLE`, safetensors, or weights step artifacts) rather than generic output files.
- Added a Prime-RL-only `kubectl` wrapper in the agent PATH. It blocks `exec`, `cp`, `run`, `debug`, `attach`, ad hoc workload manifests, imperative workload creation, and logs from non-Prime-RL release pods.
- Added defense-in-depth Claude Code deny patterns for direct `/usr/bin/kubectl` and `/usr/local/bin/kubectl` bypass attempts.

## Verification

- `npm run build` passed in `packages/env-base`.
- `npm run build` passed in `packages/env-orchestrator`.
- `npm run build` passed in `environment`.
- Narrative coverage check: 56 active tests, 56 narratives, no missing IDs, no duplicate IDs.
- Narrative reason check: no malformed messages, no old `PASS:`/`FAIL:` prefixes inside reasons, and no bare `oracle` wording.
- Synthetic logging check: PASS/FAIL evaluation outcomes emit as info-level lines; skipped tests do not emit outcome lines; failed/partial tests do not use error-level logging.
- Kubectl wrapper check: release-pod logs and storage manifests are allowed; custom pod logs, pod manifests, and `kubectl exec` are blocked.

## Latest Rollout Review

- Latest Opus rollout finished with raw granular score 84% and weighted score 70.2/100.
- Deployment/runtime integration scored 100% and training outcome scored 100%: pods ran, trainer completed all 20 steps, checkpoint validation passed, and vf-eval reward was 0.814.
- Core trainer algorithm scored 28%: deterministic unit probes exposed genuine implementation defects in `compute_loss`, including list-vs-tensor handling and incorrect loss/masking behavior.
- The previous `Training Makes Progress` false failure is fixed: it passed with reason `trainer log showed RL trainer finished`.
- The previous checkpoint wording issue is fixed: `Checkpoint Saved` now reports checkpoint artifacts, and `Training Checkpoint Valid` confirmed one safetensors checkpoint at 1.40GB.
- Runtime-image source extraction was blocked. The agent attempted `kubectl exec` several times, including `/app` and `/data` inspection, and each attempt was denied. No custom inspection pod or image-source copy appeared.
- A follow-up harness issue was found after the rollout: rubric tests above their configured pass threshold were still displayed as partial because generic score status treated every score below 1.0 as partial. This affected the `Agent Process Quality` display at 69% versus its 60% threshold, not the weighted score. The status mapper has been fixed so future rubric tests use their `success` threshold result.

## Readiness

The environment is materially stronger after the fixes. The latest rollout is useful evidence that the task produces high-signal failures, but a clean VC-facing run should be rerun after the rubric-status display fix so the process-quality status and final counts are not misleading.
