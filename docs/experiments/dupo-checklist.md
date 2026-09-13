# DUPO implementation checklist

- [x] Inspect protocol, current Miles source, repository rules, and check commands.
- [x] Create isolated worktree on `codex/dupo-v001` from `origin/main`.
- [x] Implement opt-in type-level Bayesian rejection and mixed reward groups.
- [x] Preserve all graded observations, checkpoint state, and exact resume.
- [x] Preserve survivor groups and counts through training conversion and scheduling.
- [x] Handle empty batches without optimizer updates; reject unsupported execution modes.
- [x] Exercise algorithm, actual conversion/scheduling/advantage path, resume, and GRPO control.
- [x] Run all repository pre-commit hooks and scoped CPU checks; record unavailable runtime checks.
- [x] Document reproduction settings, input contract, limits, and portable handoff.

- [x] Pass 14 CPU tests in the pinned Linux image; pass 35 local DUPO/GRPO conversion checks.
- [x] Complete exactly two real GRPO and DUPO training steps, one GPU each, on training-dev only.
- [x] Record each job status, W&B run, checkpoint path, and DUPO posterior evidence.

- [ ] Implement and test versioned Qwen3.5 logical FLOP ledger, budget stopping/checkpoints and LR.
- [ ] Freeze one shared all-model compute budget with bounded overshoot.
- [ ] Verify shared full-training launcher and all four model configurations.
- [x] Preserve both smoke evidence sets and record independent reward/group audit.

- [x] Launch the six user-authorized full jobs (2B/4B/9B), original grader, shared no-std, estimated steps.
- [ ] Verify full jobs progress through optimizer steps and checkpoint saves.
- [ ] Reconcile actual costs and manually adjust/resume toward the common estimate.
- [ ] Evaluate only frozen held-out data and complete the comparison.

- [x] Narrow active matrix to 2B/4B × GRPO/DUPO after the user cancelled 9B.

- [x] Restore both 0.8B arms at 334 provisional steps, recorded before submission; leave 9B cancelled.
- [x] Observe successful optimizer steps in both 2B/4B algorithms.
- [ ] Observe first 0.8B full-training steps and enable its recorded request-cost logging.
- [ ] Verify first ten-step checkpoints in the full jobs.
