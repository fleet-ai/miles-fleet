# DUPO training handoff

The active matrix is **0.8B, 2B and 4B × GRPO and DUPO**. Both 9B training
jobs were cancelled by the user; do not recreate them. Preserve the original
`deepscaler` training reward. The parent's separate final-integer rescoring is
analysis only. Record every subsequent experiment decision in the canonical
`dataminer_v2/experiments/dupo-v001/protocol.md` before execution.

## Running jobs

Observed 2026-09-13 approximately 14:50 UTC. Every row has one GPU, q1/c1 and is
active on context `nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6`, namespace
`fleet-train-jobs`. Names below all end in `-train-01539b81`.

| Job prefix | Completed optimizer steps | Provisional total | W&B run |
| --- | ---: | ---: | --- |
| neeraj-grpo-0p8b | 0 | 334 | 4bbznwon |
| neeraj-dupo-0p8b | 0 | 334 | oasf552e |
| neeraj-grpo-2b | 3 | 141 | dmmnzqdb |
| neeraj-dupo-2b | 3 | 141 | 70jxu61m |
| neeraj-grpo-4b | 2 | 63 | ddwyj2r2 |
| neeraj-dupo-4b | 2 | 63 | xrybmkt3 |

W&B: `https://wandb.ai/thefleet/dynamic-ungrouped-po/runs/<run-id>`.
All observed completed steps logged `outcome=NORMAL valid_step=true`.
The 0.8B jobs were submitted after the explicit restoration decision and were
still loading at this snapshot. Its prior two-step execution smokes passed.
The next agent must inspect live status; this table is not a live monitor.

Outputs: `/sfs/neeraj/dupo-v001/training/<full-job-name>/`.
Each contains `settings.json`, source hashes and `rollouts/<id>.pt`.
DUPO additionally writes `checkpoints/dupo_steps/<id>.json` with all graded
rewards, accepted/rejected IDs, groups and posterior updates. Model/data/Beta
checkpoints are saved every ten steps and at completion. At this snapshot no
full job had reached its first ten-step save; no full-run HF export existed.
Do not claim full-run checkpoint validation yet.

## Recipe and cost status

All jobs restart exact official Qwen3.5 weights. The shared balanced schedule is
`/sfs/neeraj/dupo-v001/training/data-v1/train.jsonl`, SHA256
`5de15338859d28980c0a2a9b060c58c4e5a25f824b41e6132d8a2e8a68e19879`.
Its 22419 rows cover GSM8K train and the fixed 20+20 AIME training subsets;
no rows were removed by the 2048 prompt-token filter. Validation stays separate.

Shared recipe: 3 prompts ×4 trajectories, no oversampling, 8192 response cap,
thinking enabled, mean-only reward normalization for BOTH algorithms, Adam
LR2e-6, KL coefficient0.001, full-layer recomputation, TP=PP=CP=DP=1.
DUPO uses cumulative Beta(1,1) by explicit task type, frozen pre-step means,
`min(1,4p(1-p)+0.05)`, groups4 including partials, and unnormalized centered rewards.

The provisional target is 2e17 logical FLOPs per arm. Step estimates assume
256 prompt tokens, 4096 response tokens, 12 retained trajectories, separate
reference/policy scoring, forward/backward and full recomputation. **These are
not equal measured compute runs.** The user explicitly chose estimate-first
launches with later manual adjustment. Online compute/LR hooks in this branch
are opt-in and DISABLED in the running source. Initial jobs use constant LR.

The initial DUPO trace omitted rejected token lengths. Versioned lower/upper
bounds are available from `summarize_training_compute.py`; supply the actual
last completed rollout from logs and actual trainable parameter count:
0.8B873438784, 2B2274069824, 4B4659865088. Do not count merely generated batches
as trained. The later local source adds token/cache metadata to every DUPO row.

SGLang request dumps were enabled without restarting 2B/4B engines at
14:48:47–14:48:54 UTC. Each run has `request-cost-logging.json` and
`sglang-requests/*.pkl` (12 completions per file), retaining cached/prompt/output
counts even for rejected samples. Earlier requests and a final incomplete
12-request group can be missing. One GRPO2B dump was confirmed by14:52 UTC;
verify other dumps and match timestamps/request IDs before using them. Both 0.8B engines became ready and their request logging was enabled
during the first generation at approximately14:53 UTC; exact timestamps are
in `evidence/0p8b-request-logging.json`. No further enablement is needed.

## Reproducibility and verification

The exact running source is `evidence/training-01539b81-source.tar.gz`, SHA256
`01539b819cb3a9d9b3e07faf5e606e6be5b273bf55db4cfe1dfe722b65a43f85`.
Its relative file hashes, job manifests, compressed logs and status snapshot
are beside it. The current branch includes later unexecuted counter/report
improvements; it is not byte-identical to that archive. The pinned image is
`radixark/miles@sha256:a7ef79b5c0d0cb1a5d6ab0f39a2693ea63ea369d9339d04b7b3e0b7ec23b5dfe`.
Runtime source is Miles2799fe386 plus the recorded patch, over the image's
native dbbab1566ae438f7202fff653eae938e07b1d4b6 environment. An optional session
adapter import was made lazy because the image's SGLang lacks that newer adapter.
No fake imports or model stubs are used in training.

`evidence/dupo-smoke-evidence.tar.gz` preserves both actual training-dev smoke
rollout/train tensors and posterior states. SHA256:
`3dc831055f690563e25d866ab941522d6a1b3e123daf38a54dc6f506626c3548`.
GRPO and DUPO completed two optimizer steps. DUPO graded16, accepted14;
final arithmetic Beta(4,6), AIME Beta(1,9), including two rejected failures.
Smoke weights saw an AIME question and must not initialize full comparisons.

Verification: 19 scoped algorithm/compute tests pass locally, including actual
reward conversion/DP scheduling/advantage tensors, nonbinary/singleton/all-zero
groups, resume and empty-step handling. Four 0.8B/2B model snapshot checks pass.
Earlier pinned GPU-image execution passed all14 algorithm/collection tests and
both training smokes; 35 combined existing/DUPO conversion checks passed locally.
The broad launcher attempt had360 passes,24 failures,122 errors from unavailable
local dependencies and Linux paths; it is not claimed fully passing. The later
CPU-node runtime counter check failed to import Transformer Engine because
`libcuda.so.1` is unavailable on that CPU node. New online counter hooks still
need GPU validation; do not enable them silently in running experiments.

## Next actions

1. Inspect the six live jobs and confirm 0.8B optimizer progress. Do not create duplicate jobs.
2. Confirm each first ten-step checkpoint, HF export and DUPO posterior save.
3. Reconcile actual costs from completed-step traces/request dumps, expose
   uncovered generation bounds, and record any step adjustment before acting.
4. If resuming, use the same run's checkpoint and data/Beta state. The newer
   launcher supports `--resume-checkpoint <run>/checkpoints` and a revised
   total `--estimated-rollout-steps`; it preserves original launch settings.
5. Evaluate paired checkpoints on the frozen held-out GSM8K/AIME subsets only:
   GSM8K one sample/problem, AIME64, as recorded in the parent protocol.
6. Finish analysis and handoff. No further model/dataset/reward expansion.

Before the artifact commit, every hook in `pre-commit run --all-files` passed.
The complete result is preserved in `evidence/pre-commit-final.log`.
