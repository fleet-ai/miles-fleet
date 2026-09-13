# DUPO v001

Enable `--dupo` on the synchronous `train.py` driver to reject rollouts using
estimated success by task type, then shuffle the survivors into mixed reward
groups. Without this flag the GRPO control keeps its existing behavior.

Each input row must have explicit, nonempty `metadata.task_type` and
`metadata.task_id`. Each rollout must have a unique sample index and one sample;
compact multi-segment rollouts and adapters are not supported. A missing reward
is ungraded; finite scalar rewards (or the selected `--reward-key`) are graded.
The pass threshold defaults to 1.0. Infrastructure failures must return no reward.

Each type starts at Beta(1,1). For each rollout step, DUPO freezes the current
posterior means, keeps eligible rollouts independently with probability
`min(1, 4*p*(1-p) + 0.05)`, then updates the type counts from every graded
observation, including rejected samples and late completions during abort.
The full comparison sets oversampling equal to the requested prompt batch, so
every submitted generation completes before selection; no completion-speed filter is used.
No discounting is applied. `--dupo-retention easy` selects
`min(1, 1-p+epsilon)` instead. Current rewards do not change their own retention
probabilities. Per-step seeded randomness gives the same decisions on resume.

Survivors are shuffled into groups of four (`--dupo-group-size`). Rewards are
centered once before distribution to training ranks, using the true group size,
including a partial final group. No standard-deviation or advantage normalization
is permitted. Original task group indices remain available for auditing.

## Supported training configuration

Use Megatron with pipeline parallelism 1, no virtual pipeline stages, dynamic
token batches, dynamic global batch size, and per-token loss. Tensor parallelism
is supported; data parallelism is supported when survivors can fill every rank.
A nonempty batch smaller than the data-parallel count fails visibly; use DP=1
for small smoke tests. No accepted rollout is silently trimmed or duplicated.
An empty survivor batch skips optimization, keeps Bayesian updates, and follows
the normal model/data checkpoint and evaluation schedule.

Async training, fault-tolerance rewind, compact rollouts, multi-LoRA, custom
reward conversion/filter hooks, and legacy pass-rate logging are rejected.
These modes need additional integration work; they do not silently substitute
a different algorithm.

Required flags in addition to an existing GRPO recipe:

```text
--dupo
--rollout-function-path miles.rollout.sglang_rollout.generate_rollout
--disable-grpo-std-normalization
--use-dynamic-batch-size --max-tokens-per-gpu 4096
--use-dynamic-global-batch-size
--calculate-per-token-loss
--pipeline-model-parallel-size 1
```

`--dupo-epsilon`, `--dupo-threshold`, and `--dupo-group-size` default to
0.05, 1.0, and 4. Keep identical model, data, optimizer, sampling and generated
rollout budgets for the GRPO control. Removing `--dupo` runs the control.

## State and evidence

Posterior counts, configuration, cumulative launched/graded/accepted counts and
last completed rollout ID are saved with each model checkpoint in
`<save>/rollout/dupo_<rollout_id>.json`. Resume requires the matching file and
identical algorithm configuration; missing or incompatible state is an error.
The standard training initialization calls `load(start_rollout_id - 1)` after
resolving the integer start ID; fresh training passes -1 and keeps Beta(1,1).

`<save>/dupo_steps/<rollout_id>.json` records observed sample IDs, task IDs/types,
rewards, candidate membership, acceptance, mixed groups, and per-step metrics.
Logs include each type's parameters/mean before and after updating, keep
probability, acceptance by task/type, nonzero advantage fraction, mean absolute
advantage and the absolute-advantage share on types with posterior mean outside
[0.1,0.9] after updating. Zero total advantage reports zero share.
Cumulative launched counts include all requests, including ungraded/aborted
ones, so generated and accepted budgets can be compared separately.

## Brief training-dev smoke

`run_dupo_smoke.py` downloads the exact official Qwen3.5-0.8B revision
`2fc06364715b967f1860aea9cf38778875588b17`, records its source/configuration,
and runs exactly two steps on one GPU. It uses an addition problem and the
first question from pinned `zhuzilin/aime-2024`, four samples per problem,
256 output tokens, and natural rule-based grading. This verifies execution,
not model accuracy. Use a fresh dedicated output directory for each run:

```text
python docs/experiments/run_dupo_smoke.py --algorithm dupo --output-dir <dupo-output>
python docs/experiments/run_dupo_smoke.py --algorithm grpo --output-dir <grpo-output>
```

Both run on training-dev at q1/c1, one GPU each, in W&B project
`dynamic-ungrouped-po`. Each launch manifest must use a `neeraj-` prefix and
`kueue.x-k8s.io/podset-required-topology: kubernetes.io/hostname` on the pod
metadata for the dev queue's B300 flavor. Do not use the main cluster for smoke
tests. Full training is not authorized by these commands.

## Validation

CPU tests cover frozen priors, rejected graded samples, missing rewards,
checkpoint/resume, all-rejected optimizer skip and subsequent training, explicit
metadata, clipping, current-reward independence, all-zero task contributions,
partial mixed groups through the real reward conversion, DP scheduling at
1/2/3 ranks, and the existing token-level GRPO advantage implementation.

```text
python -m pytest --noconftest tests/fast/rollout/test_dupo.py tests/fast/rollout/test_dupo_collection.py
```

The first 13 cases passed locally. All 14 passed in the pinned Linux Miles image.
Existing GRPO reward/scheduler checks and DUPO cases also passed together (35).
The complete launcher suite was attempted on macOS: 360 passed; Linux `/proc`
and unavailable local dependencies prevented the remaining checks. The new
0.8B model definition snapshot passed. Both two-step training-dev Jobs completed: GRPO `neeraj-grpo-0p8b-smoke-80c36868`
at 2026-09-13 08:06:58 UTC, DUPO `neeraj-dupo-0p8b-smoke-6ae860d9` at 08:11:47 UTC.
DUPO graded 16 rollouts and trained on 14; its final Beta counts were arithmetic
(4,6), AIME (1,9). The two rejected AIME failures were included in the update.

Smoke checkpoints have seen an AIME24 question and must never initialize a
held-out comparison. Start full experiments again from the official weights.
The counters here distinguish generated and accepted rollouts; they are not
a complete FLOPs ledger. Matching total compute requires accounting for
generation, reference/actor scoring and optimizer passes as well.


## Current full runs and compute convention

The initial six q1/c1 Jobs on the main training cluster were named
`neeraj-{grpo,dupo}-{2b,4b,9b}-train-01539b81`. Their manifests are in `jobs/`.
The user restored 0.8B and cancelled both 9B training Jobs after launch.
Only 0.8B/2B/4B × GRPO/DUPO remain active. All runs restart the pinned
Alibaba weights and use the frozen balanced three-dataset schedule, original
`deepscaler` rewards, shared mean-only reward normalization, 12 trajectories per
step and an 8192 output-token limit. Validation data is not passed to training.
W&B project is `dynamic-ungrouped-po`, group `neeraj-dupo-v001-compute-v1`.

These initial runs use provisional 334/141/63 steps for 0.8B/2B/4B, respectively,
against an estimated common budget of 2e17 logical FLOPs. The step estimate
assumes 256 prompt tokens, 4096 response tokens, all trajectories retained and
full-layer recomputation. These runs use constant LR 2e-6, save every ten steps
and at completion, and require manual cost reconciliation and checkpoint resume.
They do not yet establish equal actual compute. The user explicitly requested
launching with estimates instead of waiting for the automatic counter.

`qwen35-logical-v1` counts multiply-add as two operations. It includes dense
projection/MLP matrices, causal full attention, the Gated DeltaNet (GDN) linear
attention state recurrence, convolution, output logits, and fifteen operations
per trainable parameter for Adam and gradient accounting. GDN uses seven
operations per key/value state element plus two per value element. Nonlinear
functions, normalization, communication, padding, kernel overhead and optimizer
memory traffic are excluded. Backward is estimated as twice forward; full-layer
recomputation adds one forward. This is a reproducible arithmetic convention,
not a hardware measurement or exact count of the chunked GDN kernel.

Generation uses the reported cached prefix, excludes the final generated token's
unneeded next forward, and charges only the generated logit positions. Actor and
reference scoring are separate forwards; training is a separate forward/backward.
`summarize_training_compute.py` reads saved accepted trajectories and DUPO
observation reports. In the initial `01539b81` source bundle, rejected token
lengths were not retained; the report exposes lower/upper bounds for their cost.
The later source adds rejected token/cache counts for future checkpoint resumes.
Do not label the initial reports exact or silently impute missing lengths.

The opt-in `--compute-budget-flops` ledger and budget-based LR/checkpoint hooks
are under validation and are disabled in these six initial runs. Its supported
configuration is one GPU, TP=PP=CP=DP=1, Adam, one optimizer pass, separate actor
and reference scoring, no recomputation, no oversampling or speculative decoding.
Its CPU arithmetic/resume tests pass, but pinned CPU-node runtime validation
could not import Transformer Engine without `libcuda.so.1`; full GPU validation
of these new hooks remains outstanding.

The current full-training launcher accepts only 0.8B, 2B and 4B. The original 9B
manifests remain as launch evidence and must not be submitted again.
