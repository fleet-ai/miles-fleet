"""Train a fresh official Qwen3.5 model against a common logical compute budget.

One GPU, TP=PP=CP=DP=1; provisional runs use full-layer recomputation. The data directory must contain the
frozen balanced train.jsonl and ready.json. Example:
python docs/experiments/run_dupo_training.py --algorithm dupo --model-size 0.8B \
  --model-dir /sfs/models/0p8b --data-dir /sfs/neeraj/dupo-v001/training/data-v1 \
  --compute-budget-flops 1e17 --output-dir /sfs/neeraj/dupo-v001/training/example
"""

import hashlib
import json
import os
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import typer
from huggingface_hub import snapshot_download

import miles.utils.external_utils.command_utils as U
from miles.rollout.compute_accounting import VERSION, Qwen35Flops

REWARDS = {
    "deepscaler": "--rm-type deepscaler",
    "math-answer": "--custom-rm-path miles.rollout.rm_hub.math_answer.miles_reward",
    # math-answer with -0.5 for a capped response without a scorable answer (protocol 09-14).
    "math-answer-tp05": "--custom-rm-path miles.rollout.rm_hub.math_answer.miles_reward_tp05",
}

REVISIONS = {
    "0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
    "2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    algorithm: str = "dupo"
    model_size: str = "2B"
    model_dir: str = ""
    data_dir: str = ""
    compute_budget_flops: float = 2e17
    estimated_rollout_steps: int = 0
    enable_compute_budget: bool = False
    resume_checkpoint: str = ""
    overshoot_tolerance: float = 0.05
    checkpoint_count: int = 4
    max_response_len: int = 8192
    max_prompt_len: int = 2048
    rollout_batch_size: int = 3
    megatron_path: str = "/root/Megatron-LM"
    # Training reward: "deepscaler" is Miles' boxed-only grader (the canonical
    # condition); "math-answer" is miles.rollout.rm_hub.math_answer, which
    # reads the committed final answer in any format.
    reward: str = "deepscaler"
    # SHA-256 of the frozen train.jsonl this run must use.
    expected_train_sha256: str = "5de15338859d28980c0a2a9b060c58c4e5a25f824b41e6132d8a2e8a68e19879"
    # W&B run name (Miles names the run after the group); default: the output directory name.
    wandb_group: str = ""
    # Optimizer learning rate (constant) and samples per prompt (the GRPO group).
    lr: float = 2e-6
    # Coefficient of the low-variance KL loss against the reference (base) checkpoint.
    kl_loss_coef: float = 0.001
    n_samples_per_prompt: int = 4
    # GPUs for the run: colocated, one sglang engine per GPU and Megatron data
    # parallel across them. Scale rollout_batch_size with it; the FLOP budget
    # is per run, not per GPU.
    gpus: int = 1
    # Drop prompt groups whose rewards have zero variance (all correct or all
    # wrong) and keep sampling until the batch is full (DAPO-style dynamic
    # sampling); over-sampling granularity is 2x the batch.
    dynamic_filter: bool = False
    # Checkpoint cadence in optimizer steps (provisional runs only).
    save_interval: int = 10


def execute(args):
    if args.algorithm not in ("dupo", "grpo") or args.model_size not in ("0.8B", "2B", "4B"):
        raise ValueError("Full training is limited to GRPO/DUPO on 0.8B/2B/4B")
    if not args.model_dir or not args.data_dir or args.compute_budget_flops <= 0:
        raise ValueError("Explicit model/data paths and a positive common compute budget are required")
    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError("Training requires WANDB_API_KEY")
    output, model, data = Path(args.output_dir), Path(args.model_dir), Path(args.data_dir)
    if output.exists() and not args.resume_checkpoint:
        raise ValueError("A fresh run requires a new output directory; resume must explicitly load its checkpoint")
    if args.resume_checkpoint and Path(args.resume_checkpoint).resolve() != (output / "checkpoints").resolve():
        raise ValueError("Resume must load the existing run's own checkpoint directory")
    ready = json.loads((data / "ready.json").read_text())
    train_hash = hashlib.sha256((data / "train.jsonl").read_bytes()).hexdigest()
    if train_hash != args.expected_train_sha256:
        raise ValueError(f"Training data differs from the expected schedule: {train_hash} != {args.expected_train_sha256}")
    if args.reward not in REWARDS:
        raise ValueError(f"reward must be one of {sorted(REWARDS)}")
    snapshot_download(f"Qwen/Qwen3.5-{args.model_size}", revision=REVISIONS[args.model_size], local_dir=model)
    config = json.loads((model / "config.json").read_text())["text_config"]
    estimator = Qwen35Flops(config)
    # Six full forwards bound generation + reference + actor + train + backward.
    # 40B parameters is a conservative bound for every supported dense model.
    max_step = (
        6 * args.rollout_batch_size * args.n_samples_per_prompt * estimator.forward(args.max_prompt_len + args.max_response_len)
        + 15 * 40_000_000_000
    )
    if args.enable_compute_budget and max_step > args.compute_budget_flops * args.overshoot_tolerance:
        raise ValueError(f"Maximum step estimate {max_step} exceeds the declared budget overshoot allowance")
    if args.estimated_rollout_steps < 1 and not args.enable_compute_budget:
        raise ValueError("A positive provisional step count is required when compute hooks are disabled")
    output.mkdir(parents=True, exist_ok=bool(args.resume_checkpoint))
    q = shlex.quote
    train_args = (
        f"--hf-checkpoint {q(str(model))} --ref-load {q(str(model))} --load {q(args.resume_checkpoint or str(model))} "
        f"--save {q(str(output / 'checkpoints'))} --save-hf {q(str(output / 'hf' / '{rollout_id}'))} "
        "--megatron-to-hf-mode bridge "
        f"--prompt-data {q(str(data / 'train.jsonl'))} --input-key prompt --label-key label --metadata-key metadata "
        "--apply-chat-template --apply-chat-template-kwargs '{\"enable_thinking\": true}' --rollout-shuffle "
        f"--rollout-function-path miles.rollout.sglang_rollout.generate_rollout {REWARDS[args.reward]} "
        f"--num-rollout {100000 if args.enable_compute_budget else args.estimated_rollout_steps} --rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} --rollout-max-response-len {args.max_response_len} --rollout-max-prompt-len {args.max_prompt_len} "
        f"--global-batch-size {args.rollout_batch_size * args.n_samples_per_prompt} --rollout-seed 1234 --seed 1234 "
        "--rollout-temperature 0.6 --rollout-top-p 0.95 --rollout-top-k 20 "
        "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 "
        f"--use-dynamic-batch-size --max-tokens-per-gpu {args.max_prompt_len + args.max_response_len + 512} --use-dynamic-global-batch-size "
        "--advantage-estimator grpo --calculate-per-token-loss --disable-grpo-std-normalization "
        f"--use-kl-loss --kl-loss-coef {args.kl_loss_coef} --kl-loss-type low_var_kl "
        f"--optimizer adam --lr {args.lr} --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
        "--attention-dropout 0.0 --hidden-dropout 0.0 --clip-grad 1.0 --bf16 --use-distributed-optimizer "
        "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.15 --sglang-disable-cuda-graph "
        f"--sglang-max-running-requests {max(1, (args.rollout_batch_size * args.n_samples_per_prompt + args.gpus - 1) // args.gpus)} "
        f"--colocate --actor-num-nodes 1 --actor-num-gpus-per-node {args.gpus} --num-gpus-per-node {args.gpus} "
        f"--save-debug-rollout-data {q(str(output / 'rollouts' / '{rollout_id}.pt'))} "
        f"--use-wandb --wandb-project dynamic-ungrouped-po --wandb-group {q(args.wandb_group or output.name)} "
        "--disable-wandb-random-suffix "
    )
    if args.enable_compute_budget:
        train_args += f"--compute-budget-flops {args.compute_budget_flops} --compute-checkpoint-count {args.checkpoint_count} --compute-overshoot-tolerance {args.overshoot_tolerance} "
    else:
        train_args += (
            f"--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --save-interval {args.save_interval} "
        )
    if not args.dynamic_filter:
        train_args += f"--over-sampling-batch-size {args.rollout_batch_size} "
    if args.dynamic_filter:
        train_args += (
            "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
            f"--over-sampling-batch-size {2 * args.rollout_batch_size} "
        )
    if args.algorithm == "dupo":
        train_args += "--dupo --dupo-group-size 4 --dupo-epsilon 0.05 --dupo-threshold 1 --dupo-retention symmetric "
    settings = dict(
        gpus=args.gpus,
        lr=args.lr,
        kl_loss_coef=args.kl_loss_coef,
        n_samples_per_prompt=args.n_samples_per_prompt,
        rollout_batch_size=args.rollout_batch_size,
        dynamic_filter=args.dynamic_filter,
        save_interval=args.save_interval,
        max_response_len=args.max_response_len,
        reward=args.reward,
        reward_args=REWARDS[args.reward],
        provisional_budget=args.compute_budget_flops,
        compute_hooks_enabled=args.enable_compute_budget,
        estimator=VERSION,
        model_revision=REVISIONS[args.model_size],
        train_sha256=train_hash,
        dataset_ready=ready,
        max_step_flops=max_step,
        argv=shlex.split(train_args),
    )
    record = output
    if args.resume_checkpoint:
        record = output / "resumes" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        record.mkdir(parents=True)
    (record / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    (record / "source-hashes.json").write_bytes((U.repo_base_dir / "source-hashes.json").read_bytes())
    U.execute_train(
        train_args,
        num_gpus_per_node=args.gpus,
        megatron_model_type=f"qwen3.5-{args.model_size}",
        config=args,
        megatron_path=args.megatron_path,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
