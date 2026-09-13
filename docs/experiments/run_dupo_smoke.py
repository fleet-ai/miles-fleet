"""Run exactly two synchronous DUPO steps on one GPU using official Qwen3.5-0.8B.

Requires a GPU Miles runtime and a writable output directory. Example:
  python docs/experiments/run_dupo_smoke.py --output-dir /sfs/neeraj/dupo-v001/smoke/<run-id>
"""

import json
import os
import shlex
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import typer
from huggingface_hub import snapshot_download

import miles.utils.external_utils.command_utils as U

REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
AIME_REVISION = "1c625e328db94ec7ef7ff169016b097c468d60b9"


REWARDS = {
    "deepscaler": "--rm-type deepscaler",
    "math-answer": "--custom-rm-path miles.rollout.rm_hub.math_answer.miles_batched_reward",
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    algorithm: str = "dupo"
    model_dir: str = "/sfs/neeraj/dupo-v001/smoke-models"
    megatron_path: str = "/root/Megatron-LM"
    reward: str = "deepscaler"
    max_response_len: int = 256


def execute(args):
    if args.algorithm not in ("dupo", "grpo"):
        raise ValueError("algorithm must be dupo or grpo")
    if args.reward not in REWARDS:
        raise ValueError(f"reward must be one of {sorted(REWARDS)}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "source-hashes.json").write_bytes((U.repo_base_dir / "source-hashes.json").read_bytes())
    model = Path(args.model_dir) / REVISION
    snapshot_download("Qwen/Qwen3.5-0.8B", revision=REVISION, local_dir=model)
    url = f"https://huggingface.co/datasets/zhuzilin/aime-2024/resolve/{AIME_REVISION}/aime-2024.jsonl"
    raw = urllib.request.urlopen(url).read().decode()
    aime = json.loads(raw.splitlines()[0])
    records = [
        {
            "prompt": [{"role": "user", "content": "What is 1 + 1?" + (" Put your final answer in \\boxed{}." if args.reward == "deepscaler" else "")}],
            "label": "2",
            "metadata": {"task_id": "arithmetic-addition-1", "task_type": "arithmetic"},
        },
        {"prompt": aime["prompt"], "label": aime["label"], "metadata": {"task_id": "aime2024-0", "task_type": "aime"}},
    ]
    data = output / "problems.jsonl"
    data.write_text("".join(json.dumps(record) + "\n" for record in records))
    q = shlex.quote
    train_args = (
        f"--hf-checkpoint {q(str(model))} --ref-load {q(str(model))} --load {q(str(model))} "
        f"--save {q(str(output / 'checkpoints'))} --save-interval 1 "
        f"--save-hf {q(str(output / 'hf' / '{rollout_id}'))} --megatron-to-hf-mode bridge "
        f"--prompt-data {q(str(data))} --input-key prompt --label-key label --metadata-key metadata "
        "--apply-chat-template --apply-chat-template-kwargs '{\"enable_thinking\": true}' "
        "--rollout-function-path miles.rollout.sglang_rollout.generate_rollout "
        f"{REWARDS[args.reward]} --num-rollout 2 --rollout-batch-size 2 --over-sampling-batch-size 2 "
        f"--n-samples-per-prompt 4 --rollout-max-response-len {args.max_response_len} --rollout-temperature 0.6 "
        "--rollout-top-p 0.95 --rollout-top-k 20 --global-batch-size 8 "
        "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 4096 --use-dynamic-global-batch-size "
        "--dupo-group-size 4 --dupo-epsilon 0.05 --dupo-threshold 1 --dupo-retention symmetric "
        "--advantage-estimator grpo --disable-grpo-std-normalization --calculate-per-token-loss "
        "--use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl "
        "--optimizer adam --lr 2e-6 --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 --attention-dropout 0.0 --hidden-dropout 0.0 "
        "--clip-grad 1.0 --bf16 --use-distributed-optimizer "
        "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.15 --sglang-disable-cuda-graph --sglang-max-running-requests 8 "
        "--colocate --actor-num-nodes 1 --actor-num-gpus-per-node 1 --num-gpus-per-node 1 "
        f"--save-debug-rollout-data {q(str(output / 'rollouts' / '{rollout_id}.pt'))} "
        f"--save-debug-train-data {q(str(output / 'train' / '{rollout_id}_{rank}.pt'))} "
        "--use-wandb --wandb-project dynamic-ungrouped-po --wandb-group dupo-v001-0p8b-smoke "
        "--disable-wandb-random-suffix "
    )
    if args.algorithm == "dupo":
        train_args += "--dupo "
    train_args = train_args.replace("dupo-v001-0p8b-smoke", f"neeraj-{args.algorithm}-v001-0p8b-smoke")
    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError("The smoke requires the dev cluster's WANDB_API_KEY")
    (output / "settings.json").write_text(
        json.dumps(dict(model_revision=REVISION, aime_revision=AIME_REVISION, argv=shlex.split(train_args)), indent=2)
        + "\n"
    )
    U.execute_train(
        train_args,
        num_gpus_per_node=1,
        megatron_model_type="qwen3.5-0.8B",
        config=args,
        megatron_path=args.megatron_path,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
