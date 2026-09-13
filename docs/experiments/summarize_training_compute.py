"""Estimate completed training cost and expose missing rejected-generation uncertainty.

Run in a Miles environment with the run directory and official checkpoint visible.
The caller must select the last completed optimizer rollout from logs/checkpoints;
rollout files alone do not prove that optimization completed.
"""

import argparse
import json
from pathlib import Path

import torch

from miles.rollout.compute_accounting import VERSION, Qwen35Flops
from miles.utils.types import Sample


def summarize(args):
    root = Path(args.run_dir)
    settings = json.loads((root / "settings.json").read_text())
    argv = settings["argv"]
    model = Path(argv[argv.index("--hf-checkpoint") + 1])
    estimator = Qwen35Flops(json.loads((model / "config.json").read_text())["text_config"])
    max_response = int(argv[argv.index("--rollout-max-response-len") + 1])
    max_prompt = int(argv[argv.index("--rollout-max-prompt-len") + 1])
    recomputation = int("--recompute-granularity" in argv)
    steps = []
    for rollout_id in range(args.through_rollout_id + 1):
        raw = torch.load(root / "rollouts" / f"{rollout_id}.pt", map_location="cpu", weights_only=False)
        samples = [Sample.from_dict(row) for row in raw["samples"]]
        forward = sum(estimator.forward(len(sample.tokens)) for sample in samples)
        generated = sum(estimator.generation(sample) for sample in samples)
        report = root / "checkpoints" / "dupo_steps" / f"{rollout_id}.json"
        missing = 0
        if report.exists():
            rows = json.loads(report.read_text())["observations"]
            for row in rows:
                if row["accepted"]:
                    continue
                if "total_tokens" in row:
                    generated += estimator.forward(
                        row["total_tokens"] - 1,
                        cached=row["prefix_cache"]["cached_tokens"],
                        logits=row["response_tokens"],
                    )
                else:
                    missing += 1
        fixed = generated + (5 + recomputation) * forward + (15 * args.parameters if samples else 0)
        # Initial jobs did not retain rejected lengths. Keep an explicit interval
        # rather than presenting an estimate based on survivors as exact compute.
        low = fixed + missing * estimator.forward(1, logits=1)
        high = fixed + missing * estimator.forward(max_prompt + max_response - 1, logits=max_response)
        steps.append(
            dict(
                rollout_id=rollout_id,
                accepted=len(samples),
                missing_rejected_lengths=missing,
                known_flops=fixed,
                lower_flops=low,
                upper_flops=high,
            )
        )
    result = dict(
        estimator=VERSION,
        hardware_measured=False,
        run_dir=str(root),
        completed_through=args.through_rollout_id,
        stages="generation + ref + actor + train + backward + recompute + Adam",
        lower_flops=sum(s["lower_flops"] for s in steps),
        upper_flops=sum(s["upper_flops"] for s in steps),
        steps=steps,
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "steps"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--through-rollout-id", required=True, type=int)
    parser.add_argument(
        "--parameters", required=True, type=int, help="Actual trainable parameter count from trainer log"
    )
    parser.add_argument("--output", required=True)
    summarize(parser.parse_args())


if __name__ == "__main__":
    main()
