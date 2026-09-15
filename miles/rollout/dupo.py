"""Dynamic ungrouped policy optimization: type-level rejection and shuffled groups."""

import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from miles.utils.types import Sample


@dataclass(frozen=True)
class DupoConfig:
    group_size: int = 4
    epsilon: float = 0.05
    threshold: float = 1.0
    retention: str = "symmetric"
    seed: int = 1234

    def __post_init__(self):
        if self.group_size < 1 or not 0 <= self.epsilon <= 1 or not math.isfinite(self.threshold):
            raise ValueError("DUPO requires group_size >= 1, epsilon in [0, 1], and a finite threshold")
        if self.retention not in ("symmetric", "easy"):
            raise ValueError("DUPO retention must be symmetric or easy")


def keep_probability(mean: float, config: DupoConfig) -> float:
    probability = 4 * mean * (1 - mean) if config.retention == "symmetric" else 1 - mean
    return min(1.0, probability + config.epsilon)


def flatten_rollouts(groups: list[list[Sample]]) -> list[Sample]:
    samples = [sample for group in groups for sample in group]
    if any(not isinstance(sample, Sample) for sample in samples):
        raise ValueError("DUPO currently requires one Sample per rollout, without compact segments")
    return samples


def validate_sample(sample: Sample) -> None:
    for key in ("task_type", "task_id"):
        if not isinstance(sample.metadata.get(key), str) or not sample.metadata[key]:
            raise ValueError(f"DUPO requires explicit nonempty metadata[{key!r}] on every rollout")
    if sample.index is None or sample.adapter is not None:
        raise ValueError("DUPO requires unique sample indices and no adapters")


def validate_single_segment(samples: list[Sample]) -> None:
    """Reject compact multi-segment rollouts: siblings of one rollout share a
    ``rollout_id``. A rollout_id on a lone sample (the agentic generator sets
    one per rollout) is fine; the same id on two samples is not."""
    ids = [sample.rollout_id for sample in samples if sample.rollout_id is not None]
    if len(ids) != len(set(ids)):
        raise ValueError("DUPO requires one Sample per rollout; compact multi-segment rollouts are not supported")


def graded_reward(sample: Sample, args) -> float | None:
    # An aborted sample is ungraded whatever its reward field holds: reward
    # hooks that must not raise mark ABORTED and return a placeholder.
    if sample.reward is None or sample.status == Sample.Status.ABORTED:
        return None
    reward = sample.get_reward_value(args)
    if reward is None:
        return None
    reward = float(reward)
    if not math.isfinite(reward):
        raise ValueError(f"DUPO received nonfinite reward for sample {sample.index}")
    return reward


@dataclass
class DupoState:
    config: DupoConfig
    counts: dict[str, list[int]] = field(default_factory=dict)
    last_rollout_id: int = -1
    total_launched: int = 0
    total_graded: int = 0
    total_accepted: int = 0

    def process(self, args, rollout_id, selected, observations, launched_count):
        if rollout_id != self.last_rollout_id + 1:
            raise ValueError(f"DUPO expected rollout {self.last_rollout_id + 1}, received {rollout_id}")
        candidates = flatten_rollouts(selected)
        samples = flatten_rollouts(observations)
        for sample in samples:
            validate_sample(sample)
        validate_single_segment(samples)
        if len({sample.index for sample in samples}) != len(samples) or launched_count < len(samples):
            raise ValueError("DUPO observations must have unique sample indices within the launched population")
        observed = {sample.index: sample for sample in samples}
        if len({sample.index for sample in candidates}) != len(candidates):
            raise ValueError("DUPO candidates contain duplicate sample indices")
        if any(observed.get(sample.index) is not sample for sample in candidates):
            raise ValueError("DUPO candidates must be drawn from the recorded observations")
        before = {key: list(value) for key, value in self.counts.items()}
        for sample in samples:
            before.setdefault(sample.metadata["task_type"], [1, 1])
        means = {key: alpha / (alpha + beta) for key, (alpha, beta) in before.items()}
        rewards = {sample.index: graded_reward(sample, args) for sample in samples}
        # A per-step seed makes resume exact without depending on unrelated RNG use.
        rng = random.Random(self.config.seed + rollout_id)
        accepted = [
            sample
            for sample in sorted(candidates, key=lambda item: item.index)
            if rewards[sample.index] is not None
            and not sample.remove_sample
            and rng.random() < keep_probability(means[sample.metadata["task_type"]], self.config)
        ]
        rng.shuffle(accepted)
        groups = [
            accepted[start : start + self.config.group_size]
            for start in range(0, len(accepted), self.config.group_size)
        ]
        after = {key: list(value) for key, value in before.items()}
        for sample in samples:
            if (reward := rewards[sample.index]) is not None:
                after[sample.metadata["task_type"]][0 if reward >= self.config.threshold else 1] += 1
        metrics = step_metrics(self.config, before, after, groups, rewards)
        for sample in samples:
            metrics.setdefault(f"dupo/task/{sample.metadata['task_id']}/accepted", 0)
        graded = sum(reward is not None for reward in rewards.values())
        self.counts = after
        self.last_rollout_id = rollout_id
        self.total_launched += launched_count
        self.total_graded += graded
        self.total_accepted += len(accepted)
        metrics.update(
            {
                "dupo/launched": launched_count,
                "dupo/graded": graded,
                "dupo/ungraded": launched_count - graded,
                "dupo/eligible": len(candidates),
                "dupo/accepted": len(accepted),
                "dupo/total_launched": self.total_launched,
                "dupo/total_graded": self.total_graded,
                "dupo/total_accepted": self.total_accepted,
            }
        )
        return groups, metrics

    def save(self, root: str, rollout_id: int) -> None:
        if rollout_id != self.last_rollout_id:
            raise ValueError("DUPO checkpoint and model rollout IDs differ")
        path = Path(root) / "rollout" / f"dupo_{rollout_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), sort_keys=True) + "\n")

    def load(self, root: str, rollout_id: int) -> None:
        if rollout_id < 0:
            return
        payload = json.loads((Path(root) / "rollout" / f"dupo_{rollout_id}.json").read_text())
        if payload["config"] != asdict(self.config) or payload["last_rollout_id"] != rollout_id:
            raise ValueError("DUPO checkpoint configuration or rollout ID differs from this run")
        counts = payload["counts"]
        if any(len(value) != 2 or any(type(n) is not int or n < 1 for n in value) for value in counts.values()):
            raise ValueError("DUPO checkpoint must contain positive integer Beta parameters")
        self.counts = counts
        self.last_rollout_id = rollout_id
        self.total_launched = payload["total_launched"]
        self.total_graded = payload["total_graded"]
        self.total_accepted = payload["total_accepted"]


def step_metrics(config, before, after, groups, rewards):
    accepted = [sample for group in groups for sample in group]
    type_counts = Counter(sample.metadata["task_type"] for sample in accepted)
    task_counts = Counter(sample.metadata["task_id"] for sample in accepted)
    metrics = {f"dupo/task/{key}/accepted": count for key, count in task_counts.items()}
    mass = Counter()
    nonzero = 0
    for group in groups:
        mean = sum(rewards[sample.index] for sample in group) / len(group)
        for sample in group:
            advantage = rewards[sample.index] - mean
            mass[sample.metadata["task_type"]] += abs(advantage)
            nonzero += advantage != 0
    total_mass = sum(mass.values())
    saturated_mass = 0
    for key, (alpha, beta) in before.items():
        updated_alpha, updated_beta = after[key]
        mean = alpha / (alpha + beta)
        after_mean = updated_alpha / (updated_alpha + updated_beta)
        values = dict(
            alpha_before=alpha,
            beta_before=beta,
            mean_before=mean,
            alpha_after=updated_alpha,
            beta_after=updated_beta,
            mean_after=after_mean,
            accepted=type_counts[key],
            keep_probability=keep_probability(mean, config),
        )
        metrics.update({f"dupo/type/{key}/{name}": value for name, value in values.items()})
        if after_mean > 0.9 or after_mean < 0.1:
            saturated_mass += mass[key]
    graded = [reward for reward in rewards.values() if reward is not None]
    metrics["dupo/graded_reward_mean"] = sum(graded) / len(graded) if graded else 0.0
    metrics["dupo/graded_pass_rate"] = (
        sum(reward >= config.threshold for reward in graded) / len(graded) if graded else 0.0
    )
    metrics["dupo/nonzero_advantage_fraction"] = nonzero / len(accepted) if accepted else 0.0
    metrics["dupo/mean_absolute_advantage"] = total_mass / len(accepted) if accepted else 0.0
    metrics["dupo/saturated_gradient_mass_fraction"] = saturated_mass / total_mass if total_mass else 0.0
    return metrics


def validate_dupo_args(args):
    if not args.dupo:
        return
    required = {
        "advantage_estimator": "grpo",
        "train_backend": "megatron",
        "rewards_normalization": True,
        "grpo_std_normalization": False,
        "normalize_advantages": False,
        "use_dynamic_global_batch_size": True,
        "use_dynamic_batch_size": True,
        "rollout_function_path": "miles.rollout.sglang_rollout.generate_rollout",
        "pipeline_model_parallel_size": 1,
        "calculate_per_token_loss": True,
        "ci_inject_rollout_data_path": None,
        "log_passrate": False,
        "use_fault_tolerance": False,
        "partial_rollout": False,
        "fully_async": False,
        "multi_lora": False,
        "custom_reward_post_process_path": None,
        "custom_convert_samples_to_train_data_path": None,
        "rollout_sample_filter_path": None,
        "rollout_all_samples_process_path": None,
        "load_debug_rollout_data": None,
        "delay_split_train_data_by_dp": False,
    }
    for name, expected in required.items():
        if getattr(args, name) != expected:
            raise ValueError(f"DUPO requires {name}={expected!r}")
    if not args.save:
        raise ValueError("DUPO requires --save for posterior checkpoints")
    DupoConfig(args.dupo_group_size, args.dupo_epsilon, args.dupo_threshold, args.dupo_retention, args.rollout_seed)


def write_step_report(root, rollout_id, args, observations, candidates, accepted, metrics):
    eligible_indices = {sample.index for sample in flatten_rollouts(candidates)}
    groups = {sample.index: group_index for group_index, group in enumerate(accepted) for sample in group}
    rows = [
        dict(
            index=sample.index,
            task_type=sample.metadata["task_type"],
            task_id=sample.metadata["task_id"],
            reward=graded_reward(sample, args),
            eligible=sample.index in eligible_indices,
            accepted=sample.index in groups,
            mixed_group=groups.get(sample.index),
            total_tokens=len(sample.tokens),
            response_tokens=sample.response_length,
            prefix_cache=sample.prefix_cache_info.to_dict(),
            status=sample.status.value,
        )
        for sample in flatten_rollouts(observations)
    ]
    path = Path(root) / "dupo_steps" / f"{rollout_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(rollout_id=rollout_id, metrics=metrics, observations=rows), sort_keys=True) + "\n")
