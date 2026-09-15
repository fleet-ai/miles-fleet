"""CPU integration checks for Bayesian selection and Miles' unchanged GRPO loss path."""

import asyncio
import json
from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils.dupo import train_actor_or_skip
from miles.backends.training_utils.loss_hub.advantages import compute_advantages
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data, split_train_data_by_dp_scheduled_raw
from miles.rollout.dupo import DupoConfig, DupoState, keep_probability, write_step_report
from miles.utils.types import Sample


def make_args(**changes):
    values = dict(
        dupo=True,
        reward_key=None,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        use_dynamic_global_batch_size=True,
        use_dynamic_batch_size=True,
        max_tokens_per_gpu=24,
        balance_data=False,
        balance_by_flops=False,
        multi_lora=False,
        n_samples_per_prompt=4,
        rollout_batch_size=2,
        global_batch_size=8,
        disable_rollout_trim_samples=False,
    )
    return Namespace(**(values | changes))


def make_sample(index, reward, task_type="arithmetic", **changes):
    values = dict(
        index=index,
        group_index=index // 4,
        reward=reward,
        tokens=[1, 2, 3, 4],
        response_length=2,
        metadata={"task_type": task_type, "task_id": f"task-{index // 4}"},
    )
    return Sample(**(values | changes))


def parallel_config(dp_size=1):
    return dict(dp_size=dp_size, cp_size=1, vpp_size=1, microbatch_group_size_per_vp_stage=1)


def training_data(args, groups, dp_size=1):
    samples, metadata = postprocess_rollout_data(args, groups, parallel_config(dp_size))
    return convert_samples_to_train_data(args, samples, metadata, None, None)


def test_prior_is_frozen_and_counts_include_all_graded():
    args = make_args()
    samples = [make_sample(i, reward) for i, reward in enumerate([1, 0, 1, None])]
    samples[1].remove_sample = True
    state = DupoState(DupoConfig())
    groups, metrics = state.process(args, 0, [samples[:2]], [samples], 5)
    assert [sample.index for group in groups for sample in group] == [0]
    assert state.counts == {"arithmetic": [3, 2]}
    assert metrics["dupo/type/arithmetic/mean_before"] == 0.5
    assert metrics["dupo/type/arithmetic/mean_after"] == 0.6
    assert metrics["dupo/type/arithmetic/keep_probability"] == 1.0
    assert metrics["dupo/ungraded"] == 2
    assert metrics["dupo/graded"] == 3
    assert metrics["dupo/launched"] == 5


def test_rejected_successes_and_failures_update_and_resume_exactly(tmp_path):
    args = make_args()
    state = DupoState(DupoConfig(epsilon=0), counts={"arithmetic": [100000, 1]})
    samples = [make_sample(i, float(i % 2)) for i in range(8)]
    groups, _ = state.process(args, 0, [samples], [samples], 8)
    assert groups == []
    assert state.counts == {"arithmetic": [100004, 5]}
    state.save(str(tmp_path), 0)
    restored = DupoState(DupoConfig(epsilon=0))
    restored.load(str(tmp_path), 0)
    assert restored == state
    for instance in (state, restored):
        next_samples = [make_sample(i, float(i % 2)) for i in range(8, 24)]
        output = instance.process(args, 1, [next_samples], [next_samples], 16)
        if instance is state:
            expected = output
        else:
            assert output == expected
    assert restored == state


def test_checkpoint_mismatch_missing_and_duplicate_steps_fail(tmp_path):
    state = DupoState(DupoConfig())
    sample = make_sample(0, 1)
    state.process(make_args(), 0, [[sample]], [[sample]], 1)
    state.save(str(tmp_path), 0)
    with pytest.raises(ValueError, match="configuration"):
        DupoState(DupoConfig(group_size=2)).load(str(tmp_path), 0)
    with pytest.raises(FileNotFoundError):
        DupoState(DupoConfig()).load(str(tmp_path), 1)
    with pytest.raises(ValueError, match="expected rollout 1"):
        state.process(make_args(), 0, [[sample]], [[sample]], 1)


@pytest.mark.parametrize("dp_size", [1, 2, 3])
def test_mixed_partial_groups_survive_conversion_dp_and_token_advantages(dp_size):
    args = make_args()
    samples = [make_sample(i, float(i % 2), task_type="easy" if i % 2 else "hard") for i in range(7)]
    groups, _ = DupoState(DupoConfig()).process(args, 0, [samples], [samples], 7)
    assert [len(group) for group in groups] == [4, 3]
    expected = {
        sample.index: sample.reward - sum(s.reward for s in group) / len(group) for group in groups for sample in group
    }
    data = training_data(args, groups, dp_size)
    assert data["dynamic_global_batch_size"] == 7
    shards = split_train_data_by_dp_scheduled_raw(args, data, train_parallel_config=parallel_config(dp_size))
    assert sorted(index for shard in shards for index in shard["sample_indices"]) == list(range(7))
    for shard in shards:
        assert shard["num_rollouts"] == [7]
        assert len(shard["num_microbatches"]) == 1
        advantages, _ = compute_advantages(
            args, [torch.zeros(2) for _ in shard["rewards"]], shard["rewards"], None, [], [], []
        )
        for index, advantage in zip(shard["sample_indices"], advantages, strict=True):
            torch.testing.assert_close(advantage, torch.full((2,), expected[index]))


def test_standard_grpo_keeps_original_prompt_groups():
    args = make_args(dupo=False, use_dynamic_global_batch_size=False)
    samples = [make_sample(i, float(i >= 4)) for i in range(8)]
    data = training_data(args, [samples[:4], samples[4:]])
    assert data["rewards"] == [0.0] * 8
    assert "dynamic_global_batch_size" not in data
    dupo_args = make_args()
    groups, _ = DupoState(DupoConfig()).process(dupo_args, 0, [samples], [samples], 8)
    assert any(reward != 0 for reward in training_data(dupo_args, groups)["rewards"])


def test_all_rejected_step_does_not_update_optimizer_then_training_resumes(tmp_path):
    args = make_args()
    state = DupoState(DupoConfig(epsilon=0), counts={"arithmetic": [100000, 1]})
    samples = [make_sample(i, float(i % 2)) for i in range(8)]
    groups, _ = state.process(args, 0, [samples], [samples], 8)
    flat, metadata = postprocess_rollout_data(args, groups, parallel_config())
    assert flat == [] and metadata["dynamic_global_batch_size"] == 0

    class CpuActor:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.optimizer = torch.optim.Adam([self.weight], lr=0.1)
            self.steps = 0

        async def train(self, rollout_id, pack):
            self.optimizer.zero_grad()
            self.weight.square().backward()
            self.optimizer.step()
            self.steps += 1

    actor = CpuActor()
    assert not asyncio.run(train_actor_or_skip(actor, 0, dict(skip_training=True, sample_indices=[], data_ref=None)))
    assert actor.steps == 0 and actor.weight.item() == 1 and not actor.optimizer.state
    state.save(str(tmp_path), 0)
    restored = DupoState(DupoConfig(epsilon=0))
    restored.load(str(tmp_path), 0)
    new_samples = [make_sample(8 + i, i % 2, task_type="new-type") for i in range(4)]
    groups, _ = restored.process(args, 1, [new_samples], [new_samples], 4)
    data = training_data(args, groups)
    assert asyncio.run(train_actor_or_skip(actor, 1, dict(sample_indices=data["sample_indices"], data_ref=data)))
    assert actor.steps == 1 and actor.weight.item() < 1
    assert restored.counts["arithmetic"] == [100004, 5]
    assert restored.counts["new-type"] == [3, 3]


def test_selection_does_not_depend_on_current_rewards_and_all_zero_task_has_gradient():
    args = make_args()
    samples = [make_sample(i, float(i % 2), task_type="mixed") for i in range(6000)]
    state = DupoState(DupoConfig(), counts={"mixed": [9, 1]})
    groups, _ = state.process(args, 0, [samples], [samples], len(samples))
    accepted = [s for group in groups for s in group]
    assert 0.46 < sum(s.reward for s in accepted) / len(accepted) < 0.54
    flipped = [make_sample(i, 1 - float(i % 2), task_type="mixed") for i in range(6000)]
    other = DupoState(DupoConfig(), counts={"mixed": [9, 1]})
    other_groups, _ = other.process(args, 0, [flipped], [flipped], len(flipped))
    assert [[s.index for s in group] for group in groups] == [[s.index for s in group] for group in other_groups]
    zero_and_one = [make_sample(i, float(i >= 4), task_type="zero" if i < 4 else "one") for i in range(8)]
    groups, _ = DupoState(DupoConfig()).process(args, 0, [zero_and_one], [zero_and_one], 8)
    data = training_data(args, groups)
    assert any(reward < 0 for index, reward in zip(data["sample_indices"], data["rewards"], strict=True) if index < 4)


@pytest.mark.parametrize("metadata", [{}, {"task_type": "x"}, {"task_type": "", "task_id": "x"}])
def test_explicit_metadata_required(metadata):
    sample = make_sample(0, 1, metadata=metadata)
    with pytest.raises(ValueError, match="explicit nonempty metadata"):
        DupoState(DupoConfig()).process(make_args(), 0, [[sample]], [[sample]], 1)


def test_probability_clipping_and_unsupported_partition():
    assert keep_probability(0.5, DupoConfig()) == 1
    assert keep_probability(0, DupoConfig(retention="easy")) == 1
    assert keep_probability(1, DupoConfig()) == 0.05
    with pytest.raises(ValueError, match="cannot fill"):
        postprocess_rollout_data(make_args(), [[make_sample(0, 1)]], parallel_config(2))
    with pytest.raises(ValueError, match="virtual pipeline"):
        postprocess_rollout_data(make_args(), [[make_sample(0, 1)]], parallel_config() | {"vpp_size": 2})


@pytest.mark.parametrize("rewards", [[0.2, 0.7, -0.3, 1.4, 0.1], [0, 0, 0, 0, 0], [0.6]])
def test_partial_singleton_and_nonbinary_rewards_reach_training_tensors(rewards):
    args = make_args()
    samples = [make_sample(i, value) for i, value in enumerate(rewards)]
    state = DupoState(DupoConfig())
    groups, _ = state.process(args, 0, [samples], [samples], len(samples))
    data = training_data(args, groups)
    expected = {
        sample.index: sample.reward - sum(s.reward for s in group) / len(group) for group in groups for sample in group
    }
    assert len(data["rewards"]) == len(rewards)
    assert data["rewards"] == pytest.approx([expected[index] for index in data["sample_indices"]])
    assert state.counts["arithmetic"] == [1 + sum(r >= 1 for r in rewards), 1 + sum(r < 1 for r in rewards)]
    singleton = groups[-1][0]
    assert expected[singleton.index] == 0


def agentic_sample(index, reward, status=Sample.Status.COMPLETED, **changes):
    """One Sample per rollout as the agentic generator returns it: rollout_id
    set, the harness's own metadata beside task_type/task_id."""
    values = dict(
        rollout_id=index,
        status=status,
        metadata={
            "task_type": "comp_n4",
            "task_id": f"dataminer_v2_comp_n4_src{index}",
            "task_key": f"dataminer_v2_comp_n4_src{index}",
            "data_version": "dev-20260907",
            "reward": reward,
        },
    )
    return make_sample(index, reward, **(values | changes))


def test_agentic_single_sample_with_rollout_id_passes_and_compact_siblings_fail():
    args = make_args()
    samples = [agentic_sample(i, float(i % 2)) for i in range(4)]
    groups, _ = DupoState(DupoConfig()).process(args, 0, [samples], [samples], 4)
    assert sorted(sample.index for group in groups for sample in group) == [0, 1, 2, 3]
    siblings = [agentic_sample(i, 1.0, rollout_id=0) for i in range(2)]
    with pytest.raises(ValueError, match="compact multi-segment"):
        DupoState(DupoConfig()).process(args, 0, [siblings], [siblings], 2)
    nested = [[agentic_sample(0, 1.0)]]
    with pytest.raises(ValueError, match="one Sample per rollout"):
        DupoState(DupoConfig()).process(args, 0, [nested], [nested], 1)


def test_aborted_sample_is_ungraded_whatever_its_reward_field_holds(tmp_path):
    args = make_args()
    graded = [agentic_sample(i, float(i % 2)) for i in range(4)]
    aborted = [
        agentic_sample(4, 0.0, status=Sample.Status.ABORTED, tokens=[], response_length=0),
        agentic_sample(5, None, status=Sample.Status.ABORTED, tokens=[], response_length=0),
    ]
    samples = graded + aborted
    state = DupoState(DupoConfig())
    groups, metrics = state.process(args, 0, [samples], [samples], 6)
    assert state.counts == {"comp_n4": [3, 3]}
    assert metrics["dupo/graded"] == 4 and metrics["dupo/ungraded"] == 2
    assert metrics["dupo/graded_reward_mean"] == 0.5 and metrics["dupo/graded_pass_rate"] == 0.5
    assert sorted(sample.index for group in groups for sample in group) == [0, 1, 2, 3]
    write_step_report(str(tmp_path), 0, args, [samples], [samples], groups, metrics)
    report = json.loads((tmp_path / "dupo_steps" / "0.json").read_text())
    rows = {row["index"]: row for row in report["observations"]}
    assert rows[4]["reward"] is None and rows[4]["status"] == "aborted" and rows[4]["total_tokens"] == 0
    assert rows[4]["accepted"] is False and rows[5]["accepted"] is False
    assert rows[0]["status"] == "completed" and rows[0]["response_tokens"] == 2 and rows[0]["prefix_cache"] is not None
