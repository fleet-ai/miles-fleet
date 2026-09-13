import json
from argparse import Namespace

import pytest

from miles.rollout.compute_accounting import ComputeLedger, Qwen35Flops
from miles.utils.types import Sample


def config():
    return dict(
        model_type="qwen3_5_text",
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        vocab_size=32,
    )


def sample(index=0, cached=0):
    return Sample(
        index=index,
        tokens=[1] * 7,
        response_length=3,
        status=Sample.Status.COMPLETED,
        prefix_cache_info=Sample.PrefixCacheInfo(cached_tokens=cached, total_prompt_tokens=4),
    )


def ledger(tmp_path, budget=1e9):
    (tmp_path / "config.json").write_text(json.dumps(dict(text_config=config())))
    return ComputeLedger(
        Namespace(
            hf_checkpoint=str(tmp_path),
            save=str(tmp_path),
            compute_budget_flops=budget,
            compute_checkpoint_count=4,
            compute_overshoot_tolerance=0.05,
        )
    )


def test_generation_counts_prefix_cache_and_only_generated_logit_positions():
    model = Qwen35Flops(config())
    assert model.generation(sample()) == model.forward(6, logits=3)
    assert model.generation(sample(cached=2)) == model.forward(6, cached=2, logits=3)
    assert model.generation(sample()) > model.generation(sample(cached=2))
    assert model.forward(6) - model.generation(sample()) == 3 * model.head_token
    with pytest.raises(ValueError, match="shorter"):
        model.generation(sample(cached=4))


def test_completion_resume_and_empty_step_charge_only_generation(tmp_path):
    state = ledger(tmp_path)
    samples = [sample(i) for i in range(2)]
    state.prepare(0, [samples], 2, [], 100)
    assert state.total == 0
    result = state.complete(0)
    assert result["stages"]["generation"] > 0
    assert sum(result["stages"].values()) == result["stages"]["generation"]
    state.save(0)
    restored = ledger(tmp_path)
    restored.load(str(tmp_path), 0)
    assert restored.schedule() == state.schedule()
    restored.prepare(1, [[sample(2)]], 1, [sample(2)], 100)
    result = restored.complete(1)
    f = state.estimator.forward(7)
    assert result["stages"] == dict(
        generation=state.estimator.generation(sample(2)),
        reference_scoring=f,
        policy_scoring=f,
        training_forward=f,
        training_backward=2 * f,
        recomputation=0,
        optimizer=1500,
    )
    assert result["total"] > result["before"]
    with pytest.raises(ValueError, match="matching"):
        restored.complete(1)


def test_budget_crossing_is_recorded_and_missing_generation_fails(tmp_path):
    state = ledger(tmp_path, budget=1)
    with pytest.raises(ValueError, match="Every launched"):
        state.prepare(0, [[sample()]], 2, [], 100)
    state.prepare(0, [[sample()]], 1, [], 100)
    result = state.complete(0)
    assert result["budget_exhausted"] and result["checkpoint_due"]
    assert result["overshoot_fraction"] == result["total"] - 1
    assert state.schedule()["consumed_fraction"] == 1
