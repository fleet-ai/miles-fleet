import asyncio
from types import SimpleNamespace

from miles.rollout.rm_hub import math_answer


def _sample(response, label, status="completed", metadata=None):
    return SimpleNamespace(response=response, label=label, status=status, metadata=metadata if metadata is not None else {})


def test_batched_reward_reads_committed_answers():
    samples = [
        _sample("think</think>\n\n**Answer:** Kerry is **8** years old.", "8"),
        _sample("think</think>\n\nThe walk takes 204 minutes.\n\n204", "204"),
        _sample("think</think>\n\n\\boxed{7}", "8"),
        _sample("<think>still thinking about 8", "8", status="truncated"),
    ]
    rewards = asyncio.run(math_answer.miles_batched_reward(None, samples))
    assert rewards == [1.0, 1.0, 0.0, 0.0]
    assert samples[0].metadata["scorer"]["tier"] == "stated"
    assert samples[3].metadata["scorer"]["reason"] == "no_answer_section"


def test_truncated_sample_only_scores_committed_answers():
    truncated = _sample("think</think>\n\nSo bc is not 10", "10", status="truncated")
    boxed = _sample("think</think>\n\n\\boxed{10} and more", "10", status="truncated")
    assert asyncio.run(math_answer.miles_batched_reward(None, [truncated, boxed])) == [0.0, 1.0]
