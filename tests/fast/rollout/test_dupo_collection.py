"""Exercise the real SGLang collection and abort paths without GPU generation."""

import asyncio
from argparse import Namespace
from unittest.mock import AsyncMock

from miles.rollout import sglang_rollout
from miles.utils.types import Sample


def test_abort_keeps_late_graded_observations_for_dupo(monkeypatch):
    async def run():
        sample = Sample(index=7, reward=1.0)

        async def completed_group():
            return [sample]

        state = Namespace(aborted=False, pendings={asyncio.create_task(completed_group())})
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)
        monkeypatch.setattr(sglang_rollout, "get", AsyncMock(return_value={"urls": []}))
        monkeypatch.setattr(sglang_rollout, "call_agent_abort_hook", AsyncMock())
        args = Namespace(partial_rollout=False, use_miles_router=True, sglang_router_ip="unused", sglang_router_port=0)
        observations = []
        assert await sglang_rollout.abort(args, 0, observations=observations) == []
        assert observations == [[sample]]
        assert state.aborted

    asyncio.run(run())
