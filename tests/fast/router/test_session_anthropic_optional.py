"""The session server must come up on an sglang without the anthropic entrypoint."""

import importlib
import sys

from fastapi import FastAPI

import miles.rollout.session.sessions as sessions


async def _handler(request, session_id):  # pragma: no cover - never called
    return None


def _paths(app):
    return [route.path for route in app.routes]


def test_anthropic_route_is_skipped_when_the_entrypoint_is_missing(monkeypatch):
    monkeypatch.setattr(sessions, "ANTHROPIC_MESSAGES_IMPORT_ERROR", ImportError("no sglang anthropic utils"))
    app = FastAPI()
    assert sessions.register_anthropic_messages_route(app, _handler) is False
    assert "/sessions/{session_id}/v1/messages" not in _paths(app)
    monkeypatch.setattr(sessions, "ANTHROPIC_MESSAGES_IMPORT_ERROR", None)
    assert sessions.register_anthropic_messages_route(app, _handler) is True
    assert "/sessions/{session_id}/v1/messages" in _paths(app)


def test_sessions_imports_without_sglang_anthropic_utils(monkeypatch):
    # A None entry in sys.modules makes the import raise ImportError, as an old sglang does.
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints.anthropic.utils", None)
    monkeypatch.delitem(sys.modules, "miles.rollout.session.anthropic_adapter", raising=False)
    try:
        reloaded = importlib.reload(sessions)
        assert isinstance(reloaded.ANTHROPIC_MESSAGES_IMPORT_ERROR, ImportError)
        assert callable(reloaded.setup_session_routes)
    finally:
        monkeypatch.undo()
        importlib.reload(sessions)
