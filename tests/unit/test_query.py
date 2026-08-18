from types import SimpleNamespace
from unittest.mock import MagicMock

from api.query import run_query


class _FakeStreamCtx:
    def __init__(self, events):
        self._events = events

    def __enter__(self):
        return iter(self._events)

    def __exit__(self, *a):
        return False


def _fake_client(captured: dict):
    client = MagicMock()

    def fake_stream(**kwargs):
        captured.update(kwargs)
        return _FakeStreamCtx([])

    client.beta.messages.stream.side_effect = fake_stream
    return client


def test_run_query_prepends_context_prefix():
    captured = {}
    run_query(
        repo_path="/repo", question="where is auth?", client=_fake_client(captured),
        mcp_server_url="http://x/mcp/", context_prefix="# relevant chunk\ndef login(): ...",
    )
    content = captured["messages"][0]["content"]
    assert content.startswith("# relevant chunk\ndef login(): ...")
    assert content.endswith("where is auth?")


def test_run_query_without_context_prefix_sends_question_only():
    captured = {}
    run_query(repo_path="/repo", question="where is auth?", client=_fake_client(captured), mcp_server_url="http://x/mcp/")
    assert captured["messages"][0]["content"] == "where is auth?"


def test_run_query_uses_given_mcp_server_url():
    captured = {}
    run_query(repo_path="/repo", question="q", client=_fake_client(captured), mcp_server_url="http://custom/mcp/")
    assert captured["mcp_servers"] == [{"type": "url", "url": "http://custom/mcp/", "name": "codebase"}]


def test_run_query_thinking_disabled_omits_thinking_param():
    captured = {}
    run_query(repo_path="/repo", question="q", client=_fake_client(captured), mcp_server_url="http://x/mcp/", thinking_enabled=False)
    assert "thinking" not in captured


def test_run_query_prepends_history_before_new_user_turn():
    captured = {}
    history = [
        {"role": "user", "content": "where is auth?"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "checking auth.py", "signature": "sig-1"},
            {"type": "text", "text": '{"confidence": "high"} It is in auth.py.'},
        ]},
    ]
    run_query(
        repo_path="/repo", question="what about the token refresh path?", client=_fake_client(captured),
        mcp_server_url="http://x/mcp/", history=history,
    )
    assert captured["messages"] == [
        *history,
        {"role": "user", "content": "what about the token refresh path?"},
    ]


def test_run_query_without_history_sends_single_user_turn():
    captured = {}
    run_query(repo_path="/repo", question="q", client=_fake_client(captured), mcp_server_url="http://x/mcp/")
    assert captured["messages"] == [{"role": "user", "content": "q"}]
