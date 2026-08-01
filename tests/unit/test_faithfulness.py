import json
from types import SimpleNamespace

import xai.faithfulness as faithfulness_mod
from xai.faithfulness import run_faithfulness_check


def _write_session(tmp_path, repo_path, prose, citations):
    session = {
        "session_id": "abc123",
        "repo_path": str(repo_path),
        "answer": {"prose": prose},
        "attribution": {"citations": citations, "valid_count": len(citations), "invalid_count": 0},
    }
    (tmp_path / "abc123.json").write_text(json.dumps(session))
    return session


class FakeClient:
    def __init__(self, reply_json, leading_thinking_block=False, captured=None, fenced=False):
        self.reply_json = reply_json
        self.leading_thinking_block = leading_thinking_block
        self.captured = captured
        self.fenced = fenced
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        if self.captured is not None:
            self.captured.update(kwargs)
        content = []
        if self.leading_thinking_block:
            content.append(SimpleNamespace(type="thinking", thinking="pondering..."))
        text = json.dumps(self.reply_json)
        if self.fenced:
            text = f"```json\n{text}\n```"
        content.append(SimpleNamespace(type="text", text=text))
        return SimpleNamespace(content=content)


def test_faithfulness_score_computed_from_verdicts(tmp_path, monkeypatch):
    monkeypatch.setattr(faithfulness_mod, "LOGS_DIR", tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def f():\n    return 1\n")

    _write_session(
        tmp_path, repo, "`a.py:1-2` defines f.",
        [{"file": "a.py", "start": 1, "end": 2, "status": "VALID"}],
    )

    reply = [{"claim_index": 0, "claim_summary": "defines f", "verdict": "VERIFIED", "evidence": "", "explanation": ""}]
    result = run_faithfulness_check("abc123", client=FakeClient(reply))

    assert result["faithfulness_score"] == 1.0


def test_faithfulness_skips_leading_thinking_block(tmp_path, monkeypatch):
    # A real run showed claude-sonnet-5 emitting a leading ThinkingBlock even
    # without thinking explicitly requested, so content[0] isn't reliably text.
    monkeypatch.setattr(faithfulness_mod, "LOGS_DIR", tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def f():\n    return 1\n")

    _write_session(
        tmp_path, repo, "`a.py:1-2` defines f.",
        [{"file": "a.py", "start": 1, "end": 2, "status": "VALID"}],
    )

    reply = [{"claim_index": 0, "claim_summary": "defines f", "verdict": "VERIFIED", "evidence": "", "explanation": ""}]
    result = run_faithfulness_check("abc123", client=FakeClient(reply, leading_thinking_block=True))

    assert result["faithfulness_score"] == 1.0
    saved = json.loads((tmp_path / "abc123.json").read_text())
    assert saved["faithfulness"]["faithfulness_score"] == 1.0


def test_faithfulness_disables_thinking(tmp_path, monkeypatch):
    # A real run burned the whole max_tokens budget on unrequested adaptive
    # thinking and never produced text (stop_reason: max_tokens). Verify
    # thinking is explicitly turned off for this deterministic-JSON call.
    monkeypatch.setattr(faithfulness_mod, "LOGS_DIR", tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def f():\n    return 1\n")
    _write_session(tmp_path, repo, "`a.py:1-2` defines f.", [{"file": "a.py", "start": 1, "end": 2, "status": "VALID"}])

    reply = [{"claim_index": 0, "claim_summary": "defines f", "verdict": "VERIFIED", "evidence": "", "explanation": ""}]
    captured = {}
    run_faithfulness_check("abc123", client=FakeClient(reply, captured=captured))
    assert captured["thinking"] == {"type": "disabled"}


def test_faithfulness_strips_json_fence_despite_prompt_saying_not_to(tmp_path, monkeypatch):
    # A real run wrapped the reply in ```json fences despite the prompt
    # explicitly saying "no code fences" — parsing must tolerate it anyway.
    monkeypatch.setattr(faithfulness_mod, "LOGS_DIR", tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def f():\n    return 1\n")
    _write_session(tmp_path, repo, "`a.py:1-2` defines f.", [{"file": "a.py", "start": 1, "end": 2, "status": "VALID"}])

    reply = [{"claim_index": 0, "claim_summary": "defines f", "verdict": "VERIFIED", "evidence": "", "explanation": ""}]
    result = run_faithfulness_check("abc123", client=FakeClient(reply, fenced=True))
    assert result["faithfulness_score"] == 1.0


def test_faithfulness_partial_and_unsupported_average(tmp_path, monkeypatch):
    monkeypatch.setattr(faithfulness_mod, "LOGS_DIR", tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("x = 1\ny = 2\n")

    _write_session(
        tmp_path, repo, "`a.py:1-2` does two things.",
        [{"file": "a.py", "start": 1, "end": 2, "status": "VALID"}],
    )

    reply = [
        {"claim_index": 0, "claim_summary": "c1", "verdict": "PARTIAL", "evidence": "", "explanation": ""},
    ]
    result = run_faithfulness_check("abc123", client=FakeClient(reply))
    assert result["faithfulness_score"] == 0.5


def test_faithfulness_zero_when_no_valid_citations(tmp_path, monkeypatch):
    monkeypatch.setattr(faithfulness_mod, "LOGS_DIR", tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()

    _write_session(tmp_path, repo, "No citations here.", [])
    result = run_faithfulness_check("abc123", client=FakeClient([]))
    assert result["faithfulness_score"] == 0.0
