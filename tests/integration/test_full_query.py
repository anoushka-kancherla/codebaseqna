import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import api.query as query_mod
import cli
import qa_types.session as session_mod
import xai.faithfulness as faithfulness_mod

SAMPLE_REPO = Path(__file__).resolve().parent.parent / "fixtures" / "sample_repo"

ANSWER_HEADER = {
    "confidence": "high",
    "uncertainty_type": "epistemic",
    "call_graph_complete": True,
    "files_read": ["auth.py"],
    "files_read_reasons": ["contains the login entry point"],
    "caveats": "",
}
ANSWER_PROSE = "Login happens in `auth.py:1-2`."


def _events(json_header: dict, prose: str):
    text = json.dumps(json_header) + "\n" + prose
    return [
        SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="text")),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(text=text)),
        SimpleNamespace(type="message_delta", usage=SimpleNamespace(output_tokens=10, input_tokens=5)),
    ]


class _StreamCtx:
    def __init__(self, events):
        self._events = events

    def __enter__(self):
        return iter(self._events)

    def __exit__(self, *a):
        return False


class FakeAnthropic:
    """Stands in for anthropic.Anthropic — the only thing these integration
    tests mock. Everything else (local MCP server, nav log, attribution,
    confidence, session log, faithfulness write-back) runs for real.

    _stream reads repo://tree and each cited file through the actual running
    MCP server (mirroring what Claude's server-side MCP connector would do),
    so the navigation log reflects real reads rather than staying empty.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kwargs):
        from server import mcp_server

        json_header, prose = self._responses.pop(0)
        mcp_server.get_tree_json()
        for path in json_header.get("files_read", []):
            mcp_server.get_file_json(path)
        return _StreamCtx(_events(json_header, prose))


@pytest.fixture
def logs_dir(tmp_path, monkeypatch):
    # qa_types.session and xai.faithfulness each imported LOGS_DIR by name,
    # so each holds its own binding and both need patching independently.
    monkeypatch.setattr(session_mod, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(faithfulness_mod, "LOGS_DIR", tmp_path)
    return tmp_path


def _run_cli(monkeypatch, responses, extra_args, input=None):
    monkeypatch.setattr(query_mod, "Anthropic", lambda: FakeAnthropic(responses))
    runner = CliRunner()
    args = ["--repo", str(SAMPLE_REPO), "--question", "where does login happen?"] + extra_args
    return runner.invoke(cli.main, args, input=input)


def test_session_log_written(logs_dir, monkeypatch):
    result = _run_cli(monkeypatch, [(ANSWER_HEADER, ANSWER_PROSE)], ["--port", "8100"])
    assert result.exit_code == 0, result.output

    session_files = list(logs_dir.glob("*.json"))
    assert len(session_files) == 1
    data = json.loads(session_files[0].read_text())
    for key in [
        "session_id", "created_at", "repo_path", "question", "model", "flags",
        "navigation", "attribution", "confidence", "answer", "faithfulness", "api_usage",
    ]:
        assert key in data


def test_navigation_log_nonempty(logs_dir, monkeypatch):
    result = _run_cli(monkeypatch, [(ANSWER_HEADER, ANSWER_PROSE)], ["--port", "8101"])
    assert result.exit_code == 0, result.output
    data = json.loads(next(logs_dir.glob("*.json")).read_text())
    assert len(data["navigation"]["reads"]) > 0
    assert data["navigation"]["reads"][0]["event"] == "tree_read"


def test_attribution_all_valid(logs_dir, monkeypatch):
    result = _run_cli(monkeypatch, [(ANSWER_HEADER, ANSWER_PROSE)], ["--port", "8102"])
    assert result.exit_code == 0, result.output
    data = json.loads(next(logs_dir.glob("*.json")).read_text())
    assert data["attribution"]["invalid_count"] == 0
    assert data["attribution"]["valid_count"] == 1


def test_verify_appends_to_session(logs_dir, monkeypatch):
    result = _run_cli(monkeypatch, [(ANSWER_HEADER, ANSWER_PROSE)], ["--port", "8103", "--output", "json"])
    assert result.exit_code == 0, result.output
    session_id = json.loads(result.output)["session_id"]

    verdicts = [{"claim_index": 0, "claim_summary": "login location", "verdict": "VERIFIED", "evidence": "", "explanation": ""}]
    fake_verify_client = SimpleNamespace(
        messages=SimpleNamespace(create=lambda **k: SimpleNamespace(content=[SimpleNamespace(text=json.dumps(verdicts))]))
    )
    monkeypatch.setattr(faithfulness_mod, "Anthropic", lambda: fake_verify_client)

    runner = CliRunner()
    verify_result = runner.invoke(cli.main, ["--verify", session_id])
    assert verify_result.exit_code == 0, verify_result.output

    data = json.loads((logs_dir / f"{session_id}.json").read_text())
    assert "faithfulness" in data
    assert data["faithfulness"]["faithfulness_score"] == 1.0


def test_path_traversal_rejected(monkeypatch):
    from server import mcp_server
    from server.navigation_log import NavigationLog

    monkeypatch.setattr(mcp_server, "ROOT", SAMPLE_REPO)
    monkeypatch.setattr(mcp_server, "NAV_LOG", NavigationLog())

    result = json.loads(mcp_server.get_file_json("../../../../etc/passwd"))
    assert result["error"] == "PATH_TRAVERSAL"
    assert mcp_server.NAV_LOG.entries == []


def test_stability_jaccard_range(logs_dir, monkeypatch):
    responses = [(ANSWER_HEADER, ANSWER_PROSE) for _ in range(3)]
    result = _run_cli(monkeypatch, responses, ["--stability", "--runs", "3", "--yes", "--port", "8104"])
    assert result.exit_code == 0, result.output
    assert "Mean Jaccard" in result.output
    assert "Rating: HIGH" in result.output
