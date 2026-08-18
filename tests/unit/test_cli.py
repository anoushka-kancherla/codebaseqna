import importlib
import json

from click.testing import CliRunner

import cli
import xai.faithfulness as faithfulness_mod
from api.stream_parser import ParsedResponse
from server import mcp_server


def test_cli_end_to_end_with_mocked_claude(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("hello\nworld\n")
    monkeypatch.setattr("qa_types.session.LOGS_DIR", tmp_path)

    def fake_run_query(repo_path, question, model, max_files, mcp_server_url, context_prefix=None):
        # Simulate what Claude would do via the MCP connector: read the tree,
        # then read a file, then answer citing it.
        mcp_server.get_tree_json()
        mcp_server.get_file_json("README.md")
        return ParsedResponse(
            thinking="I should check the README first.",
            json_header={"confidence": "high"},
            prose="The greeting lives in `README.md:1-2`.",
            tool_results=[],
            usage={"output_tokens": 12},
        )

    monkeypatch.setattr(cli, "run_query", fake_run_query)

    runner = CliRunner()
    result = runner.invoke(cli.main, ["--repo", str(tmp_path), "--question", "where is the greeting?", "--port", "8001"])

    assert result.exit_code == 0, result.output
    assert "README.md:1-2" in result.output
    assert "Session saved to" in result.output
    assert mcp_server.NAV_LOG.entries[0].event == "tree_read"
    assert any(e.event == "file_read" for e in mcp_server.NAV_LOG.entries)

    session_files = list(tmp_path.glob("*.json"))
    assert len(session_files) == 1
    data = json.loads(session_files[0].read_text())
    assert data["navigation"]["files_read"] == 1
    assert data["answer"]["prose"] == "The greeting lives in `README.md:1-2`."
    assert data["confidence"]["level"] == "low"  # override: files_read missing from json_header


def test_cli_not_found_shows_search_report_instead_of_prose(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("hello\n")
    monkeypatch.setattr("qa_types.session.LOGS_DIR", tmp_path)

    def fake_run_query(repo_path, question, model, max_files, mcp_server_url, context_prefix=None):
        return ParsedResponse(
            thinking="",
            json_header={
                "result": "not_found",
                "files_checked": ["README.md"],
                "closest_match": None,
                "suggested_next_steps": ["grep for the term"],
            },
            prose="",
            tool_results=[],
            usage={},
        )

    monkeypatch.setattr(cli, "run_query", fake_run_query)
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--repo", str(tmp_path), "--question", "where is X?", "--port", "8002"])

    assert result.exit_code == 0, result.output
    assert "Search report" in result.output
    assert "grep for the term" in result.output

    data = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert data["answer"]["not_found"] == "not_found"


def test_cli_flags_invalid_citation(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("hello\n")
    monkeypatch.setattr("qa_types.session.LOGS_DIR", tmp_path)

    def fake_run_query(repo_path, question, model, max_files, mcp_server_url, context_prefix=None):
        return ParsedResponse(
            thinking="",
            json_header={"confidence": "low", "files_read": ["README.md"]},
            prose="See `nonexistent.py:1-2` for the answer.",
            tool_results=[],
            usage={},
        )

    monkeypatch.setattr(cli, "run_query", fake_run_query)
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--repo", str(tmp_path), "--question", "where is X?", "--port", "8003"])

    assert result.exit_code == 0, result.output
    assert "INVALID_PATH" in result.output

    data = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert data["attribution"]["invalid_count"] == 1


def test_cli_verify_warns_below_threshold(monkeypatch):
    monkeypatch.setattr(faithfulness_mod, "run_faithfulness_check", lambda session_id, model: {"faithfulness_score": 0.4, "claims": []})
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--verify", "abc123"])
    assert result.exit_code == 0, result.output
    assert "WARNING" in result.output


def test_cli_verify_no_warning_above_threshold(monkeypatch):
    monkeypatch.setattr(faithfulness_mod, "run_faithfulness_check", lambda session_id, model: {"faithfulness_score": 0.9, "claims": []})
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--verify", "abc123"])
    assert result.exit_code == 0, result.output
    assert "WARNING" not in result.output


def test_cli_stability_dispatch(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("hello\n")
    fake_result = {
        "runs": 2, "per_run_files": [["a.py"], ["a.py"]],
        "pairwise_jaccard": [{"run_a": 0, "run_b": 1, "jaccard": 1.0}],
        "mean_jaccard": 1.0, "rating": "high",
        "consistent_files": ["a.py"], "inconsistent_files": [],
    }
    monkeypatch.setattr("xai.stability.run_stability_check", lambda *a, **k: fake_result)

    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        ["--repo", str(tmp_path), "--question", "q?", "--stability", "--runs", "2", "--yes", "--port", "8004"],
    )
    assert result.exit_code == 0, result.output
    assert "Stability check (2 runs)" in result.output
    assert "HIGH" in result.output


def test_cli_stability_aborts_without_confirmation(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("hello\n")
    monkeypatch.setattr("xai.stability.run_stability_check", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")))

    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        ["--repo", str(tmp_path), "--question", "q?", "--stability", "--port", "8005"],
        input="n\n",
    )
    assert result.exit_code == 0, result.output
    assert "Aborted" in result.output


def test_cli_interactive_two_questions_independent_nav_logs(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("hello\nworld\n")
    monkeypatch.setattr("qa_types.session.LOGS_DIR", tmp_path)

    def fake_run_query(repo_path, question, model, max_files, mcp_server_url, context_prefix=None):
        mcp_server.get_tree_json()
        mcp_server.get_file_json("README.md")
        return ParsedResponse(
            thinking="", json_header={"confidence": "low"}, prose=f"answer to {question}",
            tool_results=[], usage={},
        )

    monkeypatch.setattr(cli, "run_query", fake_run_query)
    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        ["--repo", str(tmp_path), "--interactive", "--port", "8007"],
        input="question one\nquestion two\nexit\n",
    )

    assert result.exit_code == 0, result.output
    session_files = sorted(tmp_path.glob("*.json"))
    assert len(session_files) == 2
    for f in session_files:
        data = json.loads(f.read_text())
        # If NAV_LOG weren't reset between questions, the second session would show
        # 2 accumulated file reads instead of 1.
        assert data["navigation"]["files_read"] == 1


def test_cli_interactive_rejects_combination_with_stability():
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--interactive", "--stability"])
    assert result.exit_code != 0
    assert "cannot be combined" in result.output


def test_cli_interactive_rejects_combination_with_verify():
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--interactive", "--verify", "abc123"])
    assert result.exit_code != 0
    assert "cannot be combined" in result.output


def test_cli_index_flag_prepends_context(tmp_path, monkeypatch):
    (tmp_path / "auth.py").write_text("def login():\n    pass\n")
    monkeypatch.setattr("qa_types.session.LOGS_DIR", tmp_path)

    captured = {}

    def fake_run_query(repo_path, question, model, max_files, mcp_server_url, context_prefix=None):
        captured["context_prefix"] = context_prefix
        return ParsedResponse(thinking="", json_header={"confidence": "low"}, prose="answer", tool_results=[], usage={})

    monkeypatch.setattr(cli, "run_query", fake_run_query)
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--repo", str(tmp_path), "--question", "where is login?", "--index", "--port", "8006"])

    assert result.exit_code == 0, result.output
    assert captured["context_prefix"] is not None
    assert "login" in captured["context_prefix"]


def test_cli_verbose_shows_full_thinking(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("hello\n")
    monkeypatch.setattr("qa_types.session.LOGS_DIR", tmp_path)
    long_thinking = "x" * 500

    def fake_run_query(repo_path, question, model, max_files, mcp_server_url, context_prefix=None):
        return ParsedResponse(thinking=long_thinking, json_header={"confidence": "low"}, prose="answer", tool_results=[], usage={})

    monkeypatch.setattr(cli, "run_query", fake_run_query)
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--repo", str(tmp_path), "--question", "q?", "--verbose", "--port", "8010"])

    assert result.exit_code == 0, result.output
    assert long_thinking in result.output
    assert "[--verbose to expand]" not in result.output


def test_cli_without_verbose_truncates_thinking(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("hello\n")
    monkeypatch.setattr("qa_types.session.LOGS_DIR", tmp_path)
    long_thinking = "x" * 500

    def fake_run_query(repo_path, question, model, max_files, mcp_server_url, context_prefix=None):
        return ParsedResponse(thinking=long_thinking, json_header={"confidence": "low"}, prose="answer", tool_results=[], usage={})

    monkeypatch.setattr(cli, "run_query", fake_run_query)
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--repo", str(tmp_path), "--question", "q?", "--port", "8011"])

    assert result.exit_code == 0, result.output
    assert long_thinking not in result.output
    assert "[--verbose to expand]" in result.output


def test_cli_prints_estimated_cost(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("hello\n")
    monkeypatch.setattr("qa_types.session.LOGS_DIR", tmp_path)

    def fake_run_query(repo_path, question, model, max_files, mcp_server_url, context_prefix=None):
        return ParsedResponse(
            thinking="", json_header={"confidence": "low"}, prose="answer", tool_results=[],
            usage={"input_tokens": 1_000_000, "output_tokens": 1_000_000},
        )

    monkeypatch.setattr(cli, "run_query", fake_run_query)
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--repo", str(tmp_path), "--question", "q?", "--port", "8012"])

    assert result.exit_code == 0, result.output
    # default sonnet pricing: $3/M input + $15/M output = $18.00 for 1M+1M tokens
    assert "Estimated cost: $18.0000" in result.output


def test_cli_list_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr("qa_types.session.LOGS_DIR", tmp_path)
    (tmp_path / "abc.json").write_text(json.dumps({
        "session_id": "abc", "created_at": "2026-01-01T00:00:00Z", "question": "where is auth?",
    }))

    runner = CliRunner()
    result = runner.invoke(cli.main, ["--list-sessions"])

    assert result.exit_code == 0, result.output
    assert "abc" in result.output
    assert "where is auth?" in result.output


def test_cli_list_sessions_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("qa_types.session.LOGS_DIR", tmp_path / "nonexistent")

    runner = CliRunner()
    result = runner.invoke(cli.main, ["--list-sessions"])

    assert result.exit_code == 0, result.output
    assert "No sessions found." in result.output


def test_cli_show_session(tmp_path, monkeypatch):
    monkeypatch.setattr("qa_types.session.LOGS_DIR", tmp_path)
    session_data = {
        "session_id": "abc123",
        "repo_path": "/some/repo",
        "question": "where is auth?",
        "created_at": "2026-01-01T00:00:00Z",
        "answer": {"prose": "Auth lives in `auth.py:1-2`.", "not_found": None},
        "confidence": {"level": "high", "uncertainty_type": "none"},
        "navigation": {
            "files_read": 1, "total_files_in_repo": 5, "coverage_pct": 20.0,
            "reads": [{"event": "file_read", "path": "auth.py"}],
        },
    }
    (tmp_path / "abc123.json").write_text(json.dumps(session_data))

    runner = CliRunner()
    result = runner.invoke(cli.main, ["--show-session", "abc123"])

    assert result.exit_code == 0, result.output
    assert "auth.py:1-2" in result.output
    assert "HIGH" in result.output
    assert "auth.py" in result.output


def test_cli_show_session_missing_id_errors(tmp_path, monkeypatch):
    monkeypatch.setattr("qa_types.session.LOGS_DIR", tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--show-session", "nonexistent"])
    assert result.exit_code != 0
    assert "No session found" in result.output


def test_cli_env_defaults_applied(monkeypatch):
    monkeypatch.setenv("CODEBASEQNA_MODEL", "claude-haiku-4-5")
    monkeypatch.setenv("CODEBASEQNA_MAX_FILES", "5")
    monkeypatch.setenv("CODEBASEQNA_REPO", "/tmp/some-repo")
    importlib.reload(cli)
    try:
        assert cli.DEFAULT_MODEL_ENV == "claude-haiku-4-5"
        assert cli.DEFAULT_MAX_FILES_ENV == 5
        assert cli.DEFAULT_REPO_ENV == "/tmp/some-repo"
    finally:
        monkeypatch.delenv("CODEBASEQNA_MODEL", raising=False)
        monkeypatch.delenv("CODEBASEQNA_MAX_FILES", raising=False)
        monkeypatch.delenv("CODEBASEQNA_REPO", raising=False)
        importlib.reload(cli)


def test_cli_auto_activates_index_for_large_repo(tmp_path, monkeypatch):
    (tmp_path / "auth.py").write_text("def login():\n    pass\n")
    monkeypatch.setattr("qa_types.session.LOGS_DIR", tmp_path)
    monkeypatch.setattr("retrieval.chroma_index.LARGE_REPO_THRESHOLD", 0)

    captured = {}

    def fake_run_query(repo_path, question, model, max_files, mcp_server_url, context_prefix=None):
        captured["context_prefix"] = context_prefix
        return ParsedResponse(thinking="", json_header={"confidence": "low"}, prose="answer", tool_results=[], usage={})

    monkeypatch.setattr(cli, "run_query", fake_run_query)
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--repo", str(tmp_path), "--question", "where is login?", "--port", "8013"])

    assert result.exit_code == 0, result.output
    assert "auto-enabling --index" in result.output
    assert captured["context_prefix"] is not None
    assert "login" in captured["context_prefix"]


def test_cli_small_repo_does_not_auto_activate_index(tmp_path, monkeypatch):
    (tmp_path / "auth.py").write_text("def login():\n    pass\n")
    monkeypatch.setattr("qa_types.session.LOGS_DIR", tmp_path)

    captured = {}

    def fake_run_query(repo_path, question, model, max_files, mcp_server_url, context_prefix=None):
        captured["context_prefix"] = context_prefix
        return ParsedResponse(thinking="", json_header={"confidence": "low"}, prose="answer", tool_results=[], usage={})

    monkeypatch.setattr(cli, "run_query", fake_run_query)
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--repo", str(tmp_path), "--question", "where is login?", "--port", "8014"])

    assert result.exit_code == 0, result.output
    assert "auto-enabling --index" not in result.output
    assert captured["context_prefix"] is None


def test_cli_tunnel_flag_uses_ngrok_url(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("hello\n")
    monkeypatch.setattr("qa_types.session.LOGS_DIR", tmp_path)

    calls = {"start_port": None, "stop_count": 0}

    def fake_start_tunnel(port):
        calls["start_port"] = port
        return "https://fake.ngrok.io/mcp/"

    def fake_stop_tunnel():
        calls["stop_count"] += 1

    captured = {}

    def fake_run_query(repo_path, question, model, max_files, mcp_server_url, context_prefix=None):
        captured["mcp_server_url"] = mcp_server_url
        return ParsedResponse(thinking="", json_header={"confidence": "low"}, prose="answer", tool_results=[], usage={})

    monkeypatch.setattr(cli, "_start_ngrok_tunnel", fake_start_tunnel)
    monkeypatch.setattr(cli, "_stop_ngrok_tunnel", fake_stop_tunnel)
    monkeypatch.setattr(cli, "run_query", fake_run_query)

    runner = CliRunner()
    result = runner.invoke(cli.main, ["--repo", str(tmp_path), "--question", "q?", "--tunnel", "--port", "8015"])

    assert result.exit_code == 0, result.output
    assert calls["start_port"] == 8015
    assert captured["mcp_server_url"] == "https://fake.ngrok.io/mcp/"
    assert "ngrok tunnel: https://fake.ngrok.io/mcp/" in result.output
    assert calls["stop_count"] == 1


def test_cli_tunnel_stopped_even_when_query_raises(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("hello\n")
    monkeypatch.setattr("qa_types.session.LOGS_DIR", tmp_path)

    calls = {"stop_count": 0}
    monkeypatch.setattr(cli, "_start_ngrok_tunnel", lambda port: "https://fake.ngrok.io/mcp/")
    monkeypatch.setattr(cli, "_stop_ngrok_tunnel", lambda: calls.__setitem__("stop_count", calls["stop_count"] + 1))

    def raising_run_query(repo_path, question, model, max_files, mcp_server_url, context_prefix=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "run_query", raising_run_query)

    runner = CliRunner()
    result = runner.invoke(cli.main, ["--repo", str(tmp_path), "--question", "q?", "--tunnel", "--port", "8016"])

    assert result.exit_code != 0
    assert calls["stop_count"] == 1


def test_cli_without_tunnel_flag_ngrok_never_touched(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("hello\n")
    monkeypatch.setattr("qa_types.session.LOGS_DIR", tmp_path)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("ngrok should not be touched without --tunnel")

    monkeypatch.setattr(cli, "_start_ngrok_tunnel", fail_if_called)
    monkeypatch.setattr(cli, "_stop_ngrok_tunnel", fail_if_called)

    def fake_run_query(repo_path, question, model, max_files, mcp_server_url, context_prefix=None):
        return ParsedResponse(thinking="", json_header={"confidence": "low"}, prose="answer", tool_results=[], usage={})

    monkeypatch.setattr(cli, "run_query", fake_run_query)
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--repo", str(tmp_path), "--question", "q?", "--port", "8017"])

    assert result.exit_code == 0, result.output
