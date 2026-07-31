import json

from click.testing import CliRunner

import cli
import xai.faithfulness as faithfulness_mod
from api.stream_parser import ParsedResponse
from server import mcp_server


def test_cli_end_to_end_with_mocked_claude(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("hello\nworld\n")
    monkeypatch.setattr("qa_types.session.LOGS_DIR", tmp_path)

    def fake_run_query(repo_path, question, model, max_files, mcp_server_url):
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

    def fake_run_query(repo_path, question, model, max_files, mcp_server_url):
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

    def fake_run_query(repo_path, question, model, max_files, mcp_server_url):
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
