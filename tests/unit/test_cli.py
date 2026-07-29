import json

from click.testing import CliRunner

import cli
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
    result = runner.invoke(cli.main, ["--repo", str(tmp_path), "--question", "where is the greeting?"])

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
