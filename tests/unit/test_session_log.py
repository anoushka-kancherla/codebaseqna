import json

from api.stream_parser import ParsedResponse
from qa_types.session import build_session_log
from server.navigation_log import NavigationLog


def test_build_session_log_computes_coverage():
    nav_log = NavigationLog()
    nav_log.record_tree_read()
    nav_log.record_file_read("a.py", "line1\nline2\n")
    nav_log.record_file_read("b.py", "line1\n")

    response = ParsedResponse(
        thinking="thinking about it",
        json_header={"confidence": "high"},
        prose="a.py:1-2 handles this.",
        tool_results=[],
        usage={"output_tokens": 5},
    )

    session = build_session_log(
        repo_path="/repo",
        question="where is X?",
        model="claude-sonnet-4-20250514",
        max_files=8,
        thinking_enabled=True,
        nav_log=nav_log,
        total_files_in_repo=10,
        response=response,
    )

    assert session.navigation["files_read"] == 2
    assert session.navigation["coverage_pct"] == 20.0
    assert session.navigation["reads"][0]["event"] == "tree_read"
    assert session.attribution is None
    assert session.confidence is None
    json.dumps(session.to_dict())


def test_session_save_writes_file(tmp_path, monkeypatch):
    import qa_types.session as session_mod
    monkeypatch.setattr(session_mod, "LOGS_DIR", tmp_path)

    nav_log = NavigationLog()
    response = ParsedResponse(thinking="", json_header={}, prose="answer", tool_results=[], usage={})
    session = build_session_log(
        repo_path="/repo", question="q", model="m", max_files=8,
        thinking_enabled=True, nav_log=nav_log, total_files_in_repo=1, response=response,
    )
    path = session.save()
    assert path.exists()
    assert json.loads(path.read_text())["session_id"] == nav_log.session_id
