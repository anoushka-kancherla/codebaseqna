from pathlib import Path

from server import mcp_server
from server.navigation_log import NavigationLog

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _fresh(monkeypatch):
    monkeypatch.setattr(mcp_server, "ROOT", REPO_ROOT)
    monkeypatch.setattr(mcp_server, "NAV_LOG", NavigationLog())


def test_call_registered_tool_logs_and_wraps_git_log(monkeypatch):
    _fresh(monkeypatch)
    result = mcp_server.call_registered_tool("git_log", {"n": 3})
    assert "result" in result
    assert len(result["result"]) <= 3
    assert mcp_server.NAV_LOG.entries[0].event == "tool_call"
    assert mcp_server.NAV_LOG.entries[0].tool_name == "git_log"
    assert mcp_server.NAV_LOG.entries[0].tool_args == {"n": 3}


def test_call_registered_tool_search_symbol_logs(monkeypatch):
    _fresh(monkeypatch)
    result = mcp_server.call_registered_tool("search_symbol", {"symbol": "NavigationLog"})
    assert len(result["result"]) > 0
    assert mcp_server.NAV_LOG.entries[0].tool_name == "search_symbol"


def test_call_registered_tool_unknown_tool(monkeypatch):
    _fresh(monkeypatch)
    result = mcp_server.call_registered_tool("nonexistent_tool", {})
    assert result == {"error": "UNKNOWN_TOOL", "tool_name": "nonexistent_tool"}


def test_list_tree_tool_logs_tool_call_and_tree_read(monkeypatch):
    # Claude reaches repo listing via the list_tree *tool*, not the repo://tree
    # *resource* — the Anthropic MCP connector only exposes tools to the model,
    # confirmed by a real run where Claude said "I don't have a direct tool to
    # fetch repo://tree" and never issued a single resource read.
    _fresh(monkeypatch)
    result = mcp_server.call_registered_tool("list_tree", {})
    assert "total_files" in result["result"]
    assert result["result"]["total_files"] > 0
    events = [e.event for e in mcp_server.NAV_LOG.entries]
    assert events == ["tool_call", "tree_read"]


def test_read_file_tool_logs_tool_call_and_file_read(monkeypatch):
    _fresh(monkeypatch)
    result = mcp_server.call_registered_tool("read_file", {"path": "README.md"})
    assert result["result"]["path"] == "README.md"
    assert "content" in result["result"]
    events = [e.event for e in mcp_server.NAV_LOG.entries]
    assert events == ["tool_call", "file_read"]


def test_read_file_tool_path_traversal_still_rejected(monkeypatch):
    _fresh(monkeypatch)
    result = mcp_server.call_registered_tool("read_file", {"path": "../../etc/passwd"})
    assert result["result"]["error"] == "PATH_TRAVERSAL"


async def _dispatch_through_protocol_handler(tool_name: str, args: dict):
    # Exercise the real registered handler (not just call_registered_tool directly)
    # so the dict-wrapping required by mcp.server.Server.call_tool is verified
    # against the actual framework, not just our own assumptions about it.
    handler = mcp_server.server.request_handlers[__import__("mcp.types", fromlist=["CallToolRequest"]).CallToolRequest]
    import mcp.types as types
    req = types.CallToolRequest(method="tools/call", params=types.CallToolRequestParams(name=tool_name, arguments=args))
    return await handler(req)


def test_git_log_survives_real_protocol_handler(monkeypatch):
    import anyio
    _fresh(monkeypatch)
    result = anyio.run(_dispatch_through_protocol_handler, "git_log", {"n": 2})
    assert result.root.isError is False
    assert result.root.structuredContent["result"]


def test_git_diff_str_survives_real_protocol_handler(monkeypatch):
    import anyio
    import git as gitmod
    _fresh(monkeypatch)
    commit_hash = next(gitmod.Repo(REPO_ROOT).iter_commits(max_count=1)).hexsha
    result = anyio.run(_dispatch_through_protocol_handler, "git_diff", {"commit_hash": commit_hash})
    assert result.root.isError is False
    assert isinstance(result.root.structuredContent["result"], str)
