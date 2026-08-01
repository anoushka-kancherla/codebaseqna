from __future__ import annotations
import json
import sys
from pathlib import Path
from urllib.parse import unquote

import anyio
import git
from mcp.server import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.stdio import stdio_server
from mcp.types import Resource, ResourceTemplate, Tool

from server import git_tools
from server.navigation_log import NavigationLog

EXCLUDED_DIRS = {".git", "node_modules", "__pycache__", "build", "dist", ".venv"}
EXCLUDED_EXTS = {".pyc", ".pyo", ".class", ".o"}
MAX_DEPTH = 6
MAX_FILE_BYTES = 500_000

_positional_args = [a for a in sys.argv[1:] if not a.startswith("--")]
ROOT = Path(_positional_args[0] if _positional_args else ".").resolve()
NAV_LOG = NavigationLog()


def error_resource(code: str, path: str) -> dict:
    return {"error": code, "path": path}


def _build_tree(dir_path: Path, depth: int = 0) -> list[dict]:
    if depth >= MAX_DEPTH:
        return []
    nodes = []
    for entry in sorted(dir_path.iterdir(), key=lambda p: p.name):
        if entry.is_dir():
            if entry.name in EXCLUDED_DIRS:
                continue
            nodes.append({"name": entry.name, "type": "dir", "children": _build_tree(entry, depth + 1)})
        elif entry.suffix not in EXCLUDED_EXTS:
            nodes.append({"name": entry.name, "type": "file"})
    return nodes


def _count_files(nodes: list[dict]) -> int:
    total = 0
    for node in nodes:
        if node["type"] == "file":
            total += 1
        else:
            total += _count_files(node["children"])
    return total


def get_tree_json() -> str:
    tree = _build_tree(ROOT)
    NAV_LOG.record_tree_read()
    return json.dumps({"tree": tree, "total_files": _count_files(tree)})


def get_file_json(rel: str) -> str:
    resolved = (ROOT / rel).resolve()
    if not str(resolved).startswith(str(ROOT.resolve())):
        return json.dumps(error_resource("PATH_TRAVERSAL", rel))
    if not resolved.is_file():
        return json.dumps(error_resource("NOT_FOUND", rel))
    try:
        content = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return json.dumps(error_resource("BINARY_FILE", rel))

    if resolved.stat().st_size > MAX_FILE_BYTES:
        content = content[:MAX_FILE_BYTES] + "\n[TRUNCATED — file exceeds 500 KB]"

    NAV_LOG.record_file_read(rel, content)
    return json.dumps({"path": rel, "content": content})


def _get_repo() -> git.Repo | None:
    try:
        return git.Repo(ROOT)
    except git.InvalidGitRepositoryError:
        return None


TOOLS = [
    Tool(
        name="git_log",
        description="List recent commits.",
        inputSchema={"type": "object", "properties": {"n": {"type": "integer", "default": 20}}},
    ),
    Tool(
        name="git_diff",
        description="Show the diff introduced by a commit.",
        inputSchema={
            "type": "object",
            "properties": {"commit_hash": {"type": "string"}},
            "required": ["commit_hash"],
        },
    ),
    Tool(
        name="git_blame",
        description="Blame a line range of a file.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
            "required": ["file_path", "start_line", "end_line"],
        },
    ),
    Tool(
        name="search_symbol",
        description="Search the repo for a literal symbol/string.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "file_extensions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="list_tree",
        description="List the repo's file tree. Call this first, before reading any files.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="read_file",
        description="Read a file's contents by path, relative to the repo root.",
        inputSchema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    ),
]


def call_registered_tool(tool_name: str, args: dict) -> dict:
    # Tool results must come back as a dict — the MCP server treats a bare
    # list/str return as raw ContentBlocks (and iterates a str char-by-char),
    # so every branch here wraps its result as {"result": ...}.
    NAV_LOG.record_tool_call(tool_name, args)
    if tool_name == "git_log":
        repo = _get_repo()
        if not repo:
            return {"error": "NOT_A_GIT_REPO"}
        return {"result": git_tools.git_log(repo, args.get("n", 20))}
    if tool_name == "git_diff":
        repo = _get_repo()
        if not repo:
            return {"error": "NOT_A_GIT_REPO"}
        return {"result": git_tools.git_diff(repo, args["commit_hash"])}
    if tool_name == "git_blame":
        repo = _get_repo()
        if not repo:
            return {"error": "NOT_A_GIT_REPO"}
        return {"result": git_tools.git_blame(repo, args["file_path"], args["start_line"], args["end_line"])}
    if tool_name == "search_symbol":
        return {"result": git_tools.search_symbol(ROOT, args["symbol"], args.get("file_extensions"))}
    if tool_name == "list_tree":
        # get_tree_json/get_file_json already instrument NAV_LOG and return
        # JSON strings for the resource handler below; reuse them here too
        # rather than duplicating the path-safety and truncation logic.
        return {"result": json.loads(get_tree_json())}
    if tool_name == "read_file":
        return {"result": json.loads(get_file_json(args["path"]))}
    return {"error": "UNKNOWN_TOOL", "tool_name": tool_name}


server = Server("codebase-qa")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@server.call_tool()
async def call_tool(tool_name: str, args: dict):
    return call_registered_tool(tool_name, args)


@server.list_resources()
async def list_resources() -> list[Resource]:
    return [Resource(uri="repo://tree", name="repo-tree", mimeType="application/json")]


@server.list_resource_templates()
async def list_resource_templates() -> list[ResourceTemplate]:
    return [ResourceTemplate(
        uriTemplate="repo://file/{path}", name="repo-file", mimeType="application/json"
    )]


@server.read_resource()
async def read_resource(uri) -> list[ReadResourceContents]:
    uri_str = str(uri)
    if uri_str == "repo://tree":
        data = get_tree_json()
    elif uri_str.startswith("repo://file/"):
        rel = unquote(uri_str[len("repo://file/"):])
        data = get_file_json(rel)
    else:
        data = json.dumps(error_resource("NOT_FOUND", uri_str))
    return [ReadResourceContents(content=data, mime_type="application/json")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def build_http_app():
    # The Anthropic API's MCP connector reaches this server over the network,
    # so for real use this must sit behind a public HTTPS URL (e.g. ngrok) —
    # `localhost` is only good for local smoke-testing with curl.
    import contextlib
    from starlette.applications import Starlette
    from starlette.routing import Mount
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    session_manager = StreamableHTTPSessionManager(app=server, stateless=True)

    async def handle_mcp(scope, receive, send):
        await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        async with session_manager.run():
            yield

    return Starlette(routes=[Mount("/mcp", app=handle_mcp)], lifespan=lifespan)


def run_http(port: int = 8000):
    import uvicorn
    uvicorn.run(build_http_app(), host="0.0.0.0", port=port)


if __name__ == "__main__":
    if "--http" in sys.argv:
        run_http()
    else:
        anyio.run(main)
