from __future__ import annotations
import json
import sys
from pathlib import Path
from urllib.parse import unquote

import anyio
from mcp.server import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.stdio import stdio_server
from mcp.types import Resource, ResourceTemplate

from server.navigation_log import NavigationLog

EXCLUDED_DIRS = {".git", "node_modules", "__pycache__", "build", "dist", ".venv"}
EXCLUDED_EXTS = {".pyc", ".pyo", ".class", ".o"}
MAX_DEPTH = 6
MAX_FILE_BYTES = 500_000

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
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


server = Server("codebase-qa")


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


if __name__ == "__main__":
    anyio.run(main)
