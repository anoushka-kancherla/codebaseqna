from __future__ import annotations
import json
import os
import re
import threading
import time
from pathlib import Path

import click
import uvicorn

from api.query import DEFAULT_MODEL, run_query
from server import mcp_server
from server.navigation_log import NavigationLog
from qa_types.session import build_session_log

CITATION_RE = re.compile(r"`[^`]+\.\w+:\d+(?:-\d+)?`")


def _start_local_mcp_server(repo_path: Path, port: int) -> uvicorn.Server:
    mcp_server.ROOT = repo_path
    mcp_server.NAV_LOG = NavigationLog()
    config = uvicorn.Config(mcp_server.build_http_app(), host="127.0.0.1", port=port, log_level="warning")
    server_instance = uvicorn.Server(config)
    thread = threading.Thread(target=server_instance.run, daemon=True)
    thread.start()
    while not server_instance.started:
        time.sleep(0.05)
    return server_instance


def _highlight_citations(prose: str) -> str:
    return CITATION_RE.sub(lambda m: click.style(m.group(0), fg="cyan", bold=True), prose)


@click.command()
@click.option("--repo", required=True, type=click.Path(exists=True, file_okay=False), help="Path to the repository.")
@click.option("--question", required=True, help="Natural language question about the repo.")
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
@click.option("--max-files", default=8, show_default=True, help="Max files Claude may read.")
@click.option("--port", default=8000, show_default=True, help="Local port for the MCP server.")
@click.option("--output", type=click.Choice(["text", "json"]), default="text", show_default=True)
def main(repo: str, question: str, model: str, max_files: int, port: int, output: str):
    repo_path = Path(repo).resolve()
    total_files = mcp_server._count_files(mcp_server._build_tree(repo_path))

    _start_local_mcp_server(repo_path, port)
    response = run_query(
        repo_path=str(repo_path),
        question=question,
        model=model,
        max_files=max_files,
        mcp_server_url=os.environ.get("MCP_SERVER_URL", f"http://127.0.0.1:{port}/mcp/"),
    )

    session = build_session_log(
        repo_path=str(repo_path),
        question=question,
        model=model,
        max_files=max_files,
        thinking_enabled=True,
        nav_log=mcp_server.NAV_LOG,
        total_files_in_repo=total_files,
        response=response,
    )
    session_path = session.save()

    if output == "json":
        click.echo(json.dumps(session.to_dict(), indent=2))
        return

    if response.thinking:
        excerpt = response.thinking[:400]
        click.secho(f"\n[thinking] {excerpt} [--verbose to expand]", dim=True)

    click.echo("\n" + _highlight_citations(response.prose))
    click.echo(f"\nSession saved to {session_path}")


if __name__ == "__main__":
    main()
