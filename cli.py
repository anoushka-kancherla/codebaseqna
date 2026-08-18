from __future__ import annotations
import json
import os
import re
import threading
import time
from pathlib import Path

import click
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from api.query import DEFAULT_MODEL, run_query
from server import mcp_server
from server.navigation_log import NavigationLog
from qa_types.session import build_session_log
from xai.attribution import check_citations
from xai.confidence import apply_confidence_overrides

CITATION_RE = re.compile(r"`[^`]+\.\w+:\d+(?:-\d+)?`")
CHROMA_INDEX_PATH = "./.chroma"


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


def _build_chroma_index(repo_path: Path) -> None:
    from retrieval.chroma_index import build_index

    build_index(str(repo_path), index_path=CHROMA_INDEX_PATH)


def _query_chroma_context(question: str) -> str | None:
    from retrieval.chroma_index import query_index

    chunks = query_index(question, index_path=CHROMA_INDEX_PATH)
    if not chunks:
        return None
    return "\n\n".join(f"# {c['file']}:{c['start']}\n{c['content']}" for c in chunks)


def _print_answer_panels(response, attribution: dict, confidence: dict, total_files: int, coverage_pct: float):
    if response.thinking:
        click.secho(f"\n[thinking] {response.thinking[:400]} [--verbose to expand]", dim=True)

    if response.json_header.get("result") == "not_found":
        click.secho("\nSearch report", bold=True)
        click.echo(f"  files_checked: {response.json_header.get('files_checked', [])}")
        click.echo(f"  closest_match: {response.json_header.get('closest_match')}")
        click.echo(f"  suggested_next_steps: {response.json_header.get('suggested_next_steps', [])}")
    else:
        click.echo("\n" + _highlight_citations(response.prose))
        for citation in attribution["citations"]:
            if citation["status"] != "VALID":
                click.secho(
                    f"  [WARNING] {citation['status']}: {citation['file']}:{citation['start']}-{citation['end']}",
                    fg="red",
                )

    level = (confidence["level"] or "unknown").upper()
    click.echo(f"\nConfidence: {level}  ({confidence['uncertainty_type']})")

    files_read = response.json_header.get("files_read", [])
    reasons = response.json_header.get("files_read_reasons", [])
    click.echo(f"\nFiles read: {len(files_read)} of {total_files} total ({coverage_pct}%)")
    for path, reason in zip(files_read, reasons):
        click.echo(f"  — {path}: {reason}")


def _answer_question(
    question: str, repo_path: Path, model: str, max_files: int, mcp_server_url: str,
    context_prefix: str | None, output: str, total_files: int,
):
    response = run_query(
        repo_path=str(repo_path),
        question=question,
        model=model,
        max_files=max_files,
        mcp_server_url=mcp_server_url,
        context_prefix=context_prefix,
    )

    is_not_found = response.json_header.get("result") == "not_found"
    attribution = {"citations": [], "valid_count": 0, "invalid_count": 0} if is_not_found else check_citations(response.prose, repo_path)
    confidence = apply_confidence_overrides(response.json_header)

    session = build_session_log(
        repo_path=str(repo_path),
        question=question,
        model=model,
        max_files=max_files,
        thinking_enabled=True,
        nav_log=mcp_server.NAV_LOG,
        total_files_in_repo=total_files,
        response=response,
        attribution=attribution,
        confidence=confidence,
    )
    session_path = session.save()

    if output == "json":
        click.echo(json.dumps(session.to_dict(), indent=2))
        return

    _print_answer_panels(response, attribution, confidence, total_files, session.navigation["coverage_pct"])
    click.echo(f"\nSession saved to {session_path}")


def _print_stability_report(result: dict):
    click.echo(f"\nStability check ({result['runs']} runs)")
    click.echo(f"Mean Jaccard: {result['mean_jaccard']}  Rating: {result['rating'].upper()}")
    click.echo("\nPairwise scores:")
    for p in result["pairwise_jaccard"]:
        click.echo(f"  run {p['run_a']} vs run {p['run_b']}: {p['jaccard']}")
    click.echo(f"\nConsistent files ({len(result['consistent_files'])}): {result['consistent_files']}")
    click.echo(f"Inconsistent files ({len(result['inconsistent_files'])}): {result['inconsistent_files']}")


@click.command()
@click.option("--repo", type=click.Path(exists=True, file_okay=False), help="Path to the repository.")
@click.option("--question", help="Natural language question about the repo.")
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
@click.option("--max-files", default=8, show_default=True, help="Max files Claude may read.")
@click.option("--port", default=8000, show_default=True, help="Local port for the MCP server.")
@click.option("--output", type=click.Choice(["text", "json"]), default="text", show_default=True)
@click.option("--verify", "verify_session_id", default=None, help="Run faithfulness verification on an existing session id instead of asking a new question.")
@click.option("--stability", is_flag=True, help="Run the question N times and report answer stability (Jaccard over files read).")
@click.option("--runs", default=3, show_default=True, help="Number of runs for --stability.")
@click.option("--index", "use_index", is_flag=True, help="Build/query a ChromaDB index and prepend relevant chunks to the question (best for repos >500 files).")
@click.option("--yes", "-y", is_flag=True, help="Skip the cost confirmation prompt for --stability.")
@click.option("--interactive", "-i", is_flag=True, help="Ask multiple questions against one repo without restarting the MCP server between them.")
def main(
    repo: str | None, question: str | None, model: str, max_files: int, port: int, output: str,
    verify_session_id: str | None, stability: bool, runs: int, use_index: bool, yes: bool,
    interactive: bool,
):
    if interactive and (verify_session_id or stability):
        raise click.UsageError("--interactive cannot be combined with --verify or --stability.")

    if verify_session_id:
        from xai.faithfulness import FAITHFULNESS_WARNING_THRESHOLD, run_faithfulness_check

        result = run_faithfulness_check(verify_session_id, model=model)
        click.echo(json.dumps(result, indent=2))
        if result["faithfulness_score"] < FAITHFULNESS_WARNING_THRESHOLD:
            click.secho(f"WARNING: faithfulness score {result['faithfulness_score']} is below threshold", fg="yellow")
        return

    if not repo or (not question and not interactive):
        raise click.UsageError("--repo and --question are required unless --verify or --interactive is given.")

    repo_path = Path(repo).resolve()
    total_files = mcp_server._count_files(mcp_server._build_tree(repo_path))
    _start_local_mcp_server(repo_path, port)
    mcp_server_url = os.environ.get("MCP_SERVER_URL", f"http://127.0.0.1:{port}/mcp/")

    if use_index:
        _build_chroma_index(repo_path)

    if stability:
        from xai.stability import run_stability_check

        context_prefix = _query_chroma_context(question) if use_index else None
        click.secho(f"This runs the question {runs} times (~{runs}x normal API cost).", fg="yellow")
        if not yes and not click.confirm("Proceed?", default=True):
            click.echo("Aborted.")
            return
        result = run_stability_check(
            question, str(repo_path), runs=runs, model=model, max_files=max_files,
            mcp_server_url=mcp_server_url, context_prefix=context_prefix,
        )
        if output == "json":
            click.echo(json.dumps(result, indent=2))
        else:
            _print_stability_report(result)
        return

    if interactive:
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                click.echo("\nGoodbye.")
                break
            if q in ("", "exit", "quit"):
                break
            mcp_server.NAV_LOG = NavigationLog()
            context_prefix = _query_chroma_context(q) if use_index else None
            _answer_question(q, repo_path, model, max_files, mcp_server_url, context_prefix, output, total_files)
        return

    context_prefix = _query_chroma_context(question) if use_index else None
    _answer_question(question, repo_path, model, max_files, mcp_server_url, context_prefix, output, total_files)


if __name__ == "__main__":
    main()
