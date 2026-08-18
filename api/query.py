from __future__ import annotations
import os

from anthropic import Anthropic

from api.stream_parser import ParsedResponse, parse_stream

DEFAULT_MODEL = "claude-sonnet-5"
MAX_TOKENS = 8000
THINKING_EFFORT = "medium"
MCP_BETA_FLAG = "mcp-client-2025-04-04"

# The Anthropic MCP connector calls this URL from Anthropic's servers, so it
# must be a publicly reachable HTTPS endpoint (e.g. an ngrok tunnel in front
# of `python server/mcp_server.py --http`) — localhost will not work.
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8000/mcp/")

SYSTEM_PROMPT_TEMPLATE = """
You are a senior engineer on the codebase at {repo_path}.
You have access to the repository via MCP tools: list_tree, read_file, git_log,
git_diff, git_blame, search_symbol.

## Required behaviour
1. ALWAYS call list_tree first before reading any files.
2. Before calling read_file on each file, state in one sentence why you are reading it.
3. After reading each file, state in one sentence what you concluded.
4. Read at most {max_files} files per response.
5. Every factual claim MUST cite an exact file path and line range.
   Format: `src/auth/validate.py:45-67`
6. Never state something as fact if inferred from a file name alone.
7. Stop reading files once you have sufficient evidence. Do not read speculatively.
8. If earlier turns are shown above, you may build on those findings without re-reading
   the same files, but every new factual claim still needs its own file:line citation.

## Response format
Begin every response with this JSON block, then your prose answer:
{{ "confidence": "high|partial|low",
   "confidence_reason": "one sentence",
   "uncertainty_type": "epistemic|aleatoric|both|none",
   "files_read": [...],
   "files_read_reasons": [...],
   "call_graph_complete": true|false,
   "caveats": "..." }}

## When you cannot find something
{{ "result": "not_found", "files_checked": [...],
   "closest_match": "file:line or null",
   "suggested_next_steps": [...] }}

## What not to do
- Do not read the entire repository.
- Do not make claims without file:line citations.
- Do not guess when uncertain — say so explicitly.
- Do not set uncertainty_type to "none" if call_graph_complete is false.
"""


def build_user_content(question: str, context_prefix: str | None) -> str:
    return f"{context_prefix}\n\n{question}" if context_prefix else question


def run_query(
    repo_path: str,
    question: str,
    model: str = DEFAULT_MODEL,
    max_files: int = 8,
    thinking_enabled: bool = True,
    client: Anthropic | None = None,
    mcp_server_url: str | None = None,
    context_prefix: str | None = None,
    history: list[dict] | None = None,
) -> ParsedResponse:
    client = client or Anthropic()
    content = build_user_content(question, context_prefix)
    messages = [*(history or []), {"role": "user", "content": content}]
    params = dict(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT_TEMPLATE.format(repo_path=repo_path, max_files=max_files),
        messages=messages,
        mcp_servers=[{"type": "url", "url": mcp_server_url or MCP_SERVER_URL, "name": "codebase"}],
        betas=[MCP_BETA_FLAG],
    )
    if thinking_enabled:
        # claude-sonnet-5 rejects the older thinking.type=enabled/budget_tokens shape
        # ("not supported for this model") and wants adaptive thinking + output_config.effort.
        params["thinking"] = {"type": "adaptive"}
        params["output_config"] = {"effort": THINKING_EFFORT}

    with client.beta.messages.stream(**params) as stream:
        return parse_stream(stream)
