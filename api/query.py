from __future__ import annotations
import os

from anthropic import Anthropic

from api.stream_parser import ParsedResponse, parse_stream

DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 8000
THINKING_BUDGET = 5000
MCP_BETA_FLAG = "mcp-client-2025-04-04"

# The Anthropic MCP connector calls this URL from Anthropic's servers, so it
# must be a publicly reachable HTTPS endpoint (e.g. an ngrok tunnel in front
# of `python server/mcp_server.py --http`) — localhost will not work.
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8000/mcp/")

SYSTEM_PROMPT_TEMPLATE = """
You are a senior engineer on the codebase at {repo_path}.
You have access to the repository via MCP resources and tools.

## Required behaviour
1. ALWAYS read repo://tree first before opening any files.
2. Before reading each file, state in one sentence why you are reading it.
3. After reading each file, state in one sentence what you concluded.
4. Read at most {max_files} files per response.
5. Every factual claim MUST cite an exact file path and line range.
   Format: `src/auth/validate.py:45-67`
6. Never state something as fact if inferred from a file name alone.
7. Stop reading files once you have sufficient evidence. Do not read speculatively.

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


def run_query(
    repo_path: str,
    question: str,
    model: str = DEFAULT_MODEL,
    max_files: int = 8,
    thinking_enabled: bool = True,
    client: Anthropic | None = None,
    mcp_server_url: str | None = None,
) -> ParsedResponse:
    client = client or Anthropic()
    params = dict(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT_TEMPLATE.format(repo_path=repo_path, max_files=max_files),
        messages=[{"role": "user", "content": question}],
        stream=True,
        mcp_servers=[{"type": "url", "url": mcp_server_url or MCP_SERVER_URL, "name": "codebase"}],
        betas=[MCP_BETA_FLAG],
    )
    if thinking_enabled:
        params["thinking"] = {"type": "enabled", "budget_tokens": THINKING_BUDGET}

    with client.beta.messages.stream(**params) as stream:
        return parse_stream(stream)
