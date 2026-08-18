# Local Codebase Q&A Tool — Claude Code Guide

You are building a CLI-first developer tool that answers natural language questions about a
local code repository. The user points the tool at a repo and asks questions like "where is
authentication handled?" or "what changed in the last commit that caused this bug?".
Claude reads only the relevant files via a custom MCP server, returns grounded answers with
exact `file:line` citations, confidence ratings, and a full audit trail. The XAI layer is
the primary differentiator — every answer is attributable, scoped, and verifiable.

---

## Project state

Work through phases in order. Each phase has a validation checklist — do not advance until
all checks pass. Phases 1 and 2 are strictly sequential. Phases 3 and 4 can be parallelised
once Phase 2 is complete. Phase 5 depends on all prior phases.

| Phase | Title | Hours | Risk |
|---|---|---|---|
| 1 | Custom MCP Filesystem Server | 3–4 | Low |
| 2 | Claude API Integration & Selective Retrieval | 3–4 | **High** |
| 3 | Git Tools via MCP | 2–3 | Medium |
| 4 | XAI Layer — All Five Principles | 3–4 | Medium |
| 5 | Stability Analysis, ChromaDB & Polish | 2–3 | Low |

---

## Architecture

```
User (CLI)
    │
    ▼
cli.py  (Click)
    │
    ▼
api/query.py  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────  Anthropic API
(query handler,                                                                                                                claude-sonnet-4-20250514
 stream parser,                                                                                                                tool_use + thinking
 session writer)                                                                                                                    │
    │                                                                                                                               ▼
    ▼                                                                                                                       server/mcp_server.py
xai/  (post-processing,                                                                                                     Resources:
 no side effects except                                                                                                       repo://tree
 filesystem reads for --verify)                                                                                               repo://file/{path}
    │                                                                                                                       Tools:
    ▼                                                                                                                         git_log, git_diff
logs/{session_id}.json                                                                                                        git_blame, search_symbol
```

The MCP server instruments every read into `server/navigation_log.py`. That log is the
foundation of all XAI features — scope panel, audit trail, stability score. Getting
instrumentation right in Phase 1 is the most important work in the project.

---

## File structure

```
codebase-qa/
├── CLAUDE.md
├── cli.py                          # Click entry point — build in layers across phases
├── .env                            # ANTHROPIC_API_KEY (never commit)
├── requirements.txt
├── server/
│   ├── mcp_server.py               # MCP server with two resources + tool registry
│   ├── navigation_log.py           # NavigationLog class — instrument every read here
│   └── git_tools.py                # git_log, git_diff, git_blame, search_symbol
├── api/
│   ├── query.py                    # Core query handler, API call construction
│   └── stream_parser.py            # Parse streaming response into typed blocks
├── xai/
│   ├── attribution.py              # Principle 1: parse + validate file:line citations
│   ├── scope.py                    # Principle 2: scope transparency from nav log
│   ├── negative.py                 # Principle 3: not_found handler
│   ├── confidence.py               # Principle 4: confidence + uncertainty type
│   ├── faithfulness.py             # Principle 5a: --verify second Claude call
│   └── stability.py                # Principle 5b: --stability Jaccard score
├── retrieval/
│   └── chroma_index.py             # Optional ChromaDB for repos >500 files (Phase 5)
├── types/
│   ├── session.py                  # SessionLog, NavigationEntry dataclasses
│   └── xai.py                      # Attribution, Confidence, Faithfulness types
├── tests/
│   ├── unit/
│   │   ├── test_navigation_log.py
│   │   ├── test_stream_parser.py
│   │   ├── test_attribution.py
│   │   └── test_confidence.py
│   └── integration/
│       └── test_full_query.py      # Run against tests/fixtures/sample_repo/
└── logs/                           # Session JSON audit logs (gitignore this)
```

---

## Phase 1 — Custom MCP Filesystem Server

### What to build

Two files in order: `server/navigation_log.py` first, then `server/mcp_server.py`.
Build and test the MCP server entirely without Claude — use the MCP inspector to
validate before connecting anything to the API.

### server/navigation_log.py

```python
from dataclasses import dataclass, field
from typing import Literal
import time, json, uuid

@dataclass
class NavigationEntry:
    timestamp: float
    event: Literal["file_read", "tree_read", "tool_call"]
    path: str | None = None
    tool_name: str | None = None
    tool_args: dict | None = None
    lines: int | None = None
    size_bytes: int | None = None

class NavigationLog:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.entries: list[NavigationEntry] = []

    def record_file_read(self, path: str, content: str) -> None:
        self.entries.append(NavigationEntry(
            timestamp=time.time(), event="file_read", path=path,
            lines=content.count("\n") + 1,
            size_bytes=len(content.encode("utf-8"))
        ))

    def record_tree_read(self) -> None:
        self.entries.append(NavigationEntry(timestamp=time.time(), event="tree_read"))

    def record_tool_call(self, tool_name: str, args: dict) -> None:
        self.entries.append(NavigationEntry(
            timestamp=time.time(), event="tool_call",
            tool_name=tool_name, tool_args=args
        ))

    def to_dict(self) -> list[dict]:
        return [vars(e) for e in self.entries]
```

`NavigationLog` is instantiated once per CLI session. It is not a singleton — tests
must create fresh instances.

### server/mcp_server.py — critical patterns

**Constants at module level:**
```python
EXCLUDED_DIRS  = {".git", "node_modules", "__pycache__", "build", "dist", ".venv"}
EXCLUDED_EXTS  = {".pyc", ".pyo", ".class", ".o"}
MAX_DEPTH      = 6
MAX_FILE_BYTES = 500_000
```

**Path traversal prevention — use `.resolve()` on both sides:**
```python
resolved = (ROOT / rel).resolve()
if not str(resolved).startswith(str(ROOT.resolve())):
    return error_resource("PATH_TRAVERSAL", rel)
```

Do NOT use string prefix matching on the raw input. `../../etc/passwd` must be caught.

**Instrument every successful read — call into `NAV_LOG` before returning:**
```python
NAV_LOG.record_file_read(rel, content)
return ReadResourceResult(...)
```

**File size cap — truncate, do not error:**
```python
if resolved.stat().st_size > MAX_FILE_BYTES:
    content = content[:MAX_FILE_BYTES] + "\n[TRUNCATED — file exceeds 500 KB]"
```

**Binary files — return error resource, do not crash:**
```python
try:
    content = resolved.read_text(encoding="utf-8")
except UnicodeDecodeError:
    return error_resource("BINARY_FILE", rel)
```

**`repo://tree` must include total file count:**
```python
return ReadResourceResult(contents=[TextContent(type="text",
    text=json.dumps({"tree": tree, "total_files": count_files(tree)})
)])
```

### Phase 1 unit tests — all must pass before Phase 2

Write these in `tests/unit/test_navigation_log.py`:

- `test_file_read_records_entry` — entry has `event="file_read"`, correct line count
- `test_tree_read_records_entry` — entry has `event="tree_read"`, `path` is `None`
- `test_multiple_reads_ordered_by_timestamp` — three reads produce three entries in order
- `test_to_dict_serialisable` — `json.dumps(log.to_dict())` does not raise
- `test_fresh_instance_has_no_entries` — `entries == []` on init
- `test_path_traversal_rejected` — `../../etc/passwd` returns error resource, no log entry
- `test_binary_file_rejected` — binary file returns `error_resource("BINARY_FILE", ...)`, no entry
- `test_large_file_truncated` — file >500 KB returns content ending with `[TRUNCATED...]`
  and DOES log an entry (truncation is logged, just not the full content)

### Phase 1 validation checklist

```
[ ] mcp-inspector server/mcp_server.py — inspector UI opens
[ ] Request repo://tree — JSON tree returned, tree_read entry in nav log
[ ] Request repo://file/README.md — content returned, file_read entry logged with correct line count
[ ] Request repo://file/../../etc/passwd — error resource returned, code=PATH_TRAVERSAL, no log entry
[ ] Request repo://file/nonexistent.py — error resource returned, code=NOT_FOUND
[ ] pytest tests/unit/test_navigation_log.py — all 8 tests pass
[ ] Claude is not involved in any of the above
```

---

## Phase 2 — Claude API Integration & Selective Retrieval

**This is the highest-risk phase.** The main failure mode is Claude reading too many files
at once and exhausting the context window. Validate selective retrieval explicitly.

### Build order

1. `api/stream_parser.py` — before the query handler
2. `api/query.py` — after the stream parser is tested
3. `cli.py` — basic CLI, add XAI panels in Phase 4

### api/stream_parser.py

Parse streaming events into typed blocks. This separation is critical — thinking blocks
must be kept separate from text blocks for the XAI reasoning trace panel.

```python
@dataclass
class ParsedResponse:
    thinking: str
    json_header: dict         # parsed from the start of the text block
    prose: str                # everything after the JSON header
    tool_results: list[dict]  # raw tool_use/tool_result pairs
    usage: dict               # input_tokens, output_tokens, thinking_tokens

def parse_stream(stream) -> ParsedResponse:
    thinking_parts, text_parts, tool_results = [], [], []
    current_block = None
    usage = {}
    for event in stream:
        if event.type == "content_block_start":
            current_block = event.content_block.type
        elif event.type == "content_block_delta":
            if current_block == "thinking":
                thinking_parts.append(event.delta.thinking)
            elif current_block == "text":
                text_parts.append(event.delta.text)
        elif event.type == "message_delta":
            usage = vars(event.usage)
    full_text = "".join(text_parts)
    json_header, prose = split_json_header(full_text)
    return ParsedResponse(
        thinking="".join(thinking_parts),
        json_header=json_header, prose=prose,
        tool_results=tool_results, usage=usage
    )
```

**`split_json_header` must handle both formats — Claude is inconsistent:**
```python
def split_json_header(text: str) -> tuple[dict, str]:
    import re, json
    # Try code-fenced first
    match = re.match(r"^\s*```(?:json)?\s*(\{.*?\})\s*```(.*)", text, re.DOTALL)
    if not match:
        # Try raw JSON
        match = re.match(r"^\s*(\{.*?\})(.*)", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1)), match.group(2).strip()
        except json.JSONDecodeError:
            pass
    return {}, text.strip()
```

Test both patterns explicitly before moving to the query handler.

### api/query.py — system prompt

Store as a module-level constant. Use double curly braces `{{ }}` for literal JSON braces
inside Python `.format()` strings.

```python
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
```

### api/query.py — API call

```python
params = dict(
    model=model,
    max_tokens=8000,
    system=SYSTEM_PROMPT_TEMPLATE.format(repo_path=repo_path, max_files=max_files),
    messages=[{"role": "user", "content": question}],
    stream=True,
    mcp_servers=[{"type": "url", "url": MCP_SERVER_URL, "name": "codebase"}]
)
if thinking_enabled:
    params["thinking"] = {"type": "enabled", "budget_tokens": 5000}

with client.messages.stream(**params) as stream:
    return parse_stream(stream)
```

### cli.py — Phase 2 scope only

In Phase 2, implement only these output panels:
- Thinking panel: first 400 chars of thinking block, dimmed, with `[--verbose to expand]`
- Prose answer: file:line refs highlighted via `click.style()`
- `"Session saved to logs/{id}.json"` — write the log even in Phase 2 with XAI fields as null

Add `--verify` and `--stability` dispatch in Phase 5. Add XAI panels in Phase 4.

### Validating selective retrieval

After the first successful end-to-end query, inspect the navigation log in the session JSON.

- `repo://tree` must appear as the first entry
- For a specific question ("where is the login function?"), Claude should read 3–6 files max
- If Claude reads 15+ files: the system prompt is not being injected correctly —
  verify `{max_files}` is replaced before the API call

**If selective retrieval fails:** add to the system prompt: `"Stop reading files once you
have enough evidence to answer. Do not read files speculatively."`

### Phase 2 validation checklist

```
[ ] python cli.py --repo /path --question "where is the main entry point?"
[ ] Thinking panel appears before prose answer
[ ] Prose answer contains at least one file:line citation
[ ] Session JSON written to ./logs/ with correct session_id
[ ] Navigation log shows repo://tree read first
[ ] Files read count <= max_files (default 8)
[ ] Same question on a 500+ file repo — Claude still reads selectively
[ ] --output json prints structured JSON to stdout
```

---

## Phase 3 — Git Tools via MCP

Register all four tools in `server/mcp_server.py` and log every call via `NAV_LOG.record_tool_call()`.

### Tool signatures

```python
# git_log(n: int = 20) -> list of commits
# Each: {hash, author_name, author_email, date (ISO 8601), message, files_changed}

# git_diff(commit_hash: str) -> diff string
# Cap at 50_000 chars, append "[TRUNCATED]" if exceeded

# git_blame(file_path: str, start_line: int, end_line: int) -> list of line blame
# Each: {line_number, content, commit_hash, author_name, date}
# Wrap repo.blame() in try/except — crashes on merge commits for files
# not present in all parents. Return empty list with error note on failure.

# search_symbol(symbol: str, file_extensions: list[str] | None = None) -> list of matches
# Each: {file_path, line_number, line_content, context_before (2 lines), context_after (2 lines)}
# Cap at 50 results. Skip node_modules, __pycache__, .git in path.parts.
```

### Phase 3 validation checklist

```
[ ] "what changed in the last 3 commits?" — nav log shows tool_call git_log entry
[ ] "who wrote validate_token and when was it last changed?" — nav log shows file_read + git_blame
[ ] "find all places we call requests.get" — search_symbol called, results cited with file:line
[ ] pytest tests/unit/test_git_tools.py — all tests pass
```

---

## Phase 4 — XAI Layer

Each module in `xai/` has exactly one input type and one output type. They are independent
of each other and have no shared state. Test each independently before wiring into the CLI.

### xai/attribution.py — Principle 1

Parse all `` `file.py:N-M` `` patterns from prose. Validate each:

```python
CITATION_RE = re.compile(r"`([^`]+\.\w+):(\d+)(?:-(\d+))?`")

# For each match:
# 1. resolved = (ROOT / file).resolve()
# 2. Check str(resolved).startswith(str(ROOT.resolve()))  -> INVALID_PATH if not
# 3. Check resolved.exists()                               -> INVALID_PATH if not
# 4. Check end_line <= actual line count                   -> INVALID_RANGE if not
# 5. Otherwise: VALID
```

Output: `{citations: [...], valid_count: int, invalid_count: int}`

Invalid citations are flagged and logged. They do not halt the session.

### xai/confidence.py — Principle 4

Apply two validation overrides regardless of what Claude returns:

```python
# Override 1: uncertainty_type cannot be "none" if call_graph_complete is False
if uncertainty_type == "none" and not call_graph_ok:
    logger.warning("Override: uncertainty_type none→epistemic")
    uncertainty_type = "epistemic"

# Override 2: confidence cannot be "high" if files_read is empty
if confidence == "high" and not files_read:
    logger.warning("Override: confidence high→low (files_read empty)")
    confidence = "low"
```

These are enforced programmatically, not trusted from the model output.

### xai/faithfulness.py — Principle 5a

Triggered by `--verify <session-id>`. Reads the session log, makes a second Claude call
to verify each cited claim against the actual file content, writes results back to the log.

**Critical:** Read cited files directly from the filesystem, NOT via MCP. This is
intentional — the verification run must not pollute the original session's navigation log.

```
faithfulness_score = (verified + 0.5 × partial) / total_claims
Range: [0.0, 1.0]. Display WARNING if score < 0.7.
```

Faithfulness verification prompt — return ONLY JSON, no prose, no fences:
```
For each claim: VERIFIED / PARTIAL / UNSUPPORTED
Return: [{claim_index, claim_summary, verdict, evidence, explanation}]
```

### xai/stability.py — Principle 5b (build in Phase 5)

```python
def jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    return len(a & b) / len(a | b)

# Run query N times, collect files_read set from each json_header
# Compute pairwise Jaccard over all N(N-1)/2 pairs
# mean >= 0.7 -> "high", >= 0.4 -> "moderate", < 0.4 -> "low"
```

**Warning:** Each stability run costs N × normal API usage. Show a cost warning in the
CLI before running. Default `--runs 3`.

### CLI output order — add in Phase 4

```
1. Thinking panel (collapsible, dimmed)
2. Prose answer with highlighted citations
3. Confidence: HIGH | PARTIAL | LOW  (epistemic | aleatoric | both | none)
4. Files read: N of M total (X.X%)
   — path: reason-stated-by-claude
   — ...
5. Session saved to logs/{id}.json
```

If `json_header["result"] == "not_found"`, replace the prose panel with a "Search report"
panel showing files_checked, closest_match, and suggested_next_steps.

### Phase 4 validation checklist

```
[ ] All five output panels appear in correct order after a standard query
[ ] Deliberate invalid citation (edit prose in a test) -> INVALID_PATH warning in output + log
[ ] not_found response -> "Search report" panel appears instead of prose
[ ] confidence "high" + empty files_read -> override fires, logged as warning
[ ] python cli.py --repo /path --question "..." --verify -> faithfulness result in session log
[ ] faithfulness score < 0.7 -> WARNING displayed in CLI
[ ] pytest tests/unit/test_attribution.py tests/unit/test_confidence.py -- all pass
```

---

## Phase 5 — Stability Analysis, ChromaDB & Polish

### Stability check

`xai/stability.py` — implement `run_stability_check(question, repo_path, runs=3)`.
Wire into CLI via `--stability --runs N` flags. Display pairwise Jaccard scores,
consistent files, inconsistent files, and stability rating.

### ChromaDB (optional, --index flag)

`retrieval/chroma_index.py`:
- Chunk by function boundary (split on `def ` and `class `)
- `build_index(repo_root, index_path="./.chroma")`
- `query_index(question, n_results=10)` — returns `[{file, start, content}]`
- When `--index` is active, prepend top-k chunks to the user message before the API call
- Activate only for repos >500 files. `repo_root` must be absolute and consistent
  between `build_index` and `query_index` — a mismatch causes silent empty results.
  **Implemented in `cli.py`:** after computing `total_files`, a lazy
  `from retrieval.chroma_index import LARGE_REPO_THRESHOLD` plus
  `if not use_index and total_files > LARGE_REPO_THRESHOLD: use_index = True` auto-enables
  indexing, with a `click.secho` warning so the (real, non-trivial) embedding cost isn't
  incurred silently. Explicit `--index` still works unchanged for smaller repos.

### Integration tests

Write in `tests/integration/test_full_query.py`. Use `tests/fixtures/sample_repo/`
as the fixture — a small, stable repo committed to the test directory. Do not test
against live external repositories.

Required integration tests:
- `test_session_log_written` — all required top-level keys present in log
- `test_navigation_log_nonempty` — `navigation.reads` has at least one entry
- `test_attribution_all_valid` — `attribution.invalid_count == 0` for a simple question
- `test_verify_appends_to_session` — `"faithfulness"` key present after `--verify`
- `test_path_traversal_rejected` — nav log has no entry for `../../etc/passwd` attempt
- `test_stability_jaccard_range` — `mean_jaccard` in `[0.0, 1.0]`, valid rating string

### README requirements

The README is a portfolio artifact. It must include:
- One-paragraph description of the tool and why it exists
- The architecture diagram from this file (verbatim)
- Install instructions: `git clone`, `pip install -r requirements.txt`, `.env` setup
- Usage examples for all six CLI modes: basic, `--verify`, `--stability`, `--output json`, `--index`, `--interactive`
- XAI features section: one paragraph per principle, what it does, what formal XAI
  concept it implements — written for a technical but non-specialist audience
- Connection to SHAP stability research: one paragraph explaining the Jaccard stability
  score as a methodological port from explanation stability analysis in the FAccT paper
- 90-second demo GIF recorded with `asciinema`, converted with `agg`

### Phase 5 validation checklist

```
[ ] pytest tests/unit/  -- minimum 20 test cases, all pass
[ ] pytest tests/integration/  -- all 6 tests pass
[ ] --stability --runs 3 on a real repo -- output matches stability schema
[ ] faithfulness_score always in [0.0, 1.0] across five --verify runs
[ ] README with architecture diagram and XAI documentation committed
[ ] Demo GIF recorded and embedded in README
[x] GitHub repository is public with clean commit history — verified: github.com/anoushka-kancherla/codebaseqna
    renders publicly (no login wall/"Private" label); `git log --oneline` shows a clean,
    phase-aligned sequence with no wip/noise commits
[ ] Session log for every integration test contains all required top-level keys
```

---

## Post-Phase-5 additions

Not part of the original five phases — added afterward as usability improvements on top
of the finished tool.

### `--interactive`

Added to remove the friction of restarting the MCP server for every single question
against the same repo.

- `--interactive` / `-i`: starts the local MCP server once, then loops on `input()` for
  multiple questions against the same repo. `--repo` is still required; `--question` is
  not. Rejected in combination with `--verify` or `--stability` (raises `UsageError`) —
  those stay one-shot modes.
- Each question is still an **independent, stateless Claude call** — no message history
  is threaded between turns, and each gets its own session id and its own
  `logs/{session_id}.json`. Only the MCP server process (and, with `--index`, the built
  ChromaDB index) persists across questions.
- **Critical:** `mcp_server.NAV_LOG` is a module-level instance that accumulates every
  read for as long as the server process is alive. One-shot invocations get a fresh
  instance for free (one process = one question), but the interactive loop must
  explicitly do `mcp_server.NAV_LOG = NavigationLog()` before every question — otherwise
  navigation entries and `coverage_pct` leak across questions in the same session.
- **`--index` interaction:** `build_index()` re-walks and re-embeds the whole repo on
  every call, so it must run once before the loop starts, not per question.
  `query_index()` runs fresh per question.
- Exits cleanly on `exit`, `quit`, empty input, Ctrl-D (`EOFError`), or Ctrl-C
  (`KeyboardInterrupt`) — no traceback.

### `--verbose`

Prints the full thinking block instead of truncating to 400 chars. Fixes a previously
dangling promise: the thinking panel always printed `[--verbose to expand]` even though
no such flag existed until this addition.

### `--list-sessions` / `--show-session <id>`

Browse `logs/*.json` from the CLI instead of grepping the directory by hand.
`--list-sessions` prints `session_id  created_at  question` for every saved session.
`--show-session <id>` reconstructs the same prose/confidence/files-read panels a live
query would print, sourced from the saved session dict rather than a `ParsedResponse` —
note the per-file `reason` strings aren't persisted in the session schema (only
`navigation.reads` paths are), so `--show-session` lists files read without the
original reasons Claude gave for reading them.

### Per-query cost estimate

Every query prints an approximate `$` cost after the answer, computed from
`response.usage` (`input_tokens`, `output_tokens`) against a static `PRICING_PER_MTOK`
table in `cli.py`, matched by substring against the model name (`opus`/`sonnet`/`haiku`).
This is a rough estimate for eyeballing cost before running `--stability`, not a billing
figure — the streaming parser doesn't reliably capture `input_tokens` in every code path
(see `api/stream_parser.py`), so treat it as approximate.

### Config defaults via `.env`

`CODEBASEQNA_REPO`, `CODEBASEQNA_MODEL`, and `CODEBASEQNA_MAX_FILES` env vars (loaded via
the existing `python-dotenv` dependency) override the `--repo`/`--model`/`--max-files`
CLI defaults, so a repo you query repeatedly doesn't require retyping those flags. CLI
flags still take precedence when passed explicitly.

### Shell completion

`main(prog_name="codebaseqna")` in the `if __name__ == "__main__":` block gives Click's
built-in completion a clean env var name (`_CODEBASEQNA_COMPLETE`) instead of the
dotted, invalid-as-an-env-var default Click would otherwise derive from `cli.py`. Enable
with `eval "$(_CODEBASEQNA_COMPLETE=bash_source python cli.py)"` (or `zsh_source`).

### ChromaDB auto-activation

Closed the gap between the Phase 5 spec ("activate only for repos >500 files") and the
implementation (previously `--index` was manual-only). See the ChromaDB section above
for the actual `cli.py` snippet — reuses `LARGE_REPO_THRESHOLD` from
`retrieval/chroma_index.py` rather than duplicating the constant, and warns via
`click.secho` before auto-enabling since the embedding pass has a real cost the user
didn't explicitly ask for.

### `--tunnel`

Automates the "deployment wrinkle" documented in the README: Claude's MCP connector
calls the local server from Anthropic's infrastructure, so any real query needs a public
HTTPS URL in front of it, previously a fully manual `ngrok http 8000` +
`export MCP_SERVER_URL=...` step.

- `_start_ngrok_tunnel(port)` / `_stop_ngrok_tunnel()` in `cli.py` wrap `pyngrok`, lazily
  imported (same pattern as the `chromadb` lazy imports) so invocations without
  `--tunnel` pay zero cost. `ngrok.connect(port, "http", bind_tls=True)` forces an
  HTTPS-only tunnel — a plain `http://` URL would fail the same way localhost does.
- **Precedence:** `--tunnel` always wins over a stale `MCP_SERVER_URL` env var when
  passed; without `--tunnel`, behavior is byte-for-byte unchanged (the env var still
  falls back to `http://127.0.0.1:{port}/mcp/`).
- **Critical:** the tunnel must be torn down even on error, or ngrok's agent process
  leaks past the CLI invocation and can hit ngrok's concurrent-tunnel limit on the next
  run. `main()`'s stability/interactive/single-question logic is wrapped in a
  `try`/`finally` that calls `_stop_ngrok_tunnel()` unconditionally when `--tunnel` was
  used — this covers every `return` path and any exception, not just the happy path.
- Optional `NGROK_AUTHTOKEN` env var (read via the existing `.env`/`load_dotenv()`
  mechanism) for machines without ngrok already configured; on a machine with ngrok
  pre-authenticated, `--tunnel` needs no extra setup.
- New dependency: `pyngrok` (added to `requirements.txt`).

---

## Session log schema

Every session writes to `logs/{session_id}.json`. Write this even in Phase 2 with
XAI fields as null — it avoids schema drift when Phase 4 adds the XAI fields.

```json
{
  "session_id": "uuid4",
  "created_at": "ISO 8601",
  "repo_path": "/absolute/path",
  "question": "user question",
  "model": "claude-sonnet-4-20250514",
  "flags": {"max_files": 8, "thinking_enabled": true, "verify": false},
  "navigation": {
    "total_files_in_repo": 312,
    "files_read": 4,
    "coverage_pct": 1.28,
    "reads": [{"path": "...", "timestamp": 0.0, "lines": 0, "size_bytes": 0}]
  },
  "attribution": {"citations": [...], "valid_count": 0, "invalid_count": 0},
  "confidence": {
    "level": "high|partial|low",
    "uncertainty_type": "epistemic|aleatoric|both|none",
    "call_graph_complete": true,
    "caveats": "..."
  },
  "answer": {
    "thinking_excerpt": "first 500 chars",
    "prose": "full prose answer",
    "not_found": null
  },
  "faithfulness": null,
  "api_usage": {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}
}
```

---

## Common failure modes

| Failure | Diagnosis and fix |
|---|---|
| Claude reads all files at once | `{max_files}` not injected — check `.format()` call. Add: "Stop reading files once you have sufficient evidence." |
| JSON header not parsed (`split_json_header` returns `{}`) | Claude used code fences. Test both fenced and raw patterns in `split_json_header`. |
| Navigation log empty after query | `NAV_LOG` in `mcp_server.py` is a different instance than the query handler references. Make it module-level, import by reference. |
| Path traversal test passes inadvertently | `.resolve()` not called on both sides. Both `resolved` and `ROOT.resolve()` must use `.resolve()` before `startswith()`. |
| Faithfulness score always 0.0 | Citations list is empty. Verify attribution runs before `run_faithfulness_check`. Check that prose is not empty in the session log. |
| Stability score always 1.0 | `per_run_files` entries are the same dict reference. Each `run_query` call must return an independent `json_header` dict. |
| `git_blame` crashes | Merge commit — file not in all parents. Wrap `repo.blame()` in `try/except`, return `[]` with an error note. |
| ChromaDB returns no results | `repo_root` differs between `build_index` and `query_index`. Use absolute paths, verify consistency. |
| Thinking panel empty | `budget_tokens` too low or thinking disabled. Set to 5000. Verify thinking blocks are separated from text blocks in `parse_stream`. |
| `faithfulness` key missing after `--verify` | `session_path` is relative, not absolute. Resolve before writing. |

---

## Key constants

```python
# Model
DEFAULT_MODEL   = "claude-sonnet-4-20250514"
MAX_TOKENS      = 8000
THINKING_BUDGET = 5000

# MCP server
MAX_FILE_BYTES  = 500_000
MAX_DEPTH       = 6
EXCLUDED_DIRS   = {".git", "node_modules", "__pycache__", "build", "dist", ".venv"}
EXCLUDED_EXTS   = {".pyc", ".pyo", ".class", ".o"}

# XAI
MAX_DIFF_BYTES  = 50_000
MAX_SYMBOL_HITS = 50
FAITHFULNESS_WARNING_THRESHOLD = 0.7
STABILITY_HIGH   = 0.7
STABILITY_MODERATE = 0.4

# ChromaDB
CHROMA_N_RESULTS    = 10
LARGE_REPO_THRESHOLD = 500   # files — activate ChromaDB above this
```

---

## Environment

```bash
# Install
pip install anthropic mcp gitpython click python-dotenv chromadb sentence-transformers pyngrok pytest

# .env
ANTHROPIC_API_KEY=sk-ant-...

# Run
python cli.py --repo /path/to/repo --question "where is auth handled?"
python cli.py --repo /path/to/repo --question "..." --verify
python cli.py --repo /path/to/repo --question "..." --stability --runs 3
python cli.py --repo /path/to/repo --question "..." --index
python cli.py --repo /path/to/repo --question "..." --output json
python cli.py --repo /path/to/repo --interactive
python cli.py --repo /path/to/repo --question "..." --verbose
python cli.py --list-sessions
python cli.py --show-session <session-id>
python cli.py --repo /path/to/repo --question "..." --tunnel

# Test
pytest tests/unit/
pytest tests/integration/

# MCP inspector (Phase 1 validation)
npm install -g @modelcontextprotocol/inspector
mcp-inspector server/mcp_server.py
```
