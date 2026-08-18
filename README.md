# codebaseqna

A CLI-first tool that answers natural-language questions about a local code repository —
"where is authentication handled?", "what changed in the last commit that caused this bug?"
— by giving Claude scoped, instrumented access to the repo through a custom MCP server and
returning grounded answers with exact `file:line` citations, a confidence rating, and a full
audit trail of every file it read. It exists because "ask an LLM about your codebase" tools
are easy to build and easy to not trust: the interesting engineering problem isn't getting an
answer, it's proving the answer is attributable, scoped to real evidence, and reproducible.
That's the XAI (explainable AI) layer, and it's the actual point of this project.

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

**A note on that diagram's "Resources" box**: `repo://tree` and `repo://file/{path}` are
still served as real MCP resources (Phase 1's validation checklist exercises them directly),
but a real end-to-end run against the live API showed that Anthropic's remote MCP connector
only exposes a server's *tools* to Claude as things it can call mid-conversation — resources
are a separate MCP primitive meant for the client app to attach, not something the model can
fetch on its own. Claude confirmed this directly ("I don't have a direct tool to fetch
repo://tree") and instead used `search_symbol` to get by, with zero files actually opened. So
the tree/file access Claude actually uses is exposed as two more tools, `list_tree` and
`read_file`, which wrap the same instrumented, path-safe functions the resources use — the
resources stay for direct MCP clients (like the inspector), the tools are what the model
itself calls.

Every file the MCP server serves is instrumented into `server/navigation_log.py` — that log
is the foundation of the whole XAI layer (the scope panel, the audit trail, the stability
score all read from it).

## Install

```bash
git clone <this-repo>
cd codebaseqna
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` (never commit it):

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

**One deployment wrinkle worth knowing up front:** Claude's MCP connector (the
`mcp_servers` field on the Messages API) calls your MCP server *from Anthropic's
infrastructure*, not from your machine. `cli.py` starts `server/mcp_server.py` locally
for you, but for a real (non-localhost-only) query you need to put a public HTTPS URL in
front of it — e.g. `ngrok http 8000` — and point the tool at it:

```bash
export MCP_SERVER_URL="https://<your-ngrok-subdomain>.ngrok.io/mcp/"
```

Without that, only local smoke-testing works (curling the server directly, or running the
test suite, which mocks the Anthropic call and drives the real local server instead).

## Usage

```bash
# Basic query
python cli.py --repo /path/to/repo --question "where is auth handled?"

# Structured output (for piping into other tools)
python cli.py --repo /path/to/repo --question "where is auth handled?" --output json

# Faithfulness check: re-verify a previous answer's citations against the actual files,
# using a second Claude call that reads straight off disk (not through MCP, so it can't
# pollute the original session's navigation log)
python cli.py --verify <session-id>

# Stability check: rerun the same question N times and measure how consistent the set of
# files Claude reads is, via pairwise Jaccard similarity. Costs ~N× a normal query.
python cli.py --repo /path/to/repo --question "where is auth handled?" --stability --runs 3

# ChromaDB-assisted retrieval for large repos (>500 files): chunks the repo by function/class
# boundary, embeds it, and prepends the top-k relevant chunks to the question before asking
python cli.py --repo /path/to/repo --question "where is auth handled?" --index

# Interactive mode: ask several questions against one repo without restarting the MCP
# server between them. Each question is still an independent, stateless Claude call.
python cli.py --repo /path/to/repo --interactive

# Full thinking block instead of the first 400 chars
python cli.py --repo /path/to/repo --question "where is auth handled?" --verbose

# Browse past sessions instead of asking a new question
python cli.py --list-sessions
python cli.py --show-session <session-id>
```

Every query also prints a rough per-query cost estimate (based on a static pricing table,
not live billing data) after the answer.

**Config defaults via `.env`:** set `CODEBASEQNA_REPO`, `CODEBASEQNA_MODEL`, or
`CODEBASEQNA_MAX_FILES` in `.env` to stop retyping `--repo`/`--model`/`--max-files` on a
repo you query repeatedly. CLI flags still override them.

**Shell completion:** `eval "$(_CODEBASEQNA_COMPLETE=bash_source python cli.py)"` (or
`zsh_source` for zsh) enables tab completion for flags.

## XAI features

Five principles, each independently testable, each doing one job:

**1. Attribution** (`xai/attribution.py`) — every factual claim in Claude's answer must cite
an exact `` `file.py:N-M` `` range. This module regexes those citations out of the prose and
validates each one against the real filesystem (exists, not a path-traversal attempt, line
range within the file's actual length), flagging anything else as `INVALID_PATH` or
`INVALID_RANGE`. This is the project's version of *grounding*: an answer isn't just fluent,
it's checked against the evidence it claims to rest on.

**2. Scope transparency** (the "Files read: N of M total" panel in `cli.py`) — every query
reports exactly which files were opened, why, and what fraction of the repo that represents.
This is the same idea as reporting a model's *evidence set* or *saliency scope* in extractive
QA: knowing what the model looked at is often as informative as the answer itself, especially
for catching an answer that's confidently wrong because it never read the relevant file.

**3. Negative results** (the not-found "Search report" panel in `cli.py`) — when Claude can't
find something, the tool doesn't let it guess. It reports what it checked, the closest match
it found, and suggested next steps instead of a fabricated answer. This is *selective
prediction* / abstention: a system that knows when to say "I don't know" is more trustworthy
than one that always answers.

**4. Confidence & uncertainty typing** (`xai/confidence.py`) — Claude self-reports a
confidence level and an uncertainty *type* (epistemic — "I don't have enough evidence" —
vs. aleatoric — "the answer is genuinely ambiguous" — vs. both vs. none). Because models are
bad at policing their own overconfidence, two overrides are enforced in code, not trusted
from the model's own output: uncertainty can't be "none" if the call graph wasn't fully
traced, and confidence can't be "high" if no files were actually read.

**5a. Faithfulness verification** (`xai/faithfulness.py`, `--verify`) — a second, independent
Claude call re-checks each cited claim against the real file content and scores it
VERIFIED / PARTIAL / UNSUPPORTED. This is a *faithfulness* metric in the explainability
sense: does the stated rationale actually match the evidence, or did the model cite a
plausible-looking line that doesn't actually support the claim?

**5b. Stability analysis** (`xai/stability.py`, `--stability`) — see below.

## Connection to explanation-stability research

`--stability` runs the same question N times and computes the pairwise Jaccard similarity
between the sets of files Claude reports reading across runs. This is a direct methodological
port from a well-known problem in the SHAP / feature-attribution literature (studied, among
other places, at the ACM FAccT conference): an explanation method that gives wildly different
"important features" on repeated runs of the same input isn't one you can trust, no matter how
plausible any single run looks. The same logic applies here — if Claude reads a different set
of files each time you ask the identical question, its citations are evidence of *a* reasoning
path, not *the* reasoning path, and any individual answer's confidence should be discounted
accordingly. Mean Jaccard ≥ 0.7 is rated "high" stability, ≥ 0.4 "moderate", below that "low".

## Demo

Not included in this repo — record one against a real repo with:

```bash
asciinema rec demo.cast
# ...run a few `python cli.py` queries...
agg demo.cast demo.gif
```

## Tests

```bash
pytest tests/unit/         # 68 cases across every module
pytest tests/integration/  # 6 cases against tests/fixtures/sample_repo/, Anthropic API mocked
```
