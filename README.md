# codebaseqna

A CLI tool that answers plain-English questions about a local codebase (“where’s auth
handled?”, “what changed in the last commit that broke this?”) by handing Claude scoped,
instrumented access to the repo through a custom MCP server. You get an answer with exact
`file:line` citations, a confidence rating, and a full audit trail of every file it opened
to get there.

Plenty of tools will let an LLM answer questions about your code. Fewer of them let you
check its work. That's really the point of this project: not the Q&A part, which is easy,
but the explainability layer on top of it: proving an answer is grounded in real files,
scoped to what was actually read, and reproducible if you ask again.

## Architecture

```
User (CLI)
    │
    ▼
cli.py  (Click)
    │
    ▼
api/query.py  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────  Anthropic API
(query handler,                                                                                                                claude-sonnet-5
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

Quick note on that "Resources" box, because it took a real run against the live API to
figure out: `repo://tree` and `repo://file/{path}` are genuinely served as MCP resources
(the Phase 1 checklist hits them directly), but it turns out Anthropic's remote MCP
connector only hands the model *tools*, not resources. Resources are for a client app to
attach, not something Claude can fetch mid-conversation. First time I ran this for real,
Claude said as much ("I don't have a direct tool to fetch repo://tree") and fell back to
`search_symbol` instead, without opening a single file. So the tree/file access Claude
actually uses comes through two more tools, `list_tree` and `read_file`, which just wrap
the same instrumented functions the resources use. The resources are still there for
direct MCP clients like the inspector; the tools are what the model calls.

Every file the server hands out gets logged in `server/navigation_log.py`. That log is
what the whole XAI layer is built on: the scope panel, the audit trail, the stability
score, all of it reads from there.

## Install

```bash
git clone <this-repo>
cd codebaseqna
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then a `.env` (don't commit it):

```bash
ANTHROPIC_API_KEY=sk-ant-...

# Optional (see "one wrinkle" below)
NGROK_AUTHTOKEN=...

# Optional, if you're tired of retyping these for the same repo (CLI flags still win)
CODEBASEQNA_REPO=/path/to/repo
CODEBASEQNA_MODEL=claude-sonnet-5
CODEBASEQNA_MAX_FILES=8
```

**One wrinkle worth knowing before you run anything for real:** Claude's MCP connector
calls your server *from Anthropic's infrastructure*, not from your machine. `cli.py`
starts `server/mcp_server.py` locally, which is fine for local smoke-testing, but a real
query needs that server reachable from the outside, meaning a public HTTPS URL in front
of it. Easiest way: pass `--tunnel` and let the CLI spin up ngrok for you.

```bash
python cli.py --repo /path/to/repo --question "where is auth handled?" --tunnel
```

That needs ngrok installed and authenticated once (ngrok.com). If this machine hasn't
done that yet, drop `NGROK_AUTHTOKEN` into `.env` instead. The tunnel comes down on its
own when the command finishes, error or not.

Or skip the flag and do it by hand:

```bash
ngrok http 8000
export MCP_SERVER_URL="https://<your-ngrok-subdomain>.ngrok.io/mcp/"
```

Neither one? You're limited to local smoke-testing: curling the server directly, or
running the test suite, which mocks the Anthropic call and drives the real local server
underneath it.

## Usage

```bash
# Basic query
python cli.py --repo /path/to/repo --question "where is auth handled?"

# Structured output, for piping into other tools
python cli.py --repo /path/to/repo --question "where is auth handled?" --output json

# Faithfulness check: a second, independent Claude call re-reads the cited files straight
# off disk (not through MCP, so it can't pollute the original session's nav log) and scores
# whether the earlier answer's citations actually hold up
python cli.py --verify <session-id>

# Stability check: rerun the same question N times, measure how consistent the set of
# files Claude reads is across runs. Costs roughly N× a normal query, so it asks first.
python cli.py --repo /path/to/repo --question "where is auth handled?" --stability --runs 3

# ChromaDB-assisted retrieval: chunks the repo by function/class boundary, embeds it, and
# prepends the most relevant chunks to the question before asking. Kicks in on its own for
# repos over 500 files (you'll get a warning first, since embedding isn't free); pass
# --index explicitly if you want it on a smaller repo too.
python cli.py --repo /path/to/repo --question "where is auth handled?" --index

# Interactive mode: ask a few questions against one repo without restarting the MCP
# server for each one. Still stateless per question by default.
python cli.py --repo /path/to/repo --interactive

# Interactive mode, but with actual memory: Claude's earlier turns (reasoning included)
# get carried into each new question, so "what about the token refresh path?" builds on
# what it already found instead of starting cold.
python cli.py --repo /path/to/repo --interactive --memory

# See the whole thinking block instead of the first 400 characters
python cli.py --repo /path/to/repo --question "where is auth handled?" --verbose

# Look back at past sessions instead of asking something new
python cli.py --list-sessions
python cli.py --show-session <session-id>
```

Every query prints a rough per-query cost estimate after the answer: a ballpark from a
static pricing table, not a real billing number.

**Shell completion:** `eval "$(_CODEBASEQNA_COMPLETE=bash_source python cli.py)"` (swap in
`zsh_source` for zsh).

That's not the full flag list. `--model`, `--max-files`, `--port`, `--runs`, and
`-y`/`--yes` (skips the `--stability` confirmation prompt) didn't get their own examples
above. `python cli.py --help` has everything.

## XAI features

Five principles here, each one testable on its own and doing exactly one job.

**Attribution** (`xai/attribution.py`) is the grounding check: every factual claim in an
answer has to cite an exact `` `file.py:N-M` `` range, and this module pulls those
citations out of the prose with a regex and checks each one against the real filesystem:
does the file exist, is it actually inside the repo, does the line range fit. Anything
that doesn't check out gets flagged `INVALID_PATH` or `INVALID_RANGE` rather than quietly
trusted.

Then there's scope transparency, which is really just the "Files read: N of M total"
line `cli.py` prints after every answer. It sounds minor, but it's often the most useful
signal in the whole output: knowing what the model actually looked at tells you a lot
about whether to trust the answer, especially when it's confidently wrong because it never
opened the one file that mattered.

When Claude can't find something, the tool doesn't let it improvise. The not-found path
prints what it checked, the closest match it found, and a few suggested next steps instead
of a plausible-sounding guess. Knowing when to say "I don't know" is worth more than always
having an answer.

Confidence gets a bit more scrutiny than just trusting whatever Claude reports.
`xai/confidence.py` has Claude self-rate both a confidence level and an uncertainty *type*:
epistemic ("not enough evidence yet") versus aleatoric ("this is genuinely ambiguous")
versus both versus neither. Models aren't great at policing their own overconfidence
though, so two checks run in code regardless of what the model claims: uncertainty can't
be "none" if the call graph was never fully traced, and confidence can't be "high" if no
files were actually read.

`--verify` runs a fifth check after the fact: a second Claude call re-reads the cited
files and scores each claim VERIFIED, PARTIAL, or UNSUPPORTED. It's asking a slightly
different question than attribution does: not just "does this file:line exist" but "does
it actually say what the answer claims it says."

And then stability, which gets its own section below because it's the one with an actual
research citation behind it.

## Connection to explanation-stability research

`--stability` asks the same question N times and computes pairwise Jaccard similarity
across the sets of files Claude says it read each run. This is lifted pretty directly from
a known problem in the SHAP / feature-attribution world (the ACM FAccT crowd has written
about it too): if an explanation method points at wildly different "important features"
every time you rerun it on the same input, you can't really trust any single run's
explanation, however convincing it looks in isolation. Same logic here: if Claude reads a
different set of files each time for the identical question, its citations are evidence of
*a* path through the reasoning, not *the* path, and that should knock down how much you
trust any one answer. Mean Jaccard ≥ 0.7 gets called "high" stability, ≥ 0.4 "moderate",
anything below that "low."

## Demo

![demo](demo.gif)

Recorded with [asciinema](https://asciinema.org) and converted with
[agg](https://github.com/asciinema/agg):

```bash
asciinema rec --cols 120 --rows 50 demo.cast
# ...run a few `python cli.py` queries...
agg demo.cast demo.gif
```

## Tests

```bash
pytest tests/unit/         # 100 cases across every module
pytest tests/integration/  # 6 cases against tests/fixtures/sample_repo/, Anthropic API mocked
```