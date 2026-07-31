from __future__ import annotations
import json
from pathlib import Path

from anthropic import Anthropic

from qa_types.session import LOGS_DIR
from xai.attribution import check_citations

DEFAULT_MODEL = "claude-sonnet-4-20250514"
FAITHFULNESS_WARNING_THRESHOLD = 0.7

VERIFY_PROMPT_TEMPLATE = """You are verifying claims made about a codebase against the actual file contents.
For each claim below, decide: VERIFIED, PARTIAL, or UNSUPPORTED.

Claims:
{claims_json}

Return ONLY a JSON array, no prose, no code fences:
[{{"claim_index": 0, "claim_summary": "...", "verdict": "VERIFIED|PARTIAL|UNSUPPORTED", "evidence": "...", "explanation": "..."}}]
"""


def _read_claim_evidence(repo_path: Path, file_ref: str, start: int, end: int) -> str:
    # Read directly off disk, NOT through the MCP server, so verification
    # never touches the original session's navigation log.
    resolved = (repo_path / file_ref).resolve()
    lines = resolved.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[start - 1:end])


def _build_claims(session: dict) -> list[dict]:
    repo_path = Path(session["repo_path"])
    prose = session["answer"]["prose"]
    attribution = session.get("attribution") or check_citations(prose, repo_path)

    claims = []
    for citation in attribution["citations"]:
        if citation["status"] != "VALID":
            continue
        evidence = _read_claim_evidence(repo_path, citation["file"], citation["start"], citation["end"])
        claims.append({
            "claim_index": len(claims),
            "citation": f"{citation['file']}:{citation['start']}-{citation['end']}",
            "prose": prose,
            "evidence": evidence,
        })
    return claims


def run_faithfulness_check(session_id: str, model: str = DEFAULT_MODEL, client: Anthropic | None = None) -> dict:
    session_path = (LOGS_DIR / f"{session_id}.json").resolve()
    session = json.loads(session_path.read_text())

    claims = _build_claims(session)
    if not claims:
        result = {"faithfulness_score": 0.0, "claims": []}
    else:
        client = client or Anthropic()
        prompt = VERIFY_PROMPT_TEMPLATE.format(claims_json=json.dumps(claims, indent=2))
        message = client.messages.create(model=model, max_tokens=4000, messages=[{"role": "user", "content": prompt}])
        verdicts = json.loads(message.content[0].text)

        verified = sum(1 for v in verdicts if v["verdict"] == "VERIFIED")
        partial = sum(1 for v in verdicts if v["verdict"] == "PARTIAL")
        score = (verified + 0.5 * partial) / len(verdicts) if verdicts else 0.0
        result = {"faithfulness_score": round(score, 4), "claims": verdicts}

    session["faithfulness"] = result
    session_path.write_text(json.dumps(session, indent=2))
    return result
