from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import uuid

from api.stream_parser import ParsedResponse
from server.navigation_log import NavigationLog

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


@dataclass
class SessionLog:
    session_id: str
    created_at: str
    repo_path: str
    question: str
    model: str
    flags: dict
    navigation: dict
    answer: dict
    attribution: dict | None = None
    confidence: dict | None = None
    faithfulness: dict | None = None
    api_usage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "repo_path": self.repo_path,
            "question": self.question,
            "model": self.model,
            "flags": self.flags,
            "navigation": self.navigation,
            "attribution": self.attribution,
            "confidence": self.confidence,
            "answer": self.answer,
            "faithfulness": self.faithfulness,
            "api_usage": self.api_usage,
        }

    def save(self) -> Path:
        LOGS_DIR.mkdir(exist_ok=True)
        path = LOGS_DIR / f"{self.session_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def build_session_log(
    repo_path: str,
    question: str,
    model: str,
    max_files: int,
    thinking_enabled: bool,
    nav_log: NavigationLog,
    total_files_in_repo: int,
    response: ParsedResponse,
) -> SessionLog:
    reads = nav_log.to_dict()
    files_read = sum(1 for e in reads if e["event"] == "file_read")
    coverage_pct = round(100 * files_read / total_files_in_repo, 2) if total_files_in_repo else 0.0

    return SessionLog(
        session_id=nav_log.session_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        repo_path=repo_path,
        question=question,
        model=model,
        flags={"max_files": max_files, "thinking_enabled": thinking_enabled, "verify": False},
        navigation={
            "total_files_in_repo": total_files_in_repo,
            "files_read": files_read,
            "coverage_pct": coverage_pct,
            "reads": reads,
        },
        answer={
            "thinking_excerpt": response.thinking[:500],
            "prose": response.prose,
            "not_found": response.json_header.get("result") if response.json_header.get("result") == "not_found" else None,
        },
        api_usage=response.usage,
    )
