from __future__ import annotations
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

CITATION_RE = re.compile(r"`([^`]+\.\w+):(\d+)(?:-(\d+))?`")


def check_citations(prose: str, root: Path) -> dict:
    root = root.resolve()
    citations = []
    for file_ref, start_str, end_str in CITATION_RE.findall(prose):
        start = int(start_str)
        end = int(end_str) if end_str else start
        citation = {"file": file_ref, "start": start, "end": end}

        resolved = (root / file_ref).resolve()
        if not str(resolved).startswith(str(root)) or not resolved.exists():
            citation["status"] = "INVALID_PATH"
            logger.warning("Invalid citation path: %s", file_ref)
            citations.append(citation)
            continue

        line_count = sum(1 for _ in resolved.open(encoding="utf-8", errors="replace"))
        if end > line_count:
            citation["status"] = "INVALID_RANGE"
            logger.warning("Invalid citation range: %s:%s-%s (file has %s lines)", file_ref, start, end, line_count)
            citations.append(citation)
            continue

        citation["status"] = "VALID"
        citations.append(citation)

    valid_count = sum(1 for c in citations if c["status"] == "VALID")
    return {"citations": citations, "valid_count": valid_count, "invalid_count": len(citations) - valid_count}
