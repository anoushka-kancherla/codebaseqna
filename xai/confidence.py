from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def apply_confidence_overrides(json_header: dict) -> dict:
    confidence = json_header.get("confidence")
    uncertainty_type = json_header.get("uncertainty_type")
    call_graph_ok = bool(json_header.get("call_graph_complete", False))
    files_read = json_header.get("files_read") or []

    # Enforced programmatically, not trusted from model output.
    if uncertainty_type == "none" and not call_graph_ok:
        logger.warning("Override: uncertainty_type none→epistemic")
        uncertainty_type = "epistemic"

    if confidence == "high" and not files_read:
        logger.warning("Override: confidence high→low (files_read empty)")
        confidence = "low"

    return {
        "level": confidence,
        "uncertainty_type": uncertainty_type,
        "call_graph_complete": call_graph_ok,
        "caveats": json_header.get("caveats", ""),
    }
