from xai.confidence import apply_confidence_overrides


def test_passthrough_when_valid():
    header = {
        "confidence": "high",
        "uncertainty_type": "epistemic",
        "call_graph_complete": True,
        "files_read": ["a.py"],
        "caveats": "none",
    }
    result = apply_confidence_overrides(header)
    assert result == {
        "level": "high",
        "uncertainty_type": "epistemic",
        "call_graph_complete": True,
        "caveats": "none",
    }


def test_uncertainty_none_overridden_when_call_graph_incomplete():
    header = {"confidence": "partial", "uncertainty_type": "none", "call_graph_complete": False, "files_read": ["a.py"]}
    result = apply_confidence_overrides(header)
    assert result["uncertainty_type"] == "epistemic"


def test_uncertainty_none_kept_when_call_graph_complete():
    header = {"confidence": "partial", "uncertainty_type": "none", "call_graph_complete": True, "files_read": ["a.py"]}
    result = apply_confidence_overrides(header)
    assert result["uncertainty_type"] == "none"


def test_confidence_high_overridden_when_files_read_empty():
    header = {"confidence": "high", "uncertainty_type": "epistemic", "call_graph_complete": True, "files_read": []}
    result = apply_confidence_overrides(header)
    assert result["level"] == "low"


def test_confidence_high_kept_when_files_read_present():
    header = {"confidence": "high", "uncertainty_type": "epistemic", "call_graph_complete": True, "files_read": ["a.py"]}
    result = apply_confidence_overrides(header)
    assert result["level"] == "high"


def test_missing_files_read_key_treated_as_empty():
    header = {"confidence": "high", "uncertainty_type": "epistemic", "call_graph_complete": True}
    result = apply_confidence_overrides(header)
    assert result["level"] == "low"


def test_both_overrides_fire_independently():
    header = {"confidence": "high", "uncertainty_type": "none", "call_graph_complete": False, "files_read": []}
    result = apply_confidence_overrides(header)
    assert result["level"] == "low"
    assert result["uncertainty_type"] == "epistemic"
