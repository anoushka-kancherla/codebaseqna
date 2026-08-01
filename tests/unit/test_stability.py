from unittest.mock import patch

from api.stream_parser import ParsedResponse
from xai.stability import jaccard, run_stability_check


def test_jaccard_identical_sets():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_disjoint_sets():
    assert jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_both_empty():
    assert jaccard(set(), set()) == 1.0


def test_jaccard_partial_overlap():
    assert jaccard({"a", "b"}, {"b", "c"}) == 1 / 3


def _fake_response(files_read):
    return ParsedResponse(thinking="", json_header={"files_read": files_read}, prose="", tool_results=[], usage={})


def test_run_stability_check_high_when_identical():
    responses = [_fake_response(["a.py", "b.py"]) for _ in range(3)]
    with patch("xai.stability.run_query", side_effect=responses) as mock_query:
        result = run_stability_check("q", "/repo", runs=3)
    assert mock_query.call_count == 3
    assert result["mean_jaccard"] == 1.0
    assert result["rating"] == "high"
    assert result["consistent_files"] == ["a.py", "b.py"]
    assert result["inconsistent_files"] == []


def test_run_stability_check_low_when_disjoint():
    responses = [_fake_response(["a.py"]), _fake_response(["b.py"]), _fake_response(["c.py"])]
    with patch("xai.stability.run_query", side_effect=responses):
        result = run_stability_check("q", "/repo", runs=3)
    assert result["mean_jaccard"] == 0.0
    assert result["rating"] == "low"
    assert result["consistent_files"] == []
    assert sorted(result["inconsistent_files"]) == ["a.py", "b.py", "c.py"]


def test_run_stability_check_independent_dicts_not_shared_reference():
    # Regression guard for the documented failure mode: stability must not
    # read 1.0 just because per-run json_header dicts are the same object.
    responses = [_fake_response(["a.py"]), _fake_response(["a.py", "b.py"])]
    with patch("xai.stability.run_query", side_effect=responses):
        result = run_stability_check("q", "/repo", runs=2)
    assert result["per_run_files"][0] is not result["per_run_files"][1]
    assert result["mean_jaccard"] == 0.5


def test_mean_jaccard_in_valid_range():
    responses = [_fake_response(["a.py", "b.py"]), _fake_response(["b.py"]), _fake_response(["b.py", "c.py"])]
    with patch("xai.stability.run_query", side_effect=responses):
        result = run_stability_check("q", "/repo", runs=3)
    assert 0.0 <= result["mean_jaccard"] <= 1.0
    assert result["rating"] in {"high", "moderate", "low"}
