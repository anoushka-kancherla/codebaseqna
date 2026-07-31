from xai.attribution import check_citations


def test_valid_citation(tmp_path):
    (tmp_path / "a.py").write_text("line1\nline2\nline3\n")
    result = check_citations("See `a.py:1-2` for details.", tmp_path)
    assert result == {
        "citations": [{"file": "a.py", "start": 1, "end": 2, "status": "VALID"}],
        "valid_count": 1,
        "invalid_count": 0,
    }


def test_single_line_citation(tmp_path):
    (tmp_path / "a.py").write_text("line1\nline2\n")
    result = check_citations("See `a.py:1` for details.", tmp_path)
    assert result["citations"][0] == {"file": "a.py", "start": 1, "end": 1, "status": "VALID"}


def test_nonexistent_file_is_invalid_path(tmp_path):
    result = check_citations("See `missing.py:1-2` for details.", tmp_path)
    assert result["citations"][0]["status"] == "INVALID_PATH"
    assert result["invalid_count"] == 1


def test_path_traversal_is_invalid_path(tmp_path):
    outside = tmp_path.parent / "secret.py"
    outside.write_text("x\n")
    result = check_citations("See `../secret.py:1` for details.", tmp_path)
    assert result["citations"][0]["status"] == "INVALID_PATH"


def test_out_of_range_line_is_invalid_range(tmp_path):
    (tmp_path / "a.py").write_text("line1\nline2\n")
    result = check_citations("See `a.py:1-99` for details.", tmp_path)
    assert result["citations"][0]["status"] == "INVALID_RANGE"
    assert result["invalid_count"] == 1


def test_no_citations_returns_empty():
    result = check_citations("No citations here.", __import__("pathlib").Path("."))
    assert result == {"citations": [], "valid_count": 0, "invalid_count": 0}


def test_mixed_valid_and_invalid(tmp_path):
    (tmp_path / "a.py").write_text("line1\nline2\n")
    prose = "First `a.py:1-2` is fine, but `missing.py:5` is not."
    result = check_citations(prose, tmp_path)
    assert result["valid_count"] == 1
    assert result["invalid_count"] == 1
