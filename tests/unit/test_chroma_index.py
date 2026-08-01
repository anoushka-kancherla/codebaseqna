from retrieval.chroma_index import build_index, query_index


def test_build_and_query_index(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "auth.py").write_text(
        "import os\n\n"
        "def login(user, password):\n"
        "    return check_credentials(user, password)\n\n"
        "class SessionManager:\n"
        "    def create(self):\n"
        "        pass\n"
    )
    (repo / "unrelated.py").write_text("def bake_cake():\n    return 'cake'\n")

    index_path = str(tmp_path / ".chroma")
    build_index(str(repo), index_path=index_path)

    results = query_index("where does login happen?", index_path=index_path, n_results=3)
    assert len(results) > 0
    assert all({"file", "start", "content"} == set(r) for r in results)
    assert any("login" in r["content"] for r in results)


def test_query_empty_index_returns_empty_list(tmp_path):
    results = query_index("anything", index_path=str(tmp_path / ".chroma"), n_results=5)
    assert results == []


def test_chunk_boundaries_split_on_def_and_class():
    from pathlib import Path
    from retrieval.chroma_index import _chunk_file
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        f = root / "m.py"
        f.write_text("import os\n\ndef a():\n    pass\n\nclass B:\n    pass\n")
        chunks = _chunk_file(f, root)
        assert len(chunks) == 3
        assert chunks[0]["content"].startswith("import os")
        assert chunks[1]["content"].startswith("def a")
        assert chunks[2]["content"].startswith("class B")
