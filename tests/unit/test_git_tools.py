from pathlib import Path

import git
import pytest

from server import git_tools

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def repo():
    return git.Repo(REPO_ROOT)


def test_git_log_returns_commit_fields(repo):
    commits = git_tools.git_log(repo, n=3)
    assert len(commits) <= 3
    assert len(commits) > 0
    first = commits[0]
    assert set(first) == {"hash", "author_name", "author_email", "date", "message", "files_changed"}
    assert isinstance(first["files_changed"], list)


def test_git_diff_returns_patch_text(repo):
    commit_hash = next(repo.iter_commits(max_count=1)).hexsha
    diff = git_tools.git_diff(repo, commit_hash)
    assert commit_hash[:7] in diff or "diff --git" in diff


def test_git_diff_unknown_commit_returns_error(repo):
    diff = git_tools.git_diff(repo, "not-a-real-commit")
    assert diff.startswith("[ERROR]")


def test_git_diff_truncates(monkeypatch, repo):
    monkeypatch.setattr(git_tools, "MAX_DIFF_CHARS", 20)
    commit_hash = next(repo.iter_commits(max_count=1)).hexsha
    diff = git_tools.git_diff(repo, commit_hash)
    assert diff.endswith("[TRUNCATED]")


def test_git_blame_returns_line_range(repo):
    result = git_tools.git_blame(repo, "README.md", 1, 1)
    assert len(result) == 1
    entry = result[0]
    assert set(entry) == {"line_number", "content", "commit_hash", "author_name", "date"}
    assert entry["line_number"] == 1


def test_git_blame_returns_empty_on_failure(monkeypatch, repo):
    def boom(*args, **kwargs):
        raise Exception("simulated merge-commit blame failure")

    monkeypatch.setattr(repo, "blame", boom)
    assert git_tools.git_blame(repo, "README.md", 1, 5) == []


def test_search_symbol_finds_known_string():
    results = git_tools.search_symbol(REPO_ROOT, "NavigationLog", file_extensions=[".py"])
    assert len(results) > 0
    hit = results[0]
    assert set(hit) == {"file_path", "line_number", "line_content", "context_before", "context_after"}
    assert "NavigationLog" in hit["line_content"]


def test_search_symbol_skips_excluded_dirs(tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.py").write_text("needle\n")
    (tmp_path / "real.py").write_text("needle\n")

    results = git_tools.search_symbol(tmp_path, "needle")
    assert len(results) == 1
    assert results[0]["file_path"] == "real.py"


def test_search_symbol_caps_at_max_hits(tmp_path, monkeypatch):
    monkeypatch.setattr(git_tools, "MAX_SYMBOL_HITS", 3)
    for i in range(10):
        (tmp_path / f"f{i}.py").write_text("needle\n")
    results = git_tools.search_symbol(tmp_path, "needle")
    assert len(results) == 3
