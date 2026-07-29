from __future__ import annotations
import logging
from pathlib import Path

import git

logger = logging.getLogger(__name__)

MAX_DIFF_CHARS = 50_000
MAX_SYMBOL_HITS = 50
SKIP_DIRS = {"node_modules", "__pycache__", ".git"}


def git_log(repo: git.Repo, n: int = 20) -> list[dict]:
    commits = []
    for commit in repo.iter_commits(max_count=n):
        commits.append({
            "hash": commit.hexsha,
            "author_name": commit.author.name,
            "author_email": commit.author.email,
            "date": commit.committed_datetime.isoformat(),
            "message": commit.message.strip(),
            "files_changed": list(commit.stats.files.keys()),
        })
    return commits


def git_diff(repo: git.Repo, commit_hash: str) -> str:
    try:
        commit = repo.commit(commit_hash)
    except (git.BadName, ValueError, git.GitCommandError):
        return f"[ERROR] unknown commit: {commit_hash}"
    diff = repo.git.show(commit.hexsha)
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + "\n[TRUNCATED]"
    return diff


def git_blame(repo: git.Repo, file_path: str, start_line: int, end_line: int) -> list[dict]:
    try:
        blame_data = repo.blame("HEAD", file_path)
    except Exception as exc:
        logger.warning("git_blame failed for %s: %s", file_path, exc)
        return []

    lines = []
    line_no = 0
    for commit, blame_lines in blame_data:
        for content in blame_lines:
            line_no += 1
            if start_line <= line_no <= end_line:
                lines.append({
                    "line_number": line_no,
                    "content": content,
                    "commit_hash": commit.hexsha,
                    "author_name": commit.author.name,
                    "date": commit.committed_datetime.isoformat(),
                })
    return lines


def search_symbol(root: Path, symbol: str, file_extensions: list[str] | None = None) -> list[dict]:
    results = []
    for path in sorted(root.rglob("*")):
        if len(results) >= MAX_SYMBOL_HITS:
            break
        if not path.is_file() or SKIP_DIRS & set(path.parts):
            continue
        if file_extensions and path.suffix not in file_extensions:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(lines):
            if symbol in line:
                results.append({
                    "file_path": str(path.relative_to(root)),
                    "line_number": i + 1,
                    "line_content": line,
                    "context_before": lines[max(0, i - 2):i],
                    "context_after": lines[i + 1:i + 3],
                })
                if len(results) >= MAX_SYMBOL_HITS:
                    break
    return results
