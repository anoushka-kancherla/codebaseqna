from __future__ import annotations
import re
from pathlib import Path

import chromadb

CHROMA_N_RESULTS = 10
LARGE_REPO_THRESHOLD = 500
EXCLUDED_DIRS = {".git", "node_modules", "__pycache__", "build", "dist", ".venv"}

_BOUNDARY_RE = re.compile(r"^(?:def |class )", re.MULTILINE)


def _chunk_file(path: Path, root: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    starts = sorted({0, *(m.start() for m in _BOUNDARY_RE.finditer(text)), len(text)})

    chunks = []
    for start, end in zip(starts, starts[1:]):
        content = text[start:end]
        if not content.strip():
            continue
        start_line = text.count("\n", 0, start) + 1
        chunks.append({"file": str(path.relative_to(root)), "start": start_line, "content": content})
    return chunks


def _collection(index_path: str):
    client = chromadb.PersistentClient(path=index_path)
    return client.get_or_create_collection("codebase")


def build_index(repo_root: str, index_path: str = "./.chroma") -> None:
    root = Path(repo_root).resolve()
    collection = _collection(index_path)

    documents, metadatas, ids = [], [], []
    for path in root.rglob("*.py"):
        if EXCLUDED_DIRS & set(path.parts):
            continue
        for i, chunk in enumerate(_chunk_file(path, root)):
            documents.append(chunk["content"])
            metadatas.append({"file": chunk["file"], "start": chunk["start"]})
            ids.append(f"{chunk['file']}:{chunk['start']}:{i}")

    if documents:
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)


def query_index(question: str, index_path: str = "./.chroma", n_results: int = CHROMA_N_RESULTS) -> list[dict]:
    collection = _collection(index_path)
    if collection.count() == 0:
        return []
    results = collection.query(query_texts=[question], n_results=min(n_results, collection.count()))
    return [
        {"file": meta["file"], "start": meta["start"], "content": doc}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]
