from __future__ import annotations
import itertools

from api.query import run_query

STABILITY_HIGH = 0.7
STABILITY_MODERATE = 0.4


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def run_stability_check(question: str, repo_path: str, runs: int = 3, **query_kwargs) -> dict:
    per_run_files = [
        set(run_query(repo_path=repo_path, question=question, **query_kwargs).json_header.get("files_read", []))
        for _ in range(runs)
    ]

    pairwise = [
        {"run_a": i, "run_b": j, "jaccard": jaccard(per_run_files[i], per_run_files[j])}
        for i, j in itertools.combinations(range(runs), 2)
    ]
    mean_jaccard = sum(p["jaccard"] for p in pairwise) / len(pairwise) if pairwise else 1.0

    if mean_jaccard >= STABILITY_HIGH:
        rating = "high"
    elif mean_jaccard >= STABILITY_MODERATE:
        rating = "moderate"
    else:
        rating = "low"

    consistent_files = sorted(set.intersection(*per_run_files)) if per_run_files else []
    all_files = set.union(*per_run_files) if per_run_files else set()
    inconsistent_files = sorted(all_files - set(consistent_files))

    return {
        "runs": runs,
        "per_run_files": [sorted(files) for files in per_run_files],
        "pairwise_jaccard": pairwise,
        "mean_jaccard": round(mean_jaccard, 4),
        "rating": rating,
        "consistent_files": consistent_files,
        "inconsistent_files": inconsistent_files,
    }
