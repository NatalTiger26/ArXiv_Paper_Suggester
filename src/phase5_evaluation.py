#!/usr/bin/env python3
"""
Phase 5 — Evaluation of the Phase 4 paper-ranking system.

Phase 5 evaluates the Phase 4 multi-signal ranker against a fixed personal
ground-truth ranking from:

    rankings/personal_importance.csv

The evaluation includes:

    - Spearman rank correlation against personal 1–5 importance scores
    - Citation-only baseline using the SAME retrieval pool
    - Embedding-only baseline
    - Full Phase 4 ranker
    - Optional full + LLM ranker
    - Four signal ablations:
        * ablate_embedding
        * ablate_recency
        * ablate_citation
        * ablate_author
    - NDCG@10 and NDCG@20
    - Markdown + JSON results
    - Qualitative error analysis
    - Automatic conclusion sentence

Examples:

    python src/phase5_evaluation.py run

    python src/phase5_evaluation.py run --llm

    python src/phase5_evaluation.py run \
        --query "mechanistic interpretability circuits random matrix loss surfaces"

    python src/phase5_evaluation.py run -k 40

    python src/phase5_evaluation.py errors --top 12

The evaluation intentionally uses a retrieval pool larger than the final
evaluation cutoff. By default k=40, which is intended to maximize overlap
with the personal ground-truth list.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PERSONAL_RANKINGS_PATH = PROJECT_ROOT / "rankings" / "personal_importance.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_TABLE_PATH = RESULTS_DIR / "phase5_results_table.md"
RESULTS_JSON_PATH = RESULTS_DIR / "phase5_results.json"
ERROR_ANALYSIS_PATH = RESULTS_DIR / "phase5_error_analysis.md"


# ---------------------------------------------------------------------------
# Phase 4 imports
# ---------------------------------------------------------------------------

try:
    from phase4_scoring import (
        DEFAULT_HALF_LIFE,
        DEFAULT_POOL_MULTIPLIER,
        DEFAULT_TEMPERATURE,
        DEFAULT_WEIGHTS,
        score_and_rank,
    )
except ImportError:
    from src.phase4_scoring import (
        DEFAULT_HALF_LIFE,
        DEFAULT_POOL_MULTIPLIER,
        DEFAULT_TEMPERATURE,
        DEFAULT_WEIGHTS,
        score_and_rank,
    )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_QUERY = (
    "mechanistic interpretability circuits "
    "random matrix theory neural networks "
    "loss landscapes statistical mechanics "
    "representation learning world models"
)

DEFAULT_K = 40
DEFAULT_TOP_ERRORS = 12

# Evaluation cutoffs requested by the Phase 5 specification.
NDCG_CUTOFFS = (10, 20)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float | None = None) -> float | None:
    """Convert a value to float without raising."""
    if value is None:
        return default

    if isinstance(value, str):
        value = value.strip()

    if value == "":
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int | None = None) -> int | None:
    """Convert a value to int without raising."""
    if value is None:
        return default

    if isinstance(value, str):
        value = value.strip()

    if value == "":
        return default

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _normalize_text(value: Any) -> str:
    """Normalize arbitrary text for lightweight matching."""
    if value is None:
        return ""

    return " ".join(str(value).strip().lower().split())


def _normalize_id(value: Any) -> str:
    """
    Normalize paper identifiers.

    Handles common variations such as:

        2304.14997
        arXiv:2304.14997
        https://arxiv.org/abs/2304.14997
    """
    if value is None:
        return ""

    text = str(value).strip()

    if not text:
        return ""

    lowered = text.lower()

    prefixes = (
        "https://arxiv.org/abs/",
        "http://arxiv.org/abs/",
        "https://arxiv.org/pdf/",
        "http://arxiv.org/pdf/",
        "arxiv:",
    )

    for prefix in prefixes:
        if lowered.startswith(prefix):
            text = text[len(prefix):]
            break

    if text.lower().endswith(".pdf"):
        text = text[:-4]

    return text.strip()


def _first_present(row: dict[str, Any], names: Iterable[str]) -> Any:
    """
    Return the first non-empty field from a list of possible column names.
    """
    lowered = {
        str(key).strip().lower(): value
        for key, value in row.items()
    }

    for name in names:
        value = lowered.get(name.lower())
        if value is not None and str(value).strip() != "":
            return value

    return None


# ---------------------------------------------------------------------------
# Personal ground truth
# ---------------------------------------------------------------------------

def load_personal_rankings(
    path: str | Path = PERSONAL_RANKINGS_PATH,
) -> list[dict[str, Any]]:
    """
    Load the fixed Phase 2 personal-importance CSV.

    Expected conceptual schema:

        paper identifier + personal importance score from 1 to 5

    The implementation accepts several common column names so that small
    naming differences in Phase 2 do not break Phase 5.

    Recognized ID columns include:

        paper_id
        arxiv_id
        arxiv
        id
        doi

    Recognized personal-score columns include:

        importance
        personal_importance
        personal_score
        score
        rating
        relevance
        importance_score

    A rank column is also loaded if present, but the 1–5 personal importance
    score is the ground-truth relevance used for metrics.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Personal ranking file not found:\n"
            f"  {path}\n\n"
            "Finish Phase 2 `finalize` first so that "
            "`rankings/personal_importance.csv` exists."
        )

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(
                f"{path} does not contain a CSV header."
            )

        rows = list(reader)

    if not rows:
        raise ValueError(
            f"{path} is empty."
        )

    personal: list[dict[str, Any]] = []

    id_columns = (
        "paper_id",
        "arxiv_id",
        "arxiv",
        "id",
        "doi",
        "identifier",
        "paper",
    )

    score_columns = (
        "score_1_to_5",
        "personal_importance",
        "importance",
        "importance_score",
        "personal_score",
        "rating",
        "relevance",
        "score",
    )

    rank_columns = (
        "personal_rank",
        "rank",
        "ranking",
        "position",
    )

    title_columns = (
        "title",
        "paper_title",
        "name",
    )

    for row_number, row in enumerate(rows, start=2):
        raw_id = _first_present(row, id_columns)
        raw_score = _first_present(row, score_columns)
        raw_rank = _first_present(row, rank_columns)
        raw_title = _first_present(row, title_columns)

        paper_identifier = _normalize_id(raw_id)
        score = _safe_float(raw_score)
        rank = _safe_int(raw_rank)

        if not paper_identifier:
            # Skip completely blank rows.
            if not any(str(value).strip() for value in row.values()):
                continue

            raise ValueError(
                f"{path}: row {row_number} has no recognizable paper ID. "
                f"Expected one of: {', '.join(id_columns)}"
            )

        if score is None:
            raise ValueError(
                f"{path}: row {row_number} ({paper_identifier}) has no "
                f"recognizable personal 1–5 importance score. "
                f"Expected one of: {', '.join(score_columns)}"
            )

        if score < 1.0 or score > 5.0:
            raise ValueError(
                f"{path}: row {row_number} ({paper_identifier}) has "
                f"personal importance {score}. Expected a value from 1 to 5."
            )

        personal.append(
            {
                "paper_id": paper_identifier,
                "personal_score": float(score),
                "personal_rank": rank,
                "title": str(raw_title or "").strip(),
            }
        )

    if not personal:
        raise ValueError(
            f"No valid personal rankings were loaded from {path}."
        )

    # Prevent ambiguous ground truth.
    seen: dict[str, int] = {}

    for item in personal:
        pid = item["paper_id"]
        seen[pid] = seen.get(pid, 0) + 1

    duplicates = [
        pid
        for pid, count in seen.items()
        if count > 1
    ]

    if duplicates:
        raise ValueError(
            "Duplicate paper identifiers found in personal rankings:\n"
            + "\n".join(f"  - {pid}" for pid in duplicates[:20])
        )

    # If an explicit personal rank exists, use it for deterministic ordering.
    if all(item["personal_rank"] is not None for item in personal):
        personal.sort(
            key=lambda item: (
                item["personal_rank"],
                -item["personal_score"],
            )
        )
    else:
        # Otherwise highest personal importance first.
        personal.sort(
            key=lambda item: item["personal_score"],
            reverse=True,
        )

    return personal


# ---------------------------------------------------------------------------
# Ranking / metric utilities
# ---------------------------------------------------------------------------

def _average_ranks(values: list[float]) -> list[float]:
    """
    Assign competition-independent average ranks for ties.

    Example:

        [10, 10, 5] -> [1.5, 1.5, 3.0]
    """
    indexed = sorted(
        enumerate(values),
        key=lambda item: item[1],
    )

    ranks = [0.0] * len(values)

    i = 0

    while i < len(indexed):
        j = i + 1

        while (
            j < len(indexed)
            and indexed[j][1] == indexed[i][1]
        ):
            j += 1

        average_rank = (i + 1 + j) / 2.0

        for position in range(i, j):
            original_index = indexed[position][0]
            ranks[original_index] = average_rank

        i = j

    return ranks


def spearman_rho(
    personal_scores: Iterable[float],
    system_scores: Iterable[float],
) -> float | None:
    """
    Compute Spearman's rho with tie-aware average ranks.

    Returns None when fewer than two observations are available or when
    either ranking has zero variance.
    """
    personal = [float(value) for value in personal_scores]
    system = [float(value) for value in system_scores]

    if len(personal) != len(system):
        raise ValueError(
            "Spearman inputs must have equal length."
        )

    if len(personal) < 2:
        return None

    personal_rank = _average_ranks(personal)
    system_rank = _average_ranks(system)

    mean_personal = sum(personal_rank) / len(personal_rank)
    mean_system = sum(system_rank) / len(system_rank)

    numerator = sum(
        (a - mean_personal) * (b - mean_system)
        for a, b in zip(personal_rank, system_rank)
    )

    denominator_personal = math.sqrt(
        sum(
            (a - mean_personal) ** 2
            for a in personal_rank
        )
    )

    denominator_system = math.sqrt(
        sum(
            (b - mean_system) ** 2
            for b in system_rank
        )
    )

    denominator = denominator_personal * denominator_system

    if denominator == 0.0:
        return None

    return numerator / denominator


def _relevance_from_personal_score(score: float) -> float:
    """
    Convert a 1–5 personal importance score into graded relevance.

    The original 1–5 values are already valid graded relevance levels, so
    they are retained directly for NDCG.
    """
    return float(score)


def ndcg_at_k(
    ranked_papers: list[dict[str, Any]],
    personal_by_id: dict[str, dict[str, Any]],
    k: int,
) -> float:
    """
    Compute NDCG@k using personal 1–5 scores as graded relevance.

    Only papers with personal ground-truth labels contribute relevance.
    Missing papers receive relevance zero.

    The ideal ranking is constructed from all available personal scores,
    which is the appropriate ideal DCG for this fixed ground truth.
    """
    k = max(1, int(k))

    ranked_relevance: list[float] = []

    for paper in ranked_papers[:k]:
        pid = _normalize_id(
            paper.get("arxiv_id")
            or paper.get("paper_id")
        )

        item = personal_by_id.get(pid)

        if item is None:
            ranked_relevance.append(0.0)
        else:
            ranked_relevance.append(
                _relevance_from_personal_score(
                    item["personal_score"]
                )
            )

    dcg = 0.0

    for index, relevance in enumerate(ranked_relevance):
        position = index + 1
        dcg += (
            (2.0 ** relevance - 1.0)
            / math.log2(position + 1.0)
        )

    ideal_relevance = sorted(
        (
            _relevance_from_personal_score(
                item["personal_score"]
            )
            for item in personal_by_id.values()
        ),
        reverse=True,
    )[:k]

    ideal_dcg = 0.0

    for index, relevance in enumerate(ideal_relevance):
        position = index + 1
        ideal_dcg += (
            (2.0 ** relevance - 1.0)
            / math.log2(position + 1.0)
        )

    if ideal_dcg == 0.0:
        return 0.0

    return dcg / ideal_dcg


# ---------------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------------

def _copy_weights(weights: dict[str, float]) -> dict[str, float]:
    """Return a clean copy of a Phase 4 weight dictionary."""
    return {
        "embedding": float(weights["embedding"]),
        "recency": float(weights["recency"]),
        "citation": float(weights["citation"]),
        "author": float(weights["author"]),
    }


def ablation_weights(
    signal: str,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Zero one signal and renormalize the remaining weights.

    Example:

        original = 0.50, 0.20, 0.25, 0.05

        ablate_embedding:
            0.00, 0.20, 0.25, 0.05
            -> normalized to
            0.00, 0.80, 1.00, 0.20

    The absolute scale does not matter after normalization because Phase 4
    uses a weighted linear combination of signals.
    """
    valid_signals = {
        "embedding",
        "recency",
        "citation",
        "author",
    }

    if signal not in valid_signals:
        raise ValueError(
            f"Unknown ablation signal {signal!r}. "
            f"Expected one of: {', '.join(sorted(valid_signals))}"
        )

    base = _copy_weights(weights or DEFAULT_WEIGHTS)

    base[signal] = 0.0

    total = sum(base.values())

    if total <= 0.0:
        raise ValueError(
            f"Cannot ablate {signal}: all remaining weights are zero."
        )

    return {
        name: value / total
        for name, value in base.items()
    }


def rank_with_weights(
    query: str,
    weights: dict[str, float],
    k: int = DEFAULT_K,
    half_life: float = DEFAULT_HALF_LIFE,
    pool_size: int | None = None,
) -> list[dict[str, Any]]:
    """
    Run Phase 4 using a specified set of four core signal weights.

    Temperature is deliberately zero for evaluation so that Phase 5 measures
    the actual score ordering rather than adding diversity-based reordering.
    """
    if pool_size is None:
        pool_size = max(
            DEFAULT_K,
            k,
            3 * k,
            DEFAULT_POOL_MULTIPLIER * k,
        )

    return score_and_rank(
        query=query,
        k=k,
        weights=weights,
        half_life=half_life,
        temperature=0.0,
        use_llm=False,
        pool_size=pool_size,
    )


def rank_citation_baseline(
    query: str,
    k: int = DEFAULT_K,
    half_life: float = DEFAULT_HALF_LIFE,
    pool_size: int | None = None,
) -> list[dict[str, Any]]:
    """
    Citation-only baseline.

    IMPORTANT:
    This deliberately uses Phase 4's retrieval pool and then applies only
    the citation signal. Therefore it is a fair "citation attention" baseline
    rather than a ranking over the entire corpus.
    """
    weights = {
        "embedding": 0.0,
        "recency": 0.0,
        "citation": 1.0,
        "author": 0.0,
    }

    return rank_with_weights(
        query=query,
        weights=weights,
        k=k,
        half_life=half_life,
        pool_size=pool_size,
    )


def rank_embedding_only(
    query: str,
    k: int = DEFAULT_K,
    half_life: float = DEFAULT_HALF_LIFE,
    pool_size: int | None = None,
) -> list[dict[str, Any]]:
    """Embedding-only baseline using the Phase 3 retrieval signal."""
    weights = {
        "embedding": 1.0,
        "recency": 0.0,
        "citation": 0.0,
        "author": 0.0,
    }

    return rank_with_weights(
        query=query,
        weights=weights,
        k=k,
        half_life=half_life,
        pool_size=pool_size,
    )


# ---------------------------------------------------------------------------
# Ground-truth matching
# ---------------------------------------------------------------------------

def _build_personal_lookup(
    personal: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Index personal ground truth by normalized paper identifier."""
    lookup: dict[str, dict[str, Any]] = {}

    for item in personal:
        pid = _normalize_id(item["paper_id"])
        lookup[pid] = item

    return lookup


def _build_rank_lookup(
    ranked: list[dict[str, Any]],
) -> dict[str, int]:
    """
    Map paper IDs to one-based system rank.
    """
    result: dict[str, int] = {}

    for index, paper in enumerate(ranked, start=1):
        pid = _normalize_id(
            paper.get("arxiv_id")
            or paper.get("paper_id")
        )

        if pid:
            result[pid] = index

    return result


def _intersection(
    ranked: list[dict[str, Any]],
    personal_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Return the personal papers appearing in the system ranking.

    Ordering follows the system ranking.
    """
    intersection: list[dict[str, Any]] = []

    for paper in ranked:
        pid = _normalize_id(
            paper.get("arxiv_id")
            or paper.get("paper_id")
        )

        if pid in personal_by_id:
            intersection.append(
                {
                    "paper": paper,
                    "personal": personal_by_id[pid],
                }
            )

    return intersection


def _system_final_score(paper: dict[str, Any]) -> float:
    """Extract Phase 4 final score."""
    value = _safe_float(paper.get("final_score"), 0.0)
    return float(value or 0.0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate_ranking(
    ranked: list[dict[str, Any]],
    personal_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Calculate all requested metrics for one ranking.
    """
    overlap = _intersection(
        ranked,
        personal_by_id,
    )

    personal_scores = [
        float(item["personal"]["personal_score"])
        for item in overlap
    ]

    system_scores = [
        _system_final_score(item["paper"])
        for item in overlap
    ]

    rho = spearman_rho(
        personal_scores,
        system_scores,
    )

    return {
        "spearman_rho": rho,
        "ndcg_at_10": ndcg_at_k(
            ranked,
            personal_by_id,
            10,
        ),
        "ndcg_at_20": ndcg_at_k(
            ranked,
            personal_by_id,
            20,
        ),
        "system_count": len(ranked),
        "ground_truth_count": len(personal_by_id),
        "intersection_count": len(overlap),
    }


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def _personal_rank_map(
    personal: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Build a rank map from personal 1–5 scores.

    Higher personal importance = better personal rank.

    Ties receive the same average rank.
    """
    ids = [
        _normalize_id(item["paper_id"])
        for item in personal
    ]

    scores = [
        float(item["personal_score"])
        for item in personal
    ]

    # Average rank in ascending order, then reverse so highest score is rank 1.
    ascending_ranks = _average_ranks(scores)

    max_rank = float(len(scores))

    # For score ties, average ascending ranks need to be mirrored.
    personal_ranks = {
        pid: max_rank - ascending_rank + 1.0
        for pid, ascending_rank in zip(ids, ascending_ranks)
    }

    return personal_ranks


def _make_error_rows(
    ranked: list[dict[str, Any]],
    personal: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Compare system rank against personal rank.

    Positive rank_gap:
        system rank is numerically larger -> system UNDERRANKED the paper.

    Negative rank_gap:
        system rank is numerically smaller -> system OVERRANKED the paper.
    """
    personal_by_id = _build_personal_lookup(personal)
    personal_ranks = _personal_rank_map(personal)

    system_ranks = _build_rank_lookup(ranked)

    rows: list[dict[str, Any]] = []

    for pid, system_rank in system_ranks.items():
        if pid not in personal_by_id:
            continue

        personal_item = personal_by_id[pid]
        personal_rank = personal_ranks[pid]

        rank_gap = float(system_rank) - personal_rank

        paper = next(
            (
                item
                for item in ranked
                if _normalize_id(
                    item.get("arxiv_id")
                    or item.get("paper_id")
                ) == pid
            ),
            {},
        )

        rows.append(
            {
                "paper_id": pid,
                "title": paper.get("title", ""),
                "system_rank": system_rank,
                "personal_rank": personal_rank,
                "rank_gap": rank_gap,
                "personal_score": personal_item["personal_score"],
                "final_score": _system_final_score(paper),
                "citation_count": paper.get("citation_count", ""),
            }
        )

    return rows


def write_error_analysis(
    ranked: list[dict[str, Any]],
    personal: list[dict[str, Any]],
    path: str | Path = ERROR_ANALYSIS_PATH,
    top_n: int = DEFAULT_TOP_ERRORS,
    query: str = DEFAULT_QUERY,
) -> None:
    """
    Write qualitative underrank / overrank analysis.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = _make_error_rows(
        ranked,
        personal,
    )

    underranked = sorted(
        rows,
        key=lambda item: item["rank_gap"],
        reverse=True,
    )[:top_n]

    overranked = sorted(
        rows,
        key=lambda item: item["rank_gap"],
    )[:top_n]

    lines: list[str] = []

    lines.append("# Phase 5 Error Analysis")
    lines.append("")
    lines.append(f"**Query:** {query}")
    lines.append("")
    lines.append(
        "Rank gap is defined as `system rank - personal rank`."
    )
    lines.append(
        "Positive values indicate papers that the system **underranked** "
        "relative to the personal ranking."
    )
    lines.append(
        "Negative values indicate papers that the system **overranked** "
        "relative to the personal ranking."
    )
    lines.append("")

    lines.append("## Biggest Underranks")
    lines.append("")
    lines.append(
        "| Paper | Personal score | Personal rank | "
        "System rank | Gap | Final score | Citations |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|"
    )

    if underranked:
        for item in underranked:
            title = str(item["title"]).replace("|", "\\|")
            lines.append(
                f"| {title} | "
                f"{item['personal_score']:.1f} | "
                f"{item['personal_rank']:.1f} | "
                f"{item['system_rank']} | "
                f"+{item['rank_gap']:.1f} | "
                f"{item['final_score']:.4f} | "
                f"{item['citation_count'] or '—'} |"
            )
    else:
        lines.append("| None | — | — | — | — | — | — |")

    lines.append("")
    lines.append("## Biggest Overranks")
    lines.append("")
    lines.append(
        "| Paper | Personal score | Personal rank | "
        "System rank | Gap | Final score | Citations |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|"
    )

    if overranked:
        for item in overranked:
            title = str(item["title"]).replace("|", "\\|")
            lines.append(
                f"| {title} | "
                f"{item['personal_score']:.1f} | "
                f"{item['personal_rank']:.1f} | "
                f"{item['system_rank']} | "
                f"{item['rank_gap']:.1f} | "
                f"{item['final_score']:.4f} | "
                f"{item['citation_count'] or '—'} |"
            )
    else:
        lines.append("| None | — | — | — | — | — | — |")

    lines.append("")
    lines.append("## Interpretation Notes")
    lines.append("")
    lines.append(
        "- **Underrank:** a paper the personal ranking considers important "
        "but the system placed relatively low."
    )
    lines.append(
        "- **Overrank:** a paper the system placed relatively high despite "
        "a lower personal importance score."
    )
    lines.append(
        "- These disagreements are useful for diagnosing whether the "
        "embedding, recency, citation, or author signals are misaligned "
        "with personal research interests."
    )
    lines.append("")

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Evaluation suite
# ---------------------------------------------------------------------------

def _metric_row(
    name: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Add a human-readable name to a metric dictionary."""
    return {
        "method": name,
        **metrics,
    }


def run_evaluation(
    query: str = DEFAULT_QUERY,
    k: int = DEFAULT_K,
    half_life: float = DEFAULT_HALF_LIFE,
    pool_size: int | None = None,
    use_llm: bool = False,
) -> dict[str, Any]:
    """
    Run the complete Phase 5 evaluation suite.

    Methods:

        citation_baseline
        embedding_only
        full
        full_llm (optional)
        ablate_embedding
        ablate_recency
        ablate_citation
        ablate_author
    """
    personal = load_personal_rankings()
    personal_by_id = _build_personal_lookup(personal)

    k = max(1, int(k))

    if pool_size is None:
        pool_size = max(
            k,
            3 * k,
            DEFAULT_POOL_MULTIPLIER * k,
        )

    print()
    print("=" * 78)
    print("PHASE 5 — EVALUATION")
    print("=" * 78)
    print(f"Query:            {query}")
    print(f"Ground truth:     {PERSONAL_RANKINGS_PATH}")
    print(f"Ground truth n:   {len(personal)}")
    print(f"Evaluation k:     {k}")
    print(f"Retrieval pool:   {pool_size}")
    print(f"Half-life:        {half_life}")
    print(f"LLM evaluation:   {'yes' if use_llm else 'no'}")
    print("=" * 78)
    print()

    results: list[dict[str, Any]] = []

    rankings: dict[str, list[dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Citation baseline
    # ------------------------------------------------------------------

    print("Running citation baseline...")
    citation_ranked = rank_citation_baseline(
        query=query,
        k=k,
        half_life=half_life,
        pool_size=pool_size,
    )

    rankings["citation_baseline"] = citation_ranked

    results.append(
        _metric_row(
            "citation_baseline",
            evaluate_ranking(
                citation_ranked,
                personal_by_id,
            ),
        )
    )

    # ------------------------------------------------------------------
    # Embedding-only baseline
    # ------------------------------------------------------------------

    print("Running embedding-only baseline...")
    embedding_ranked = rank_embedding_only(
        query=query,
        k=k,
        half_life=half_life,
        pool_size=pool_size,
    )

    rankings["embedding_only"] = embedding_ranked

    results.append(
        _metric_row(
            "embedding_only",
            evaluate_ranking(
                embedding_ranked,
                personal_by_id,
            ),
        )
    )

    # ------------------------------------------------------------------
    # Full Phase 4 ranker
    # ------------------------------------------------------------------

    print("Running full multi-signal ranker...")
    full_ranked = rank_with_weights(
        query=query,
        weights=DEFAULT_WEIGHTS,
        k=k,
        half_life=half_life,
        pool_size=pool_size,
    )

    rankings["full"] = full_ranked

    results.append(
        _metric_row(
            "full",
            evaluate_ranking(
                full_ranked,
                personal_by_id,
            ),
        )
    )

    # ------------------------------------------------------------------
    # Optional full + LLM
    # ------------------------------------------------------------------

    if use_llm:
        print("Running full + LLM ranker...")

        full_llm_ranked = score_and_rank(
            query=query,
            k=k,
            weights=DEFAULT_WEIGHTS,
            half_life=half_life,
            temperature=0.0,
            use_llm=True,
            pool_size=pool_size,
        )

        rankings["full_llm"] = full_llm_ranked

        results.append(
            _metric_row(
                "full_llm",
                evaluate_ranking(
                    full_llm_ranked,
                    personal_by_id,
                ),
            )
        )

    # ------------------------------------------------------------------
    # Ablations
    # ------------------------------------------------------------------

    ablation_names = (
        "embedding",
        "recency",
        "citation",
        "author",
    )

    for signal in ablation_names:
        method_name = f"ablate_{signal}"

        print(f"Running {method_name}...")

        weights = ablation_weights(
            signal,
            DEFAULT_WEIGHTS,
        )

        ranked = rank_with_weights(
            query=query,
            weights=weights,
            k=k,
            half_life=half_life,
            pool_size=pool_size,
        )

        rankings[method_name] = ranked

        results.append(
            _metric_row(
                method_name,
                evaluate_ranking(
                    ranked,
                    personal_by_id,
                ),
            )
        )

    evaluation = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "k": k,
            "pool_size": pool_size,
            "half_life": half_life,
            "llm_enabled": use_llm,
            "personal_ranking_path": str(
                PERSONAL_RANKINGS_PATH
            ),
            "ground_truth_count": len(personal),
            "default_weights": dict(DEFAULT_WEIGHTS),
        },
        "results": results,
        "rankings": rankings,
        "personal_rankings": personal,
    }

    return evaluation


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def _format_metric(value: Any) -> str:
    """Format a metric for Markdown."""
    if value is None:
        return "—"

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _find_result(
    results: list[dict[str, Any]],
    method: str,
) -> dict[str, Any] | None:
    """Find one method's metric row."""
    for row in results:
        if row["method"] == method:
            return row

    return None


def _conclusion_sentence(
    results: list[dict[str, Any]],
) -> str:
    """
    Produce the required automatic conclusion comparing full vs citation.
    """
    full = _find_result(results, "full")
    citation = _find_result(results, "citation_baseline")

    if full is None or citation is None:
        return (
            "The multi-signal ranker could not be compared with the "
            "citation baseline."
        )

    full_rho = full.get("spearman_rho")
    citation_rho = citation.get("spearman_rho")

    if full_rho is None or citation_rho is None:
        return (
            "The multi-signal ranker and citation baseline were evaluated, "
            "but Spearman ρ could not be computed for one or both methods."
        )

    difference = float(full_rho) - float(citation_rho)

    if abs(difference) < 1e-9:
        verdict = "ties"
    elif difference > 0:
        verdict = "beats"
    else:
        verdict = "loses to"

    return (
        "The multi-signal ranker "
        f"{verdict} the citation baseline on my personal ranking "
        f"(Spearman ρ = {float(full_rho):.4f} vs "
        f"{float(citation_rho):.4f})."
    )


def write_results_table(
    evaluation: dict[str, Any],
    path: str | Path = RESULTS_TABLE_PATH,
    json_path: str | Path = RESULTS_JSON_PATH,
) -> None:
    """
    Write:

        results/phase5_results_table.md
        results/phase5_results.json
    """
    path = Path(path)
    json_path = Path(json_path)

    path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = evaluation["metadata"]
    results = evaluation["results"]

    lines: list[str] = []

    lines.append("# Phase 5 Evaluation Results")
    lines.append("")
    lines.append(f"**Query:** {metadata['query']}")
    lines.append("")
    lines.append(
        f"**Ground truth:** `{metadata['personal_ranking_path']}`"
    )
    lines.append("")
    lines.append(
        f"**Ground-truth papers:** {metadata['ground_truth_count']}"
    )
    lines.append("")
    lines.append(f"**Evaluation k:** {metadata['k']}")
    lines.append("")
    lines.append(f"**Retrieval pool:** {metadata['pool_size']}")
    lines.append("")
    lines.append(f"**Recency half-life:** {metadata['half_life']}")
    lines.append("")
    lines.append(
        f"**LLM enabled:** {'yes' if metadata['llm_enabled'] else 'no'}"
    )
    lines.append("")

    lines.append("## Metrics")
    lines.append("")
    lines.append(
        "| Method | Spearman ρ | NDCG@10 | NDCG@20 | "
        "Intersection |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|"
    )

    for row in results:
        lines.append(
            f"| `{row['method']}` | "
            f"{_format_metric(row['spearman_rho'])} | "
            f"{_format_metric(row['ndcg_at_10'])} | "
            f"{_format_metric(row['ndcg_at_20'])} | "
            f"{row['intersection_count']} |"
        )

    lines.append("")
    lines.append("## Weights")
    lines.append("")
    lines.append(
        "| Signal | Default weight |"
    )
    lines.append(
        "|---|---:|"
    )

    for signal, weight in metadata["default_weights"].items():
        lines.append(
            f"| `{signal}` | {float(weight):.4f} |"
        )

    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append(_conclusion_sentence(results))
    lines.append("")

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # Do not dump all ranked papers into the human-readable table.
    # The JSON contains the complete evaluation and rankings.
    serializable = {
        "metadata": metadata,
        "results": results,
        "personal_rankings": evaluation["personal_rankings"],
    }

    json_path.write_text(
        json.dumps(
            serializable,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI output
# ---------------------------------------------------------------------------

def print_results(
    evaluation: dict[str, Any],
) -> None:
    """Print a compact results table to the terminal."""
    results = evaluation["results"]

    print()
    print("=" * 96)
    print("PHASE 5 RESULTS")
    print("=" * 96)

    print(
        f"{'Method':<22}"
        f"{'Spearman':>12}"
        f"{'NDCG@10':>12}"
        f"{'NDCG@20':>12}"
        f"{'Overlap':>10}"
    )

    print("-" * 96)

    for row in results:
        print(
            f"{row['method']:<22}"
            f"{_format_metric(row['spearman_rho']):>12}"
            f"{_format_metric(row['ndcg_at_10']):>12}"
            f"{_format_metric(row['ndcg_at_20']):>12}"
            f"{row['intersection_count']:>10}"
        )

    print("=" * 96)

    print()
    print("Conclusion:")
    print(_conclusion_sentence(results))
    print()

    print(f"Results table: {RESULTS_TABLE_PATH}")
    print(f"Results JSON:  {RESULTS_JSON_PATH}")
    print(f"Error analysis: {ERROR_ANALYSIS_PATH}")
    print()


def run_command(args: argparse.Namespace) -> None:
    """Handle the `run` command."""
    evaluation = run_evaluation(
        query=args.query,
        k=args.k,
        half_life=args.half_life,
        pool_size=args.pool_size,
        use_llm=args.llm,
    )

    write_results_table(
        evaluation,
        path=RESULTS_TABLE_PATH,
        json_path=RESULTS_JSON_PATH,
    )

    # Error analysis is always based on the main full ranker without LLM.
    personal = load_personal_rankings()

    full_ranked = evaluation["rankings"]["full"]

    write_error_analysis(
        ranked=full_ranked,
        personal=personal,
        path=ERROR_ANALYSIS_PATH,
        top_n=args.top,
        query=args.query,
    )

    print_results(evaluation)


def errors_command(args: argparse.Namespace) -> None:
    """
    Run the full ranker and produce only qualitative error analysis.
    """
    personal = load_personal_rankings()

    print()
    print(
        f"Running full Phase 4 ranking for error analysis "
        f"(k={args.k})..."
    )

    ranked = rank_with_weights(
        query=args.query,
        weights=DEFAULT_WEIGHTS,
        k=args.k,
        half_life=args.half_life,
        pool_size=args.pool_size,
    )

    write_error_analysis(
        ranked=ranked,
        personal=personal,
        path=ERROR_ANALYSIS_PATH,
        top_n=args.top,
        query=args.query,
    )

    print()
    print(f"Wrote error analysis to:")
    print(f"  {ERROR_ANALYSIS_PATH}")
    print()

    # Also print the actual disagreements so they can be inspected quickly.
    rows = _make_error_rows(
        ranked,
        personal,
    )

    underranked = sorted(
        rows,
        key=lambda item: item["rank_gap"],
        reverse=True,
    )[:args.top]

    overranked = sorted(
        rows,
        key=lambda item: item["rank_gap"],
    )[:args.top]

    print("=" * 96)
    print("BIGGEST UNDERRANKS")
    print("=" * 96)

    for index, item in enumerate(underranked, start=1):
        print(
            f"{index:2}. "
            f"system #{item['system_rank']} vs "
            f"personal #{item['personal_rank']:.1f} | "
            f"score={item['personal_score']:.1f} | "
            f"{item['title']}"
        )

    print()
    print("=" * 96)
    print("BIGGEST OVERRANKS")
    print("=" * 96)

    for index, item in enumerate(overranked, start=1):
        print(
            f"{index:2}. "
            f"system #{item['system_rank']} vs "
            f"personal #{item['personal_rank']:.1f} | "
            f"score={item['personal_score']:.1f} | "
            f"{item['title']}"
        )

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Construct the Phase 5 command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase 5 evaluation of the Phase 4 paper-ranking system."
        )
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    run = sub.add_parser(
        "run",
        help="Run the complete Phase 5 evaluation suite.",
    )

    run.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help=(
            "Evaluation query. A broad query covering the main personal "
            "research interests is recommended."
        ),
    )

    run.add_argument(
        "-k",
        type=int,
        default=DEFAULT_K,
        help=(
            "Number of papers in the final evaluation ranking. "
            "Default: 40."
        ),
    )

    run.add_argument(
        "--pool-size",
        type=int,
        default=None,
        help=(
            "Number of Phase 3 retrieval candidates before scoring. "
            "Defaults to at least 3*k."
        ),
    )

    run.add_argument(
        "--half-life",
        type=float,
        default=DEFAULT_HALF_LIFE,
        help="Recency half-life in years.",
    )

    run.add_argument(
        "--llm",
        action="store_true",
        help="Also evaluate the optional full + LLM ranker.",
    )

    run.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_ERRORS,
        help="Number of errors written to the error analysis.",
    )

    run.set_defaults(func=run_command)

    # ------------------------------------------------------------------
    # errors
    # ------------------------------------------------------------------

    errors = sub.add_parser(
        "errors",
        help="Generate qualitative underrank/overrank analysis.",
    )

    errors.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Evaluation query.",
    )

    errors.add_argument(
        "-k",
        type=int,
        default=DEFAULT_K,
        help="Number of ranked papers to inspect.",
    )

    errors.add_argument(
        "--pool-size",
        type=int,
        default=None,
        help="Number of Phase 3 retrieval candidates before scoring.",
    )

    errors.add_argument(
        "--half-life",
        type=float,
        default=DEFAULT_HALF_LIFE,
        help="Recency half-life in years.",
    )

    errors.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_ERRORS,
        help="Number of biggest underranks/overranks to show.",
    )

    errors.set_defaults(func=errors_command)

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if getattr(args, "k", 1) <= 0:
        parser.error("-k must be greater than zero.")

    if getattr(args, "top", 1) <= 0:
        parser.error("--top must be greater than zero.")

    if getattr(args, "half_life", 1.0) <= 0:
        parser.error("--half-life must be greater than zero.")

    if getattr(args, "pool_size", None) is not None:
        if args.pool_size < args.k:
            parser.error(
                "--pool-size must be at least as large as -k."
            )

    args.func(args)


if __name__ == "__main__":
    main()