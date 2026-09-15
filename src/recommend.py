#!/usr/bin/env python3
"""
Phase 6 -- User-facing entry point.

This is a *thin* wrapper around Phase 4's `score_and_rank()`. It adds no new
scoring logic: it calls Phase 4 exactly as the CLI does, then applies simple
post-hoc filters to the returned, already-ranked list.

Examples:
  python src/recommend.py "mechanistic interpretability circuits" -k 8
  python src/recommend.py "random matrix loss surfaces" --since 2020 -k 8
  python src/recommend.py "stat mech of deep learning" --prefer-recent -k 8
  python src/recommend.py --demo

Design notes (documented here since these choices are not part of Phase 4):

* --prefer-recent does not add a new scoring path. It calls Phase 4 with a
  shorter recency half-life (PREFER_RECENT_HALF_LIFE years instead of Phase
  4's default of 4.0). A shorter half-life makes `recency_score()` decay
  faster with age, so newer papers keep more of their recency credit
  relative to older ones. Everything else about the ranking is untouched.

* --since / --until / --min-citations / --arxiv-only are all applied *after*
  score_and_rank() returns. Phase 4's public API has no equivalent
  arguments, and Phase 4 is intentionally not modified to add them.

* Because filtering happens after ranking, requesting exactly `k` results
  from Phase 4 and then filtering could leave you with fewer than `k`
  papers, even when more matching papers exist. To keep --since/--until/
  --min-citations/--arxiv-only useful, this wrapper asks Phase 4 for a
  larger candidate pool whenever a filter is active, then trims the
  filtered, already-ranked list down to the requested k. No re-ranking
  happens here: order is exactly what Phase 4 returned.

* A caveat of the above: --temperature (MMR diversity) is applied by
  Phase 4 across the *overfetched* pool, not across the final filtered
  k. If a filter removes some of the papers MMR chose for diversity, the
  final displayed set may be less diverse than a temperature-only run
  would suggest. This is a straightforward consequence of filtering being
  a wrapper-level, post-hoc step, and is noted in the README.

* Missing metadata is treated conservatively: a paper with no parseable
  year fails --since/--until, and a paper with no parseable citation count
  fails --min-citations. --arxiv-only fails papers with an empty/missing
  arxiv_id. This avoids silently including papers that cannot be verified
  against the filter.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

try:
    from phase4_scoring import (
        DEFAULT_HALF_LIFE,
        parse_weights,
        print_ranked_results,
        score_and_rank,
    )
except ImportError:
    from src.phase4_scoring import (
        DEFAULT_HALF_LIFE,
        parse_weights,
        print_ranked_results,
        score_and_rank,
    )


# Shorter than Phase 4's default half-life (4.0 years) so recency dominates
# more strongly relative to embedding/citation/author signals.
PREFER_RECENT_HALF_LIFE = 1.5

# Overfetch controls used only when a post-hoc filter is active.
OVERFETCH_MULTIPLIER = 6
MIN_OVERFETCH = 40
MAX_OVERFETCH = 300

DEMO_QUERIES = [
    "mechanistic interpretability circuits",
    "random matrix theory neural network loss surfaces",
    "statistical mechanics of deep learning",
    "mechanistic interpretability",
]
DEMO_K = 5


_YEAR_RE = re.compile(r"(19|20)\d{2}")


def _parse_year(value: Any) -> int | None:
    """Extract a four-digit year from whatever Phase 4 put in `year`."""
    if value is None:
        return None
    match = _YEAR_RE.search(str(value))
    return int(match.group(0)) if match else None


def _parse_citations(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_fetch_k(k: int, filters_active: bool) -> int:
    """How many results to request from Phase 4 before filtering."""
    if not filters_active:
        return k
    overfetch = max(k * OVERFETCH_MULTIPLIER, MIN_OVERFETCH)
    return min(overfetch, MAX_OVERFETCH)


def apply_filters(
    results: list[dict[str, Any]],
    since: int | None,
    until: int | None,
    min_citations: float | None,
    arxiv_only: bool,
) -> list[dict[str, Any]]:
    """Apply wrapper-level filters to an already-ranked Phase 4 result list.

    Order is preserved -- this never re-scores or re-sorts.
    """
    filtered = []

    for item in results:
        if since is not None or until is not None:
            year = _parse_year(item.get("year"))
            if year is None:
                continue
            if since is not None and year < since:
                continue
            if until is not None and year > until:
                continue

        if min_citations is not None:
            cites = _parse_citations(item.get("citation_count"))
            if cites is None or cites < min_citations:
                continue

        if arxiv_only:
            arxiv_id = str(item.get("arxiv_id") or "").strip()
            if not arxiv_id:
                continue

        filtered.append(item)

    return filtered


def run_query(args: argparse.Namespace) -> list[dict[str, Any]]:
    filters_active = (
        args.since is not None
        or args.until is not None
        or args.min_citations is not None
        or args.arxiv_only
    )

    fetch_k = compute_fetch_k(args.k, filters_active)

    half_life = (
        PREFER_RECENT_HALF_LIFE if args.prefer_recent else DEFAULT_HALF_LIFE
    )

    results = score_and_rank(
        query=args.query,
        k=fetch_k,
        weights=args.weights,
        half_life=half_life,
        temperature=args.temperature,
        use_llm=args.llm,
    )

    filtered = apply_filters(
        results,
        since=args.since,
        until=args.until,
        min_citations=args.min_citations,
        arxiv_only=args.arxiv_only,
    )

    return filtered[: args.k]


def run_demo() -> None:
    print("Phase 6 demo -- running score_and_rank() on example queries.\n")
    for query in DEMO_QUERIES:
        results = score_and_rank(query=query, k=DEMO_K)
        print_ranked_results(query, results)
        print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 6 recommendation CLI. Thin wrapper around Phase 4's "
            "score_and_rank(), with post-hoc filtering."
        )
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Free-text research interest. Not required with --demo.",
    )
    parser.add_argument("-k", type=int, default=10, help="Number of results to show.")
    parser.add_argument("--since", type=int, default=None, metavar="YEAR")
    parser.add_argument("--until", type=int, default=None, metavar="YEAR")
    parser.add_argument(
        "--prefer-recent",
        action="store_true",
        help=f"Use a shorter recency half-life ({PREFER_RECENT_HALF_LIFE} years).",
    )
    parser.add_argument(
        "--arxiv-only",
        action="store_true",
        help="Keep only papers with a non-empty arXiv identifier.",
    )
    parser.add_argument(
        "--min-citations",
        type=float,
        default=None,
        metavar="N",
        help="Keep only papers with at least N citations.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="MMR diversity strength passed through to Phase 4, 0 to 1.",
    )
    parser.add_argument(
        "--weights",
        type=parse_weights,
        default=None,
        metavar="E,R,C,A",
        help="Weights for embedding,recency,citation,author, passed through to Phase 4.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable Phase 4's optional LLM relevance judge.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print only machine-readable JSON.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a few polished example queries and exit.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    if not args.query or not args.query.strip():
        parser.error("a query is required unless --demo is given")

    if args.k <= 0:
        parser.error("-k must be greater than zero")

    if args.temperature < 0 or args.temperature > 1:
        parser.error("--temperature must be between 0 and 1")

    if args.min_citations is not None and args.min_citations < 0:
        parser.error("--min-citations must be non-negative")

    if args.since is not None and args.until is not None and args.since > args.until:
        parser.error("--since cannot be greater than --until")

    results = run_query(args)

    if args.as_json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        if not results:
            print(
                "No papers matched the requested filters.",
                file=sys.stderr,
            )
        print_ranked_results(args.query, results)


if __name__ == "__main__":
    main()
