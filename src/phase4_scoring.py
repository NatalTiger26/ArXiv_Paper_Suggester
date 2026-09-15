#!/usr/bin/env python3
"""
Phase 4 — Multi-signal paper scoring and ranking.

This module builds on Phase 3's embedding retrieval and adds:
  - embedding similarity
  - publication recency
  - citation count
  - an author-frequency proxy
  - an optional LLM relevance judge
  - weighted score contributions
  - MMR-style title diversity

Examples:
  python src/phase4_scoring.py rank "mechanistic interpretability circuits" -k 10
  python src/phase4_scoring.py rank "random matrix loss landscape" -k 10 --temperature 0.35
  python src/phase4_scoring.py rank "statistical mechanics of deep learning" \
      --weights 0.45,0.25,0.25,0.05 --half-life 3
  python src/phase4_scoring.py rank "sparse autoencoders transformers" --llm -k 8
  python src/phase4_scoring.py rank "..." -k 5 --json

LLM judge configuration:
  NVIDIA_API_KEY       Preferred when available. NVIDIA's API is OpenAI-compatible.
  NVIDIA_MODEL         Optional model override.
  GROQ_API_KEY         Fallback provider.
  GROQ_MODEL           Optional model override.
  OPENROUTER_API_KEY   Second fallback provider.
  OPENROUTER_MODEL     Optional model override.

The LLM judge is deliberately non-blocking: any missing key, dependency,
network error, malformed response, or provider error simply disables the
LLM signal for that run.
"""



from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable

try:
    # Works both when executed as `python src/phase4_scoring.py` and when the
    # src directory is already on PYTHONPATH.
    from phase3_embeddings import load_corpus, load_model, paper_id, retrieve
except ImportError:
    from src.phase3_embeddings import load_corpus, load_model, paper_id, retrieve


DEFAULT_WEIGHTS = {
    "embedding": 0.50,
    "recency": 0.20,
    "citation": 0.25,
    "author": 0.05,
}

DEFAULT_HALF_LIFE = 4.0
DEFAULT_TEMPERATURE = 0.0
DEFAULT_POOL_MULTIPLIER = 3
MIN_POOL_SIZE = 20

# The LLM is an optional secondary signal. Keep its influence small enough
# that a provider/model failure cannot dominate the retrieval-based ranking.
DEFAULT_LLM_WEIGHT = 0.10

# NVIDIA's hosted NIM API is OpenAI-compatible.
# The model can still be overridden with NVIDIA_MODEL.
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_MODEL = "openai/gpt-oss-20b"

DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct:free"


# ---------------------------------------------------------------------------
# Basic signal functions
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float | None = None) -> float | None:
    """Convert common numeric/string values to float without raising."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _current_year() -> int:
    return datetime.now(timezone.utc).year


def _extract_year(value: Any) -> int | None:
    """Extract a four-digit publication year from a year/date-like value."""
    if value is None:
        return None
    match = re.search(r"(19|20)\d{2}", str(value))
    return int(match.group(0)) if match else None


def recency_score(year: Any, half_life_years: float = DEFAULT_HALF_LIFE) -> float:
    """
    Exponential recency score.

    score = 0.5 ** (age / half_life)

    Missing/invalid years receive a neutral score of 0.3 rather than being
    treated as either maximally recent or maximally old.
    """
    year_int = _extract_year(year)
    half_life = max(float(half_life_years), 1e-9)

    if year_int is None:
        return 0.3

    age = max(0.0, float(_current_year() - year_int))
    return float(0.5 ** (age / half_life))


def citation_score(cites: Any) -> float:
    """
    Log-normalized citation score.

    score = log1p(citations) / log1p(500)

    Missing citations are neutral (0.3). Scores are capped at 1.0 so a
    highly cited paper cannot make this nominally normalized signal exceed 1.
    """
    if cites is None or cites == "":
        return 0.3

    value = _safe_float(cites)
    if value is None or value < 0:
        return 0.3

    return min(1.0, math.log1p(value) / math.log1p(500))


def _author_name(author: Any) -> str | None:
    """Normalize the common author representations found in paper metadata."""
    if isinstance(author, str):
        name = author.strip()
        return name or None

    if isinstance(author, dict):
        for key in ("name", "full_name", "author", "display_name"):
            value = author.get(key)
            if value:
                return str(value).strip() or None

    return None


def _paper_authors(paper: dict[str, Any]) -> list[str]:
    """
    Extract normalized author names from common corpus schemas.

    The corpus is allowed to contain:
      authors: ["Alice", "Bob"]
    or:
      authors: [{"name": "Alice"}, {"name": "Bob"}]
    """
    raw = paper.get("authors")

    if raw is None:
        raw = paper.get("author")

    if isinstance(raw, str):
        raw = [raw]

    if not isinstance(raw, (list, tuple)):
        return []

    authors: list[str] = []
    for item in raw:
        name = _author_name(item)
        if name:
            authors.append(name)

    return authors


def _build_author_frequency(corpus: Iterable[dict[str, Any]]) -> Counter[str]:
    """
    Count how often each author appears across the corpus.

    Each paper contributes at most one count per author.
    """
    counts: Counter[str] = Counter()

    for paper in corpus:
        counts.update(set(_paper_authors(paper)))

    return counts


def author_score(
    paper: dict[str, Any],
    author_frequency: Counter[str],
) -> float:
    """
    Author-frequency proxy.

    Take the most frequent co-author in the corpus and log-normalize that
    frequency. This is intentionally a proxy for how established/recurrent
    the paper's authors are *within this particular corpus*, not a measure
    of author quality.

    With no author metadata, return the neutral value 0.3.
    """
    authors = _paper_authors(paper)
    if not authors:
        return 0.3

    max_frequency = max(
        (author_frequency.get(author, 0) for author in authors),
        default=0,
    )

    if max_frequency <= 0:
        return 0.3

    # A frequency of 1 maps to 0, while repeated authors approach 1.
    # log1p(10) is used as a practical saturation point for a 100-paper corpus.
    denominator = math.log1p(10)
    return min(1.0, math.log1p(max_frequency) / denominator)


# ---------------------------------------------------------------------------
# Weighted combination
# ---------------------------------------------------------------------------

def _validate_weights(weights: dict[str, float]) -> dict[str, float]:
    required = ("embedding", "recency", "citation", "author")
    cleaned: dict[str, float] = {}

    for name in required:
        value = _safe_float(weights.get(name))
        if value is None or value < 0:
            raise ValueError(f"Invalid weight for {name!r}: {weights.get(name)!r}")
        cleaned[name] = float(value)

    if sum(cleaned.values()) <= 0:
        raise ValueError("At least one scoring weight must be greater than zero.")

    return cleaned


def combine_scores(
    signals: dict[str, float],
    weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """
    Combine the four core signals and return:

      (weighted score, contribution breakdown)

    Each contribution is explicitly weight * signal, making it possible to
    inspect which signal drove a paper's score.
    """
    weights = _validate_weights(weights)

    contributions = {
        name: float(signals[name]) * weights[name]
        for name in ("embedding", "recency", "citation", "author")
    }

    return float(sum(contributions.values())), contributions


# ---------------------------------------------------------------------------
# MMR / title diversity
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _title_tokens(title: str) -> set[str]:
    return {
        token.lower()
        for token in _WORD_RE.findall(title or "")
        if len(token) > 2
    }


def _title_overlap(title_a: str, title_b: str) -> float:
    """
    Jaccard overlap between title tokens.

    This intentionally stays lightweight: Phase 3 already provides the
    semantic embedding signal, while Phase 4 uses title overlap only as a
    transparent diversity proxy.
    """
    a = _title_tokens(title_a)
    b = _title_tokens(title_b)

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def mmr_rerank(
    papers: list[dict[str, Any]],
    k: int,
    temperature: float = DEFAULT_TEMPERATURE,
) -> list[dict[str, Any]]:
    """
    MMR-style reranking using title overlap as the diversity penalty.

    temperature=0 means no diversity reranking and preserves score order.

    As temperature increases, the ranking increasingly favors papers whose
    titles overlap less with papers already selected.
    """
    k = max(0, int(k))
    if k == 0 or not papers:
        return []

    temperature = min(1.0, max(0.0, float(temperature)))

    if temperature <= 0.0:
        return papers[:k]

    remaining = list(papers)
    selected: list[dict[str, Any]] = []

    while remaining and len(selected) < k:
        best_index = 0
        best_value = -float("inf")

        for index, candidate in enumerate(remaining):
            relevance = float(candidate.get("final_score", 0.0))

            if not selected:
                diversity_penalty = 0.0
            else:
                diversity_penalty = max(
                    _title_overlap(
                        candidate.get("title", ""),
                        chosen.get("title", ""),
                    )
                    for chosen in selected
                )

            # MMR interpolation:
            #   temp=0 -> relevance only
            #   temp=1 -> minimize title overlap as much as possible
            mmr_value = (
                (1.0 - temperature) * relevance
                - temperature * diversity_penalty
            )

            if mmr_value > best_value:
                best_value = mmr_value
                best_index = index

        selected.append(remaining.pop(best_index))

    return selected


# ---------------------------------------------------------------------------
# Optional LLM judge
# ---------------------------------------------------------------------------

def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object, including one wrapped in a markdown code fence."""
    if not text:
        return None

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    # Last-resort extraction of the outermost JSON object.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(cleaned[start : end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None

    return None


def _llm_provider_config() -> tuple[str, str, str] | None:
    """
    Select the first configured provider.

    NVIDIA is preferred because the user can use an NVIDIA API key directly;
    Groq and OpenRouter remain compatible fallbacks.
    """
    if os.getenv("NVIDIA_API_KEY"):
        return (
            NVIDIA_BASE_URL,
            os.getenv("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL),
            os.environ["NVIDIA_API_KEY"],
        )

    if os.getenv("GROQ_API_KEY"):
        return (
            DEFAULT_GROQ_BASE_URL,
            os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            os.environ["GROQ_API_KEY"],
        )

    if os.getenv("OPENROUTER_API_KEY"):
        return (
            DEFAULT_OPENROUTER_BASE_URL,
            os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL),
            os.environ["OPENROUTER_API_KEY"],
        )

    return None


def _call_openai_compatible_judge(
    base_url: str,
    model: str,
    api_key: str,
    query: str,
    papers: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Call an OpenAI-compatible provider and ask it to score paper relevance.

    Returns a paper_id -> [0,1] mapping. Any exception is intentionally
    propagated to the outer optional_llm_judge(), which converts it to {}.
    """
    try:
        from openai import OpenAI
    except ImportError:
        # Keep the feature optional. The main ranking path does not require
        # the OpenAI SDK.
        raise RuntimeError("openai package is not installed")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=120.0,
    )

    candidates = []
    for paper in papers:
        candidates.append(
            {
                "paper_id": str(paper.get("paper_id", "")),
                "title": paper.get("title", ""),
                "year": paper.get("year", ""),
                "snippet": paper.get("snippet") or paper.get("_snippet", ""),
            }
        )

    prompt = f"""
You are a relevance judge for a scientific-paper retrieval system.

Query:
{query}

For every candidate paper below, assign a relevance score from 0 to 1.
The score should reflect how useful the paper is for answering or researching
the query. Judge the title/snippet against the query, not citation count,
author prominence, or publication year.

Return ONLY valid JSON in exactly this shape:
{{
  "paper_id_1": 0.0,
  "paper_id_2": 0.0
}}

Candidates:
{json.dumps(candidates, ensure_ascii=False)}
""".strip()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a precise scientific information-retrieval judge. Return JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=max(1000, len(candidates) * 100),
        # max_tokens=max(256, len(candidates) * 24),
    )

    content = response.choices[0].message.content or ""
    parsed = _extract_json_object(content)

    if not parsed:
        raise ValueError("LLM judge returned no valid JSON object.")

    scores: dict[str, float] = {}
    for pid, value in parsed.items():
        score = _safe_float(value)
        if score is None:
            continue
        scores[str(pid)] = min(1.0, max(0.0, score))

    return scores


def optional_llm_judge(
    query: str,
    papers: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Optional LLM relevance judge.

    Provider/key/model/dependency/network/response failures all result in {}.
    The retrieval system therefore never depends on the LLM being available.
    """
    if not papers:
        return {}

    config = _llm_provider_config()
    if config is None:
        return {}

    base_url, model, api_key = config

    try:
        return _call_openai_compatible_judge(
            base_url=base_url,
            model=model,
            api_key=api_key,
            query=query,
            papers=papers,
        )
    except Exception as exc:
        # The LLM is explicitly an optional signal. Do not make ranking fail.
        print(
            f"Warning: LLM judge unavailable ({type(exc).__name__}: {exc}). "
            "Continuing without LLM scores.",
            file=sys.stderr,
        )
        return {}


# ---------------------------------------------------------------------------
# Main Phase 4 API
# ---------------------------------------------------------------------------

def _paper_lookup(corpus: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index corpus records by all IDs that Phase 3 may expose."""
    lookup: dict[str, dict[str, Any]] = {}

    for paper in corpus:
        pid = paper_id(paper)
        lookup[str(pid)] = paper

        arxiv = paper.get("arxiv_id")
        if arxiv:
            lookup[str(arxiv)] = paper

        doi = paper.get("doi")
        if doi:
            lookup[f"doi:{doi}"] = paper

        internal_id = paper.get("internal_id")
        if internal_id:
            lookup[str(internal_id)] = paper

    return lookup


def _resolve_year(
    result: dict[str, Any],
    paper: dict[str, Any] | None,
) -> Any:
    if result.get("year"):
        return result.get("year")

    if paper:
        return paper.get("published_date") or paper.get("year")

    return ""


def _resolve_citations(
    result: dict[str, Any],
    paper: dict[str, Any] | None,
) -> Any:
    # Phase 3 serializes citation_count into Chroma metadata, so its result
    # is the first source of truth.
    if result.get("citation_count") not in (None, ""):
        return result.get("citation_count")

    if paper:
        return paper.get("citation_count")

    return None


def score_and_rank(
    query: str,
    k: int = 10,
    weights: dict[str, float] | None = None,
    half_life: float = DEFAULT_HALF_LIFE,
    temperature: float = DEFAULT_TEMPERATURE,
    use_llm: bool = False,
    model_name: str | None = None,
    pool_size: int | None = None,
) -> list[dict[str, Any]]:
    """
    Score and rank papers for a query.

    Exit API:
      score_and_rank(query, ...) -> ranked list with per-signal breakdown.

    The first-stage pool comes from Phase 3 retrieve(). A larger pool than k
    is used by default so MMR has meaningful alternatives to choose from.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    k = max(1, int(k))
    weights = _validate_weights(weights or DEFAULT_WEIGHTS)

    corpus = load_corpus()
    corpus_by_id = _paper_lookup(corpus)
    author_frequency = _build_author_frequency(corpus)

    if pool_size is None:
        pool_size = max(MIN_POOL_SIZE, k * DEFAULT_POOL_MULTIPLIER)
    pool_size = max(k, int(pool_size))

    # Reuse one model for the entire retrieval call. This avoids loading the
    # sentence-transformer repeatedly.
    model = load_model(model_name) if model_name else None

    retrieval = retrieve(
        query,
        k=pool_size,
        model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2",
        model=model,
    )

    scored: list[dict[str, Any]] = []

    for result in retrieval:
        pid = str(result.get("paper_id") or "")
        paper = corpus_by_id.get(pid)

        # Some corpora/indexes may use arXiv ID as the Chroma ID.
        if paper is None and result.get("arxiv_id"):
            paper = corpus_by_id.get(str(result["arxiv_id"]))

        year = _resolve_year(result, paper)
        cites = _resolve_citations(result, paper)

        signals = {
            "embedding": min(
                1.0,
                max(0.0, _safe_float(result.get("score"), 0.0) or 0.0),
            ),
            "recency": recency_score(year, half_life),
            "citation": citation_score(cites),
            "author": author_score(paper or {}, author_frequency),
        }

        final_score, contribution = combine_scores(signals, weights)

        scored.append(
            {
                "paper_id": pid,
                "arxiv_id": result.get("arxiv_id") or (paper or {}).get("arxiv_id") or "",
                "title": result.get("title") or (paper or {}).get("title") or "",
                "year": year,
                "citation_count": cites if cites is not None else "",
                "final_score": final_score,
                "signals": signals,
                "contribution": {
                    **contribution,
                    "weights": dict(weights),
                },
                # Kept internally until the optional LLM/MMR stages are done.
                "_snippet": result.get("snippet", ""),
            }
        )

    # Optional LLM judge.
    llm_scores: dict[str, float] = {}
    if use_llm and scored:
        llm_scores = optional_llm_judge(query, scored)

        if llm_scores:
            for item in scored:
                llm_value = llm_scores.get(str(item["paper_id"]))
                if llm_value is None:
                    continue

                item["signals"]["llm"] = llm_value

                # Blend the LLM as a small additional signal while keeping
                # the user's four explicit weights untouched.
                llm_contribution = DEFAULT_LLM_WEIGHT * llm_value
                item["final_score"] = (
                    (1.0 - DEFAULT_LLM_WEIGHT) * item["final_score"]
                    + llm_contribution
                )
                item["contribution"]["llm"] = llm_contribution
                item["contribution"]["llm_weight"] = DEFAULT_LLM_WEIGHT

    # Primary score ordering before diversity reranking.
    scored.sort(key=lambda item: item["final_score"], reverse=True)

    ranked = mmr_rerank(
        scored,
        k=k,
        temperature=temperature,
    )

    # Do not expose internal fields in the public exit API.
    for item in ranked:
        item.pop("_snippet", None)

        # Round only at the public boundary. Internal scoring keeps full
        # precision so the ordering is not affected by display rounding.
        item["final_score"] = round(float(item["final_score"]), 6)
        item["signals"] = {
            key: round(float(value), 6)
            for key, value in item["signals"].items()
        }

        for key, value in list(item["contribution"].items()):
            if key == "weights":
                item["contribution"][key] = {
                    wkey: round(float(wvalue), 6)
                    for wkey, wvalue in value.items()
                }
            else:
                item["contribution"][key] = round(float(value), 6)

    return ranked


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def parse_weights(text: str) -> dict[str, float]:
    """
    Parse CLI weights in the documented order:

      embedding,recency,citation,author
    """
    parts = [part.strip() for part in text.split(",")]

    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--weights must contain exactly four comma-separated values: "
            "embedding,recency,citation,author"
        )

    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--weights values must be numeric"
        ) from exc

    try:
        return _validate_weights(
            dict(zip(("embedding", "recency", "citation", "author"), values))
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def print_ranked_results(
    query: str,
    results: list[dict[str, Any]],
) -> None:
    print("\n" + "=" * 78)
    print(f"Query: {query}")
    print("=" * 78)

    for i, result in enumerate(results, 1):
        aid = result.get("arxiv_id") or result.get("paper_id") or ""
        print(
            f"{i:2}. [{result['final_score']:.3f}]  "
            f"{str(aid):16}  {result.get('title', '')[:58]}"
        )

        signals = result["signals"]
        print(
            "     signals  "
            f"emb={signals['embedding']:.2f}  "
            f"rec={signals['recency']:.2f}  "
            f"cit={signals['citation']:.2f}  "
            f"auth={signals['author']:.2f}"
            + (
                f"  llm={signals['llm']:.2f}"
                if "llm" in signals
                else ""
            )
        )

        contribution = result["contribution"]
        print(
            "     contrib  "
            f"emb={contribution['embedding']:.3f}  "
            f"rec={contribution['recency']:.3f}  "
            f"cit={contribution['citation']:.3f}  "
            f"auth={contribution['author']:.3f}"
            + (
                f"  llm={contribution['llm']:.3f}"
                if "llm" in contribution
                else ""
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4 multi-signal paper scoring and ranking."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Sentence-transformer model passed through to Phase 3.",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    rank = sub.add_parser("rank", help="Score and rank papers for a query.")
    rank.add_argument("query")
    rank.add_argument("-k", type=int, default=10)
    rank.add_argument(
        "--weights",
        type=parse_weights,
        default=dict(DEFAULT_WEIGHTS),
        metavar="E,R,C,A",
        help="Weights for embedding, recency, citation, author.",
    )
    rank.add_argument(
        "--half-life",
        type=float,
        default=DEFAULT_HALF_LIFE,
        help="Recency half-life in years.",
    )
    rank.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="MMR diversity strength, from 0 to 1.",
    )
    rank.add_argument(
        "--llm",
        action="store_true",
        help="Enable the optional LLM relevance judge.",
    )
    rank.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print only machine-readable JSON.",
    )
    rank.add_argument(
        "--pool-size",
        type=int,
        default=None,
        help="Number of Phase 3 retrieval candidates before scoring/MMR.",
    )

    args = parser.parse_args()

    if args.cmd == "rank":
        if args.temperature < 0 or args.temperature > 1:
            parser.error("--temperature must be between 0 and 1")

        if args.half_life <= 0:
            parser.error("--half-life must be greater than 0")

        results = score_and_rank(
            query=args.query,
            k=args.k,
            weights=args.weights,
            half_life=args.half_life,
            temperature=args.temperature,
            use_llm=args.llm,
            model_name=args.model,
            pool_size=args.pool_size,
        )

        if args.as_json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            print_ranked_results(args.query, results)


if __name__ == "__main__":
    main()
