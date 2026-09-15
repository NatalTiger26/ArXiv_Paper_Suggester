#!/usr/bin/env python3
"""
Phase 1 finisher: deduplicate, clean, merge all enrichment into one place,
and write a loadable corpus (~100 papers by default).

Inputs (any that exist):
  - data/papers.db          (from literature_collector_v2)
  - data/paper_scores.jsonl (from compute_impact_scores)
  - data/corpus_v1.jsonl    (optional export)

Outputs:
  - data/corpus_all.jsonl   full deduped merged corpus
  - data/corpus_100.json    curated ~100-paper corpus (Phase 1 exit file)
  - data/corpus_100.csv     same as CSV for spreadsheets

Usage:
  python src/consolidate_corpus.py
  python src/consolidate_corpus.py --target 100 --min-abstract 80
  python src/consolidate_corpus.py --no-topic-filter   # keep all topics

Load test (exit criterion):
  python -c "import json; c=json.load(open('data/corpus_100.json')); print(len(c), c[0]['title'][:60])"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


DB_PATH = Path("data/papers.db")
SCORES_PATH = Path("data/paper_scores.jsonl")
OUT_ALL = Path("data/corpus_all.jsonl")
OUT_JSON = Path("data/corpus_100.json")
OUT_CSV = Path("data/corpus_100.csv")

SEED_TOPICS = [
    "mechanistic interpretability",
    "statistical mechanics of learning",
    "random matrix theory neural networks",
]

# Title / topic keywords that strongly indicate on-topic work
TOPIC_KEYWORDS = [
    r"mechanistic interpretab",
    r"circuit discovery",
    r"sparse autoencoder",
    r"interpretab.*transformer",
    r"interpretab.*language model",
    r"interpretab.*llm",
    r"statistical mechanics of learning",
    r"statistical mechanics.*neural",
    r"statistical mechanics.*deep learning",
    r"random matrix.*neural",
    r"random matrix.*deep learning",
    r"loss surface",
    r"loss landscape",
    r"neural tangent",
    r"grokking",
    r"feature learning",
    r"implicit regularization",
    r"hessian.*neural",
    r"spectrum.*neural network",
    r"marchenko.?pastur",
]

# Obvious off-topic noise (biology / chemistry / clinical when not clearly MI)
NOISE_PATTERNS = [
    r"\bprotein structure\b",
    r"\balphafold\b",
    r"\bath erosclerosis\b",
    r"\baflatoxin\b",
    r"\bsulfonamide\b",
    r"\bbioconcentration\b",
    r"\bnitrification\b",
    r"\bchloroP\b",
    r"\bchloroplast\b",
    r"\bemt,\s*cscs\b",
    r"\bdrug resistance\b",
    r"\bclinical implications\b",
    r"\bmedical report generation\b",
    r"\bwildfire\b",
    r"\bcar-following\b",
    r"\bcryptocurrency portfolio\b",
]


def load_json(s: Optional[str]) -> Any:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def normalize_title(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def title_similarity(a: str, b: str) -> float:
    sa, sb = set(normalize_title(a).split()), set(normalize_title(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa), len(sb))


def is_real_arxiv_id(aid: Optional[str]) -> bool:
    if not aid:
        return False
    if aid.startswith(("doi:", "noarxiv:")):
        return False
    return bool(re.match(r"^(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})$", aid))


def clean_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    doi = doi.replace("https://doi.org/", "").strip()
    return doi or None


def looks_noisy(title: str, abstract: str) -> bool:
    text = f"{title or ''} {abstract or ''}"
    for pat in NOISE_PATTERNS:
        if re.search(pat, text, re.I):
            return True
    return False


def topic_relevance(title: str, abstract: str, topics: list, concepts: list) -> float:
    """0–1 heuristic relevance to seed research themes."""
    blob = " ".join(
        [
            title or "",
            abstract or "",
            " ".join(topics or []),
            " ".join(concepts or []),
        ]
    ).lower()
    hits = 0
    for pat in TOPIC_KEYWORDS:
        if re.search(pat, blob, re.I):
            hits += 1
    # also soft match seed phrases
    for seed in SEED_TOPICS:
        if seed.lower() in blob:
            hits += 2
    return min(hits / 4.0, 1.0)


def load_from_db(db_path: Path) -> list[dict]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    papers = [dict(r) for r in conn.execute("SELECT * FROM papers").fetchall()]
    oa = {r["arxiv_id"]: dict(r) for r in conn.execute("SELECT * FROM openalex").fetchall()}
    ss = {r["arxiv_id"]: dict(r) for r in conn.execute("SELECT * FROM semantic_scholar").fetchall()}
    oc = {r["arxiv_id"]: dict(r) for r in conn.execute("SELECT * FROM opencitations").fetchall()}
    seeds = defaultdict(list)
    for r in conn.execute("SELECT arxiv_id, seed_topic FROM seed_topics").fetchall():
        seeds[r["arxiv_id"]].append(r["seed_topic"])
    conn.close()

    merged = []
    for p in papers:
        aid = p["arxiv_id"]
        authors = load_json(p.get("authors_json")) or []
        categories = load_json(p.get("categories_json")) or []
        oa_row = oa.get(aid) or {}
        ss_row = ss.get(aid) or {}
        oc_row = oc.get(aid) or {}
        oa_raw = load_json(oa_row.get("raw_json")) or {}

        topics = []
        concepts = []
        if oa_raw:
            topics = [
                t.get("display_name") or t.get("id")
                for t in (oa_raw.get("topics") or [])[:10]
            ]
            concepts = [
                c.get("display_name")
                for c in (oa_raw.get("concepts") or [])[:10]
                if c.get("display_name")
            ]
        if not topics:
            tj = load_json(oa_row.get("topics_json")) or []
            topics = [t.get("display_name") for t in tj if isinstance(t, dict) and t.get("display_name")]
        if not concepts:
            cj = load_json(oa_row.get("concepts_json")) or []
            concepts = [c.get("display_name") for c in cj if isinstance(c, dict) and c.get("display_name")]

        fwci = oa_raw.get("fwci") if oa_raw else None
        cnp = oa_raw.get("citation_normalized_percentile") if oa_raw else None
        counts_by_year = oa_raw.get("counts_by_year") if oa_raw else None
        is_oa = None
        if oa_raw and oa_raw.get("open_access"):
            is_oa = oa_raw["open_access"].get("is_oa")
        if is_oa is None:
            oa_oa = load_json(oa_row.get("open_access_json")) or {}
            is_oa = oa_oa.get("is_oa")

        citation_count = None
        for c in (
            ss_row.get("citation_count"),
            oa_row.get("cited_by_count"),
            oc_row.get("citation_count"),
        ):
            if c is not None:
                citation_count = c if citation_count is None else max(citation_count, c)

        rec = {
            "arxiv_id": aid if is_real_arxiv_id(aid) else None,
            "internal_id": aid,
            "title": p.get("title"),
            "abstract": p.get("abstract"),
            "authors": authors,
            "categories": categories,
            "primary_category": p.get("primary_category"),
            "published_date": p.get("published_date"),
            "doi": clean_doi(p.get("doi")),
            "pdf_url": p.get("pdf_url"),
            "arxiv_url": p.get("arxiv_url"),
            "seed_topics": sorted(set(seeds.get(aid) or [])),
            # enrichment
            "citation_count": citation_count,
            "citation_count_openalex": oa_row.get("cited_by_count"),
            "citation_count_semantic": ss_row.get("citation_count"),
            "influential_citation_count": ss_row.get("influential_citation_count"),
            "reference_count": ss_row.get("reference_count"),
            "fwci": fwci,
            "citation_normalized_percentile": (
                cnp.get("value") if isinstance(cnp, dict) else cnp
            ),
            "is_top_1_percent": (
                cnp.get("is_in_top_1_percent") if isinstance(cnp, dict) else None
            ),
            "is_top_10_percent": (
                cnp.get("is_in_top_10_percent") if isinstance(cnp, dict) else None
            ),
            "counts_by_year": counts_by_year,
            "is_oa": is_oa,
            "topics": topics,
            "concepts": concepts,
            "openalex_id": oa_row.get("openalex_id"),
            "semantic_scholar_id": ss_row.get("paper_id"),
            "embedding_available": bool(ss_row.get("embedding_json")),
        }
        merged.append(rec)
    return merged


def load_scores(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        key = r.get("arxiv_id") or r.get("doi") or r.get("title")
        if key:
            out[str(key)] = r
    return out


def deduplicate(papers: list[dict]) -> list[dict]:
    """
    Dedup priority:
      1. real arXiv ID
      2. DOI
      3. near-identical title
    Prefer record with more enrichment / longer abstract.
    """
    by_arxiv: dict[str, dict] = {}
    by_doi: dict[str, dict] = {}
    by_title: list[dict] = []

    def richness(p: dict) -> tuple:
        return (
            1 if p.get("arxiv_id") else 0,
            1 if p.get("citation_count") is not None else 0,
            1 if p.get("fwci") is not None else 0,
            len(p.get("abstract") or ""),
            len(p.get("authors") or []),
        )

    def keep_better(old: dict, new: dict) -> dict:
        return new if richness(new) > richness(old) else old

    for p in papers:
        aid = p.get("arxiv_id")
        doi = p.get("doi")
        title = p.get("title") or ""

        if aid:
            if aid in by_arxiv:
                by_arxiv[aid] = keep_better(by_arxiv[aid], p)
            else:
                by_arxiv[aid] = p
            continue

        if doi:
            if doi in by_doi:
                by_doi[doi] = keep_better(by_doi[doi], p)
            else:
                by_doi[doi] = p
            continue

        # title near-dup against existing
        matched = False
        for i, existing in enumerate(by_title):
            if title_similarity(title, existing.get("title") or "") >= 0.92:
                by_title[i] = keep_better(existing, p)
                matched = True
                break
        if not matched:
            by_title.append(p)

    # also drop DOI records that duplicate an arXiv record by title
    arxiv_list = list(by_arxiv.values())
    doi_list = []
    for p in by_doi.values():
        dup = False
        for a in arxiv_list:
            if title_similarity(p.get("title") or "", a.get("title") or "") >= 0.92:
                # merge useful fields into arXiv record
                if not a.get("doi") and p.get("doi"):
                    a["doi"] = p["doi"]
                if (a.get("citation_count") or 0) < (p.get("citation_count") or 0):
                    a["citation_count"] = p["citation_count"]
                dup = True
                break
        if not dup:
            doi_list.append(p)

    # title-only vs arxiv/doi
    title_list = []
    for p in by_title:
        dup = False
        for other in arxiv_list + doi_list:
            if title_similarity(p.get("title") or "", other.get("title") or "") >= 0.92:
                dup = True
                break
        if not dup:
            title_list.append(p)

    return arxiv_list + doi_list + title_list


def clean_papers(
    papers: list[dict],
    min_abstract: int = 80,
    drop_noise: bool = True,
) -> list[dict]:
    cleaned = []
    for p in papers:
        title = (p.get("title") or "").strip()
        abstract = (p.get("abstract") or "").strip()
        if not title:
            continue
        if abstract and len(abstract) < min_abstract:
            # allow short abstract if we still have arXiv id + citations
            if not (p.get("arxiv_id") and p.get("citation_count") is not None):
                continue
        if drop_noise and looks_noisy(title, abstract):
            # keep if clearly on-topic keyword still matches
            if topic_relevance(title, abstract, p.get("topics") or [], p.get("concepts") or []) < 0.5:
                continue
        cleaned.append(p)
    return cleaned


def attach_scores(papers: list[dict], scores: dict[str, dict]) -> list[dict]:
    for p in papers:
        key_candidates = [
            p.get("arxiv_id"),
            p.get("internal_id"),
            p.get("doi"),
            p.get("title"),
        ]
        sc = None
        for k in key_candidates:
            if k and str(k) in scores:
                sc = scores[str(k)]
                break
        if not sc:
            # fuzzy title
            nt = normalize_title(p.get("title") or "")
            for k, v in scores.items():
                if normalize_title(v.get("title") or "") == nt:
                    sc = v
                    break
        if sc:
            for field in (
                "citation_velocity",
                "citation_growth",
                "citation_acceleration",
                "trajectory",
                "influence_ratio",
                "score_impact",
                "score_momentum",
                "score_influence",
                "score_recency",
                "score_breadth",
                "score_accessibility",
                "score_overall",
                "age_days",
                "age_months",
            ):
                if field in sc and sc[field] is not None:
                    p[field] = sc[field]
        # ensure relevance score
        p["topic_relevance"] = topic_relevance(
            p.get("title") or "",
            p.get("abstract") or "",
            p.get("topics") or [],
            p.get("concepts") or [],
        )
    return papers


def curate(
    papers: list[dict],
    target: int = 100,
    topic_filter: bool = True,
) -> list[dict]:
    """
    Prefer:
      - higher topic_relevance
      - real arXiv IDs
      - higher momentum / overall score
      - has abstract
    """
    ranked = list(papers)
    if topic_filter:
        # soft filter: prefer relevance > 0, but if too few, relax
        strong = [p for p in ranked if (p.get("topic_relevance") or 0) >= 0.25]
        if len(strong) >= max(40, target // 2):
            ranked = strong

    def sort_key(p: dict) -> tuple:
        return (
            p.get("topic_relevance") or 0,
            1 if p.get("arxiv_id") else 0,
            p.get("score_momentum") or 0,
            p.get("score_overall") or 0,
            p.get("citation_count") or 0,
            len(p.get("abstract") or ""),
        )

    ranked.sort(key=sort_key, reverse=True)

    # diversity: avoid too many near-duplicate titles in the final 100
    selected = []
    for p in ranked:
        if len(selected) >= target:
            break
        if any(
            title_similarity(p.get("title") or "", s.get("title") or "") >= 0.88
            for s in selected
        ):
            continue
        selected.append(p)

    # if still short, fill from remainder without diversity check
    if len(selected) < target:
        ids = {id(s) for s in selected}
        for p in ranked:
            if len(selected) >= target:
                break
            if id(p) in ids:
                continue
            selected.append(p)

    return selected


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "arxiv_id",
        "title",
        "authors",
        "published_date",
        "doi",
        "citation_count",
        "fwci",
        "trajectory",
        "score_overall",
        "score_momentum",
        "topic_relevance",
        "seed_topics",
        "pdf_url",
        "abstract",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = dict(r)
            row["authors"] = "; ".join(row.get("authors") or [])
            row["seed_topics"] = "; ".join(row.get("seed_topics") or [])
            w.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Dedup + merge + curate Phase 1 corpus")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--scores", default=str(SCORES_PATH))
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--min-abstract", type=int, default=80)
    parser.add_argument("--no-topic-filter", action="store_true")
    parser.add_argument("--keep-noise", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 1 — CONSOLIDATE CORPUS")
    print("=" * 70)

    papers = load_from_db(Path(args.db))
    print(f"Loaded from DB:           {len(papers)}")

    scores = load_scores(Path(args.scores))
    print(f"Loaded score records:     {len(scores)}")

    papers = deduplicate(papers)
    print(f"After deduplication:      {len(papers)}")

    papers = clean_papers(
        papers,
        min_abstract=args.min_abstract,
        drop_noise=not args.keep_noise,
    )
    print(f"After cleaning:           {len(papers)}")

    papers = attach_scores(papers, scores)

    # full corpus (one place)
    write_jsonl(OUT_ALL, papers)
    print(f"Wrote full corpus:        {OUT_ALL}  ({len(papers)} papers)")

    curated = curate(
        papers,
        target=args.target,
        topic_filter=not args.no_topic_filter,
    )
    write_json(OUT_JSON, curated)
    write_csv(OUT_CSV, curated)
    print(f"Wrote curated corpus:     {OUT_JSON}  ({len(curated)} papers)")
    print(f"Wrote curated CSV:        {OUT_CSV}")

    # summary
    with_arxiv = sum(1 for p in curated if p.get("arxiv_id"))
    with_cites = sum(1 for p in curated if p.get("citation_count") is not None)
    with_fwci = sum(1 for p in curated if p.get("fwci") is not None)
    with_scores = sum(1 for p in curated if p.get("score_overall") is not None)
    avg_rel = sum(p.get("topic_relevance") or 0 for p in curated) / max(len(curated), 1)

    print()
    print("Curated corpus coverage:")
    print(f"  Real arXiv IDs:     {with_arxiv}/{len(curated)}")
    print(f"  Citation counts:    {with_cites}/{len(curated)}")
    print(f"  FWCI:               {with_fwci}/{len(curated)}")
    print(f"  Impact scores:      {with_scores}/{len(curated)}")
    print(f"  Avg topic_relevance:{avg_rel:.2f}")

    print()
    print("Top 10 in curated set (by topic_relevance, momentum, overall):")
    show = sorted(
        curated,
        key=lambda p: (
            p.get("topic_relevance") or 0,
            p.get("score_momentum") or 0,
            p.get("score_overall") or 0,
        ),
        reverse=True,
    )[:10]
    for i, p in enumerate(show, 1):
        print(
            f"  {i:2}. rel={p.get('topic_relevance', 0):.2f}  "
            f"mom={p.get('score_momentum') or 0:.2f}  "
            f"{(p.get('arxiv_id') or '-'):12}  {(p.get('title') or '')[:55]}"
        )

    print()
    print("Exit criterion check:")
    print(f"  File exists: {OUT_JSON.exists()}")
    print(f"  Loadable:    python -c \"import json; c=json.load(open('{OUT_JSON}')); print(len(c))\"")
    print()
    print("DONE.")


if __name__ == "__main__":
    main()
