#!/usr/bin/env python3
"""
Minimal, working literature search using OpenAlex + Semantic Scholar.
No dependency on the blocked arXiv Atom API.

Usage:
    export OPENALEX_API_KEY="your_key"
    export SEMANTIC_SCHOLAR_API_KEY="your_key"   # optional but recommended

    python quick_search.py "large language models" --max 30
    python quick_search.py "mechanistic interpretability" --max 20 --out results.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests

OPENALEX_API = "https://api.openalex.org"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"

OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

HTTP_DELAY = 1.0
SS_DELAY = 1.1


def openalex_search(query: str, max_results: int = 50) -> list[dict]:
    results = []
    per_page = min(100, max_results)
    page = 1

    while len(results) < max_results:
        params: dict[str, Any] = {
            "search": query,
            "per_page": per_page,
            "page": page,
            "sort": "relevance_score:desc",
        }
        if OPENALEX_API_KEY:
            params["api_key"] = OPENALEX_API_KEY

        r = requests.get(
            f"{OPENALEX_API}/works",
            params=params,
            timeout=30,
            headers={"User-Agent": "literature-research-tool/1.0"},
        )
        if r.status_code == 429:
            print("OpenAlex rate limited – sleeping 10 s", file=sys.stderr)
            time.sleep(10)
            continue
        r.raise_for_status()
        data = r.json()
        batch = data.get("results") or []
        if not batch:
            break
        results.extend(batch)
        if len(batch) < per_page:
            break
        page += 1
        time.sleep(HTTP_DELAY)

    return results[:max_results]


def semantic_search(query: str, max_results: int = 50) -> list[dict]:
    results = []
    limit = min(100, max_results)
    offset = 0

    fields = ",".join([
        "paperId", "externalIds", "title", "abstract", "authors",
        "year", "citationCount", "influentialCitationCount",
        "url", "venue", "publicationDate", "openAccessPdf",
    ])

    headers = {"User-Agent": "literature-research-tool/1.0"}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    while len(results) < max_results:
        params = {
            "query": query,
            "limit": limit,
            "offset": offset,
            "fields": fields,
        }
        r = requests.get(
            f"{SEMANTIC_SCHOLAR_API}/paper/search",
            params=params,
            headers=headers,
            timeout=30,
        )
        if r.status_code == 429:
            print("Semantic Scholar rate limited – sleeping 5 s", file=sys.stderr)
            time.sleep(5)
            continue
        r.raise_for_status()
        data = r.json()
        batch = data.get("data") or []
        if not batch:
            break
        results.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
        time.sleep(SS_DELAY)

    return results[:max_results]


def reconstruct_abstract(inv: dict) -> Optional[str]:
    if not inv or not isinstance(inv, dict):
        return None
    try:
        positions = []
        for word, idxs in inv.items():
            for i in idxs:
                positions.append((i, word))
        positions.sort()
        return " ".join(w for _, w in positions)
    except Exception:
        return None


def extract_arxiv_id(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"(?:arxiv\.org/(?:abs|pdf)/|arXiv[:\s]|10\.48550/arXiv\.)(\d{4}\.\d{4,5})", text, re.I)
    if m:
        return m.group(1)
    return None


def normalize_openalex(work: dict) -> dict:
    title = work.get("title") or work.get("display_name") or ""
    abstract = reconstruct_abstract(work.get("abstract_inverted_index") or {})
    authors = [
        (a.get("author") or {}).get("display_name")
        for a in (work.get("authorships") or [])
        if (a.get("author") or {}).get("display_name")
    ]
    doi = work.get("doi") or ""
    if doi.startswith("https://doi.org/"):
        doi = doi.replace("https://doi.org/", "")

    arxiv_id = None
    ids = work.get("ids") or {}
    if ids.get("arxiv"):
        arxiv_id = extract_arxiv_id(ids["arxiv"])
    if not arxiv_id:
        arxiv_id = extract_arxiv_id(doi)
    if not arxiv_id:
        for loc in work.get("locations") or []:
            arxiv_id = extract_arxiv_id(
                (loc.get("landing_page_url") or "") + " " + (loc.get("pdf_url") or "")
            )
            if arxiv_id:
                break

    pdf_url = None
    primary = work.get("primary_location") or {}
    if primary.get("pdf_url"):
        pdf_url = primary["pdf_url"]
    elif (work.get("open_access") or {}).get("oa_url"):
        pdf_url = work["open_access"]["oa_url"]

    return {
        "source": "openalex",
        "openalex_id": work.get("id"),
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "doi": doi or None,
        "arxiv_id": arxiv_id,
        "year": work.get("publication_year"),
        "published": work.get("publication_date"),
        "cited_by_count": work.get("cited_by_count"),
        "pdf_url": pdf_url,
        "url": work.get("id"),
    }


def normalize_semantic(paper: dict) -> dict:
    external = paper.get("externalIds") or {}
    arxiv_id = external.get("ArXiv") or external.get("arXiv")
    if arxiv_id:
        arxiv_id = re.sub(r"v\d+$", "", str(arxiv_id))
    doi = external.get("DOI")

    pdf_url = None
    oa = paper.get("openAccessPdf") or {}
    if isinstance(oa, dict):
        pdf_url = oa.get("url")

    return {
        "source": "semantic_scholar",
        "paper_id": paper.get("paperId"),
        "title": paper.get("title"),
        "abstract": paper.get("abstract"),
        "authors": [a.get("name") for a in (paper.get("authors") or []) if a.get("name")],
        "doi": doi,
        "arxiv_id": arxiv_id,
        "year": paper.get("year"),
        "published": paper.get("publicationDate"),
        "cited_by_count": paper.get("citationCount"),
        "influential_citation_count": paper.get("influentialCitationCount"),
        "pdf_url": pdf_url,
        "url": paper.get("url"),
        "venue": paper.get("venue"),
    }


def main():
    parser = argparse.ArgumentParser(description="Quick literature search via OpenAlex + Semantic Scholar")
    parser.add_argument("query", help="Search query, e.g. 'large language models'")
    parser.add_argument("--max", type=int, default=30, help="Max results per backend")
    parser.add_argument("--out", default="data/quick_results.jsonl", help="Output JSONL path")
    parser.add_argument("--backend", action="append", choices=["openalex", "semantic_scholar"],
                        help="Which backends to use (default: both)")
    args = parser.parse_args()

    backends = args.backend or ["openalex", "semantic_scholar"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_papers = []
    seen = set()

    if "openalex" in backends:
        print(f"[OpenAlex] searching: {args.query}")
        if not OPENALEX_API_KEY:
            print("  WARNING: OPENALEX_API_KEY not set – you will hit free-tier limits quickly", file=sys.stderr)
        works = openalex_search(args.query, args.max)
        print(f"  got {len(works)} works")
        for w in works:
            rec = normalize_openalex(w)
            key = rec.get("arxiv_id") or rec.get("doi") or rec.get("title")
            if key and key not in seen:
                seen.add(key)
                all_papers.append(rec)
                print(f"  + { (rec.get('arxiv_id') or '')[:16]:16}  {rec['title'][:70]}")

    if "semantic_scholar" in backends:
        print(f"\n[Semantic Scholar] searching: {args.query}")
        if not SEMANTIC_SCHOLAR_API_KEY:
            print("  WARNING: SEMANTIC_SCHOLAR_API_KEY not set – shared rate limit applies", file=sys.stderr)
        papers = semantic_search(args.query, args.max)
        print(f"  got {len(papers)} papers")
        for p in papers:
            rec = normalize_semantic(p)
            key = rec.get("arxiv_id") or rec.get("doi") or rec.get("title")
            if key and key not in seen:
                seen.add(key)
                all_papers.append(rec)
                print(f"  + { (rec.get('arxiv_id') or '')[:16]:16}  {(rec.get('title') or '')[:70]}")

    with out_path.open("w", encoding="utf-8") as f:
        for rec in all_papers:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(all_papers)} unique papers → {out_path}")


if __name__ == "__main__":
    main()
