#!/usr/bin/env python3

"""
FREE-FIRST SCHOLARLY LITERATURE COLLECTOR

Sources
-------
1. arXiv
   - discovery
   - metadata
   - PDF
   - LaTeX/source

2. OpenAlex
   - citation count
   - references
   - concepts/topics
   - venue
   - institutions
   - open-access information

3. Semantic Scholar
   - citation count
   - references
   - citations
   - influential citations
   - paper identifiers
   - SPECTER2 embeddings
   - recommendations

4. OpenCitations
   - independent citation/reference information

Storage
-------
SQLite.

The database lets you:
    - run searches repeatedly
    - deduplicate papers
    - enrich existing papers
    - update citation counts
    - download PDFs later
    - attach your own similarity scores later

INSTALL
-------

    uv pip install arxiv requests

OPTIONAL
--------

Set an OpenAlex API key:

    export OPENALEX_API_KEY="your_key"

A free OpenAlex key is recommended.

Usage examples
--------------

Initialize:

    python literature_collector.py init

Search your seed topics:

    python literature_collector.py search

Search one custom topic:

    python literature_collector.py search \
        --query "neural quantum states"

Search with more results:

    python literature_collector.py search \
        --query "quantum many body chaos" \
        --max-results 100

Enrich all papers:

    python literature_collector.py enrich

Download PDFs:

    python literature_collector.py download

Show statistics:

    python literature_collector.py stats

Export:

    python literature_collector.py export papers.jsonl

List papers:

    python literature_collector.py list

"""


from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests
import arxiv


# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = Path("data/papers.db")

PDF_DIR = Path("data/pdfs")

SOURCE_DIR = Path("data/sources")

OPENALEX_API = "https://api.openalex.org"

SEMANTIC_SCHOLAR_API = (
    "https://api.semanticscholar.org/graph/v1"
)

# API keys (set these in your environment)
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.getenv(
    "SEMANTIC_SCHOLAR_API_KEY"
)

# Minimum seconds between Semantic Scholar requests.
SEMANTIC_SCHOLAR_RATE_LIMIT = 1.1

OPENCITATIONS_API = (
    "https://api.opencitations.net/index/v2"
)

# ------------------------------------------------------------
# PUT YOUR SEED TOPICS HERE
# ------------------------------------------------------------

SEED_TOPICS = [
    "mechanistic interpretability",
    "statistical mechanics of learning",
    "random matrix theory neural networks",
]

# SEED_TOPICS = [
#     "neural quantum states",
#     "interpretability generative models",
#     "mechanistic interpretability",
#     "sparse autoencoders",
#     "world models",
#     "physics-informed neural networks",
#     "Hamiltonian neural networks",
#     "random matrix theory neural networks",
#     "loss landscape",
#     "statistical mechanics of learning",
#     "grokking",
#     "certified robustness",
#     "large language models",
# ]

RESULTS_PER_SEED = 70

# arXiv API pacing (currently blocked with systemic 429s).
ARXIV_DELAY = 5.0

# General HTTP pacing.
HTTP_DELAY = 1.0

# Default discovery sources when arXiv API is unavailable.
# Prefer OpenAlex + Semantic Scholar.
DEFAULT_SEARCH_BACKENDS = ["openalex", "semantic_scholar"]


# ============================================================
# DATABASE
# ============================================================

def get_db():
    """
    Open the SQLite database.
    """

    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    conn = sqlite3.connect(
        DB_PATH
    )

    conn.row_factory = sqlite3.Row

    return conn


def init_db():

    conn = get_db()

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS papers (

            arxiv_id TEXT PRIMARY KEY,

            title TEXT,

            abstract TEXT,

            authors_json TEXT,

            categories_json TEXT,

            primary_category TEXT,

            published_date TEXT,

            updated_date TEXT,

            journal_reference TEXT,

            comment TEXT,

            doi TEXT,

            arxiv_url TEXT,

            pdf_url TEXT,

            source_url TEXT,

            pdf_path TEXT,

            source_path TEXT,

            created_at TEXT,

            last_updated TEXT

        );


        CREATE TABLE IF NOT EXISTS seed_topics (

            arxiv_id TEXT,

            seed_topic TEXT,

            PRIMARY KEY (
                arxiv_id,
                seed_topic
            )

        );


        CREATE TABLE IF NOT EXISTS openalex (

            arxiv_id TEXT PRIMARY KEY,

            openalex_id TEXT,

            cited_by_count INTEGER,

            referenced_works_json TEXT,

            concepts_json TEXT,

            topics_json TEXT,

            institutions_json TEXT,

            authorships_json TEXT,

            primary_location_json TEXT,

            open_access_json TEXT,

            raw_json TEXT,

            fetched_at TEXT

        );


        CREATE TABLE IF NOT EXISTS semantic_scholar (

            arxiv_id TEXT PRIMARY KEY,

            paper_id TEXT,

            corpus_id TEXT,

            citation_count INTEGER,

            influential_citation_count INTEGER,

            reference_count INTEGER,

            citations_json TEXT,

            references_json TEXT,

            embedding_json TEXT,

            url TEXT,

            raw_json TEXT,

            fetched_at TEXT

        );


        CREATE TABLE IF NOT EXISTS opencitations (

            arxiv_id TEXT PRIMARY KEY,

            citation_count INTEGER,

            reference_count INTEGER,

            citations_json TEXT,

            references_json TEXT,

            fetched_at TEXT,

            raw_json TEXT

        );


        CREATE TABLE IF NOT EXISTS similarity (

            arxiv_id_a TEXT,

            arxiv_id_b TEXT,

            text_similarity REAL,

            code_similarity REAL,

            image_similarity REAL,

            overall_similarity REAL,

            calculated_at TEXT,

            PRIMARY KEY (
                arxiv_id_a,
                arxiv_id_b
            )

        );

        """
    )

    conn.commit()

    conn.close()


# ============================================================
# UTILITY
# ============================================================

def now():
    return datetime.utcnow().isoformat()


def clean_text(text: Optional[str]):

    if not text:
        return None

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def json_dump(value):

    return json.dumps(
        value,
        ensure_ascii=False
    )


def get_arxiv_id(result):

    return re.sub(
        r"v\d+$",
        "",
        result.get_short_id()
    )


# ============================================================
# ARXIV
# ============================================================

def make_arxiv_client():

    return arxiv.Client(
        page_size=100,
        delay_seconds=ARXIV_DELAY,
        num_retries=3,
    )


def search_arxiv(
    client,
    query,
    max_results
):

    search = arxiv.Search(
        query=f'all:"{query}"',
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )

    return list(
        client.results(search)
    )


def save_arxiv_result(
    result,
    seed_topic
):

    arxiv_id = get_arxiv_id(result)

    conn = get_db()

    conn.execute(
        """
        INSERT INTO papers (
            arxiv_id,
            title,
            abstract,
            authors_json,
            categories_json,
            primary_category,
            published_date,
            updated_date,
            journal_reference,
            comment,
            doi,
            arxiv_url,
            pdf_url,
            source_url,
            created_at,
            last_updated
        )

        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?
        )

        ON CONFLICT(arxiv_id)
        DO UPDATE SET

            title = excluded.title,
            abstract = excluded.abstract,
            authors_json = excluded.authors_json,
            categories_json = excluded.categories_json,
            primary_category = excluded.primary_category,
            published_date = excluded.published_date,
            updated_date = excluded.updated_date,
            journal_reference = excluded.journal_reference,
            comment = excluded.comment,
            doi = excluded.doi,
            arxiv_url = excluded.arxiv_url,
            pdf_url = excluded.pdf_url,
            source_url = excluded.source_url,
            last_updated = excluded.last_updated
        """,
        (
            arxiv_id,

            clean_text(result.title),

            clean_text(result.summary),

            json_dump([
                author.name
                for author in result.authors
            ]),

            json_dump(
                result.categories
            ),

            result.primary_category,

            result.published.isoformat()
            if result.published
            else None,

            result.updated.isoformat()
            if result.updated
            else None,

            clean_text(
                result.journal_ref
            ),

            clean_text(
                result.comment
            ),

            result.doi,

            result.entry_id,

            result.pdf_url,

            result.source_url(),

            now(),

            now(),
        )
    )

    conn.execute(
        """
        INSERT OR IGNORE INTO seed_topics
        VALUES (?, ?)
        """,
        (
            arxiv_id,
            seed_topic,
        )
    )

    conn.commit()

    conn.close()


# ============================================================
# OPENALEX
# ============================================================

def openalex_headers():

    return {
        "User-Agent":
            "literature-research-tool/1.0"
    }


def openalex_get(
    endpoint,
    params=None
):

    params = params or {}

    key = OPENALEX_API_KEY

    if key:
        params["api_key"] = key

    response = requests.get(
        f"{OPENALEX_API}/{endpoint}",
        params=params,
        headers=openalex_headers(),
        timeout=30,
    )

    if response.status_code == 429:

        print(
            "OpenAlex rate limit reached.",
            file=sys.stderr
        )

        time.sleep(10)

        return None

    response.raise_for_status()

    return response.json()


def find_openalex_by_doi(doi):

    if not doi:
        return None

    return openalex_get(
        f"works/https://doi.org/{doi}"
    )


def find_openalex_by_title(
    title
):

    if not title:
        return None

    data = openalex_get(
        "works",
        {
            "search": title,
            "per_page": 5,
        }
    )

    if not data:
        return None

    results = data.get(
        "results",
        []
    )

    if not results:
        return None

    # Basic title matching.
    normalized_target = normalize_title(
        title
    )

    best = None
    best_score = 0

    for work in results:

        candidate = normalize_title(
            work.get("title", "")
        )

        score = title_overlap(
            normalized_target,
            candidate
        )

        if score > best_score:

            best_score = score
            best = work

    # Avoid attaching a completely unrelated paper.
    if best_score < 0.6:
        return None

    return best


def normalize_title(title):

    title = title.lower()

    title = re.sub(
        r"[^a-z0-9\s]",
        " ",
        title
    )

    return set(
        title.split()
    )


def title_overlap(a, b):

    if not a or not b:
        return 0

    return len(a & b) / max(
        len(a),
        len(b)
    )


def enrich_openalex(
    arxiv_id,
    title,
    doi
):

    print(
        f"  OpenAlex: {arxiv_id}"
    )

    work = None

    try:

        if doi:
            work = find_openalex_by_doi(
                doi
            )

        if work is None:

            work = find_openalex_by_title(
                title
            )

    except Exception as exc:

        print(
            f"  OpenAlex failed: {exc}"
        )

        return

    if not work:

        print(
            "  OpenAlex: not found"
        )

        return

    openalex_id = work.get(
        "id"
    )

    cited_by_count = work.get(
        "cited_by_count"
    )

    referenced_works = work.get(
        "referenced_works",
        []
    )

    concepts = work.get(
        "concepts",
        []
    )

    topics = work.get(
        "topics",
        []
    )

    institutions = []

    for authorship in work.get(
        "authorships",
        []
    ):

        for institution in authorship.get(
            "institutions",
            []
        ):

            institutions.append(
                institution
            )

    primary_location = work.get(
        "primary_location"
    )

    open_access = work.get(
        "open_access"
    )

    conn = get_db()

    conn.execute(
        """
        INSERT OR REPLACE INTO openalex (

            arxiv_id,
            openalex_id,
            cited_by_count,
            referenced_works_json,
            concepts_json,
            topics_json,
            institutions_json,
            authorships_json,
            primary_location_json,
            open_access_json,
            raw_json,
            fetched_at

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            arxiv_id,

            openalex_id,

            cited_by_count,

            json_dump(
                referenced_works
            ),

            json_dump(
                concepts
            ),

            json_dump(
                topics
            ),

            json_dump(
                institutions
            ),

            json_dump(
                work.get("authorships", [])
            ),

            json_dump(
                primary_location
            ),

            json_dump(
                open_access
            ),

            json_dump(
                work
            ),

            now(),
        )
    )

    conn.commit()

    conn.close()


# ============================================================
# SEMANTIC SCHOLAR
# ============================================================

# Tracks the time of the last Semantic Scholar request so
# every call, wherever it's made from, respects the same
# 1 request/sec ceiling.
_last_semantic_scholar_call = 0.0


def semantic_wait():

    global _last_semantic_scholar_call

    elapsed = (
        time.time()
        - _last_semantic_scholar_call
    )

    remaining = (
        SEMANTIC_SCHOLAR_RATE_LIMIT
        - elapsed
    )

    if remaining > 0:
        time.sleep(remaining)

    _last_semantic_scholar_call = time.time()


def semantic_headers():

    headers = {
        "User-Agent":
            "literature-research-tool/1.0"
    }

    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = (
            SEMANTIC_SCHOLAR_API_KEY
        )

    return headers


def semantic_get(
    endpoint,
    params=None
):

    semantic_wait()

    response = requests.get(
        f"{SEMANTIC_SCHOLAR_API}/{endpoint}",
        params=params or {},
        timeout=30,
        headers=semantic_headers(),
    )

    if response.status_code == 429:

        print(
            "Semantic Scholar rate limited."
        )

        # Back off a bit longer than the
        # steady-state interval, then let the
        # caller retry on its own if it wants to.
        time.sleep(
            max(
                SEMANTIC_SCHOLAR_RATE_LIMIT * 5,
                5,
            )
        )

        return None

    response.raise_for_status()

    return response.json()


def _is_real_arxiv_id(arxiv_id: Optional[str]) -> bool:
    """True only for classic arXiv IDs (new or old style)."""
    if not arxiv_id:
        return False
    if arxiv_id.startswith(("doi:", "noarxiv:")):
        return False
    return bool(
        re.match(r"^(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})$", arxiv_id)
    )


def find_semantic_scholar(
    arxiv_id: Optional[str] = None,
    doi: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Look up a paper on Semantic Scholar.
    Tries, in order:
      1. ARXIV:{id}   (only for real arXiv IDs)
      2. DOI:{doi}
      3. title search (best match)
    """
    fields = ",".join([
        "paperId",
        "corpusId",
        "title",
        "abstract",
        "authors",
        "year",
        "citationCount",
        "referenceCount",
        "influentialCitationCount",
        "citations",
        "references",
        "url",
        "externalIds",
        "embedding.specter_v2",
    ])

    # 1. Real arXiv ID
    if _is_real_arxiv_id(arxiv_id):
        data = semantic_get(
            f"paper/ARXIV:{arxiv_id}",
            {"fields": fields},
        )
        if data:
            return data

    # 2. DOI
    if doi:
        clean_doi = doi.replace("https://doi.org/", "").strip()
        data = semantic_get(
            f"paper/DOI:{clean_doi}",
            {"fields": fields},
        )
        if data:
            return data

    # 3. Title search (take top result if title overlap is high)
    if title:
        data = semantic_get(
            "paper/search",
            {
                "query": title,
                "limit": 3,
                "fields": fields,
            },
        )
        if data and data.get("data"):
            target = set(re.sub(r"[^a-z0-9\s]", " ", title.lower()).split())
            best = None
            best_score = 0.0
            for cand in data["data"]:
                cand_title = cand.get("title") or ""
                cand_set = set(re.sub(r"[^a-z0-9\s]", " ", cand_title.lower()).split())
                if not target or not cand_set:
                    continue
                score = len(target & cand_set) / max(len(target), len(cand_set))
                if score > best_score:
                    best_score = score
                    best = cand
            if best and best_score >= 0.6:
                return best

    return None


def enrich_semantic_scholar(
    arxiv_id: Optional[str] = None,
    doi: Optional[str] = None,
    title: Optional[str] = None,
):
    print(f"  Semantic Scholar: {arxiv_id or doi or (title or '')[:40]}")

    try:
        data = find_semantic_scholar(
            arxiv_id=arxiv_id,
            doi=doi,
            title=title,
        )
    except Exception as exc:
        print(f"  Semantic Scholar failed: {exc}")
        return

    if not data:
        print("  Semantic Scholar: not found")
        return

    conn = get_db()

    conn.execute(
        """
        INSERT OR REPLACE INTO semantic_scholar (

            arxiv_id,
            paper_id,
            corpus_id,
            citation_count,
            influential_citation_count,
            reference_count,
            citations_json,
            references_json,
            embedding_json,
            url,
            raw_json,
            fetched_at

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            arxiv_id,

            data.get(
                "paperId"
            ),

            str(
                data.get("corpusId")
            )
            if data.get("corpusId")
            else None,

            data.get(
                "citationCount"
            ),

            data.get(
                "influentialCitationCount"
            ),

            data.get(
                "referenceCount"
            ),

            json_dump(
                data.get(
                    "citations",
                    []
                )
            ),

            json_dump(
                data.get(
                    "references",
                    []
                )
            ),

            json_dump(
                data.get(
                    "embedding"
                )
            ),

            data.get(
                "url"
            ),

            json_dump(
                data
            ),

            now(),
        )
    )

    conn.commit()

    conn.close()


# ============================================================
# OPENCITATIONS
# ============================================================

def opencitations_get(
    endpoint
):

    response = requests.get(
        f"{OPENCITATIONS_API}/{endpoint}",
        timeout=30,
        headers={
            "User-Agent":
                "literature-research-tool/1.0"
        }
    )

    if response.status_code == 404:
        return None

    if response.status_code == 429:

        print(
            "OpenCitations rate limited."
        )

        time.sleep(10)

        return None

    response.raise_for_status()

    return response.json()


def enrich_opencitations(
    arxiv_id,
    doi
):

    # OpenCitations is most reliable when
    # we have a DOI.

    if not doi:
        return

    print(
        f"  OpenCitations: {arxiv_id}"
    )

    doi_identifier = (
        f"doi:{doi}"
    )

    try:

        citation_count = (
            opencitations_get(
                f"citation-count/{doi_identifier}"
            )
        )

        reference_count = (
            opencitations_get(
                f"reference-count/{doi_identifier}"
            )
        )

        citations = (
            opencitations_get(
                f"citations/{doi_identifier}"
            )
        )

        references = (
            opencitations_get(
                f"references/{doi_identifier}"
            )
        )

    except Exception as exc:

        print(
            f"  OpenCitations failed: {exc}"
        )

        return

    conn = get_db()

    conn.execute(
        """
        INSERT OR REPLACE INTO opencitations (

            arxiv_id,
            citation_count,
            reference_count,
            citations_json,
            references_json,
            fetched_at,
            raw_json

        )

        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            arxiv_id,

            extract_count(
                citation_count
            ),

            extract_count(
                reference_count
            ),

            json_dump(
                citations
            ),

            json_dump(
                references
            ),

            now(),

            json_dump({
                "citation_count":
                    citation_count,
                "reference_count":
                    reference_count,
                "citations":
                    citations,
                "references":
                    references,
            }),
        )
    )

    conn.commit()

    conn.close()


def extract_count(data):

    if data is None:
        return None

    if isinstance(data, dict):

        for key in [
            "count",
            "citation_count",
            "reference_count",
        ]:

            if key in data:
                return data[key]

    if isinstance(data, list):
        return len(data)

    return None


# ============================================================
# ENRICH ALL
# ============================================================

def enrich_all(
    only_missing: bool = True,
    skip_openalex: bool = False,
    skip_semantic: bool = False,
    skip_opencitations: bool = False,
):
    """
    Enrich every paper in the DB from OpenAlex, Semantic Scholar, and OpenCitations.

    only_missing=True  → skip a source if we already have a row for that paper
    """
    conn = get_db()

    papers = conn.execute(
        """
        SELECT *
        FROM papers
        ORDER BY published_date DESC
        """
    ).fetchall()

    # Already-enriched IDs
    oa_done = {
        row["arxiv_id"]
        for row in conn.execute("SELECT arxiv_id FROM openalex").fetchall()
    }
    ss_done = {
        row["arxiv_id"]
        for row in conn.execute("SELECT arxiv_id FROM semantic_scholar").fetchall()
    }
    oc_done = {
        row["arxiv_id"]
        for row in conn.execute("SELECT arxiv_id FROM opencitations").fetchall()
    }

    conn.close()

    print(f"Enriching {len(papers)} papers.")
    print(f"  already OpenAlex:        {len(oa_done)}")
    print(f"  already Semantic Scholar:{len(ss_done)}")
    print(f"  already OpenCitations:   {len(oc_done)}")
    print(f"  only_missing={only_missing}")

    for index, paper in enumerate(papers, start=1):
        arxiv_id = paper["arxiv_id"]
        title = paper["title"]
        doi = paper["doi"]

        print()
        print(f"[{index}/{len(papers)}] {title[:70] if title else arxiv_id}")

        # ---- OpenAlex ----
        if not skip_openalex:
            if only_missing and arxiv_id in oa_done:
                print("  OpenAlex: already have")
            else:
                try:
                    enrich_openalex(
                        arxiv_id=arxiv_id,
                        title=title,
                        doi=doi,
                    )
                except Exception as exc:
                    print(f"  OpenAlex error: {exc}")
                time.sleep(HTTP_DELAY)

        # ---- Semantic Scholar ----
        if not skip_semantic:
            if only_missing and arxiv_id in ss_done:
                print("  Semantic Scholar: already have")
            else:
                try:
                    enrich_semantic_scholar(
                        arxiv_id=arxiv_id,
                        doi=doi,
                        title=title,
                    )
                except Exception as exc:
                    print(f"  Semantic Scholar error: {exc}")
                # Be extra polite – SS is strict
                time.sleep(max(SEMANTIC_SCHOLAR_RATE_LIMIT, 1.5))

        # ---- OpenCitations (needs DOI) ----
        if not skip_opencitations:
            if only_missing and arxiv_id in oc_done:
                print("  OpenCitations: already have")
            elif not doi:
                print("  OpenCitations: skip (no DOI)")
            else:
                try:
                    enrich_opencitations(
                        arxiv_id=arxiv_id,
                        doi=doi,
                    )
                except Exception as exc:
                    print(f"  OpenCitations error: {exc}")
                time.sleep(HTTP_DELAY)

    print("\nEnrichment pass finished.")
    stats()


# ============================================================
# DOWNLOAD PDFs
# ============================================================

def download_pdfs():

    init_db()

    PDF_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    client = make_arxiv_client()

    conn = get_db()

    papers = conn.execute(
        """
        SELECT
            arxiv_id,
            pdf_url
        FROM papers
        """
    ).fetchall()

    conn.close()

    for paper in papers:

        arxiv_id = paper["arxiv_id"]

        filename = (
            PDF_DIR /
            f"{arxiv_id}.pdf"
        )

        if filename.exists():

            print(
                f"Already exists: {arxiv_id}"
            )

            continue

        print(
            f"Downloading: {arxiv_id}"
        )

        try:

            search = arxiv.Search(
                id_list=[
                    arxiv_id
                ]
            )

            result = next(
                client.results(search)
            )

            result.download_pdf(
                dirpath=str(
                    PDF_DIR
                ),
                filename=(
                    f"{arxiv_id}.pdf"
                )
            )

            conn = get_db()

            conn.execute(
                """
                UPDATE papers
                SET pdf_path = ?
                WHERE arxiv_id = ?
                """,
                (
                    str(filename),
                    arxiv_id,
                )
            )

            conn.commit()
            conn.close()

        except Exception as exc:

            print(
                f"  Failed: {exc}",
                file=sys.stderr
            )


# ============================================================
# STATISTICS
# ============================================================

def stats():

    conn = get_db()

    total = conn.execute(
        """
        SELECT COUNT(*)
        FROM papers
        """
    ).fetchone()[0]

    with_doi = conn.execute(
        """
        SELECT COUNT(*)
        FROM papers
        WHERE doi IS NOT NULL
        """
    ).fetchone()[0]

    openalex = conn.execute(
        """
        SELECT COUNT(*)
        FROM openalex
        """
    ).fetchone()[0]

    semantic = conn.execute(
        """
        SELECT COUNT(*)
        FROM semantic_scholar
        """
    ).fetchone()[0]

    citations = conn.execute(
        """
        SELECT
            AVG(citation_count),
            MAX(citation_count)
        FROM semantic_scholar
        WHERE citation_count IS NOT NULL
        """
    ).fetchone()

    print()
    print("=" * 70)
    print("LITERATURE CORPUS")
    print("=" * 70)

    print(
        f"Total papers:             {total}"
    )

    print(
        f"With DOI:                 {with_doi}"
    )

    print(
        f"OpenAlex enriched:        {openalex}"
    )

    print(
        f"Semantic Scholar enriched:{semantic}"
    )

    print(
        f"Average citations:        "
        f"{citations[0] or 0:.2f}"
    )

    print(
        f"Maximum citations:        "
        f"{citations[1] or 0}"
    )

    conn.close()


# ============================================================
# LIST
# ============================================================

def list_papers(
    limit=50
):

    conn = get_db()

    papers = conn.execute(
        """
        SELECT
            p.arxiv_id,
            p.title,
            p.published_date,
            s.citation_count
        FROM papers p

        LEFT JOIN semantic_scholar s
        ON p.arxiv_id = s.arxiv_id

        ORDER BY p.published_date DESC

        LIMIT ?
        """,
        (limit,)
    ).fetchall()

    conn.close()

    for paper in papers:

        citations = (
            paper["citation_count"]
            if paper["citation_count"]
            is not None
            else "?"
        )

        print(
            f"{paper['arxiv_id']:16} "
            f"{citations:>5} citations  "
            f"{paper['title']}"
        )


# ============================================================
# EXPORT
# ============================================================

def export_jsonl(
    output
):

    conn = get_db()

    papers = conn.execute(
        """
        SELECT *
        FROM papers
        ORDER BY published_date DESC
        """
    ).fetchall()

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:

        for paper in papers:

            record = dict(paper)

            # Convert JSON columns.
            for key in [
                "authors_json",
                "categories_json",
            ]:

                if record.get(key):

                    record[
                        key.replace(
                            "_json",
                            ""
                        )
                    ] = json.loads(
                        record.pop(key)
                    )

            # Seed topics.
            seeds = conn.execute(
                """
                SELECT seed_topic
                FROM seed_topics
                WHERE arxiv_id = ?
                """,
                (
                    record["arxiv_id"],
                )
            ).fetchall()

            record["seed_topics"] = [
                row["seed_topic"]
                for row in seeds
            ]

            # OpenAlex.
            oa = conn.execute(
                """
                SELECT *
                FROM openalex
                WHERE arxiv_id = ?
                """,
                (
                    record["arxiv_id"],
                )
            ).fetchone()

            if oa:

                record["openalex"] = dict(oa)

            # Semantic Scholar.
            ss = conn.execute(
                """
                SELECT *
                FROM semantic_scholar
                WHERE arxiv_id = ?
                """,
                (
                    record["arxiv_id"],
                )
            ).fetchone()

            if ss:

                record["semantic_scholar"] = dict(ss)

            # OpenCitations.
            oc = conn.execute(
                """
                SELECT *
                FROM opencitations
                WHERE arxiv_id = ?
                """,
                (
                    record["arxiv_id"],
                )
            ).fetchone()

            if oc:

                record["opencitations"] = dict(oc)

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

    conn.close()

    print(
        f"Exported to {output}"
    )


# ============================================================
# SEARCH COMMAND
# ============================================================

def extract_arxiv_id_from_openalex(work: dict) -> Optional[str]:
    """Try to pull a clean arXiv ID from an OpenAlex work."""
    # 1. From ids
    ids = work.get("ids") or {}
    arxiv_url = ids.get("arxiv")
    if arxiv_url:
        m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", arxiv_url)
        if m:
            return m.group(1)

    # 2. From locations / primary_location
    for loc in work.get("locations") or []:
        landing = (loc.get("landing_page_url") or "") + " " + (loc.get("pdf_url") or "")
        m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", landing)
        if m:
            return m.group(1)

    primary = work.get("primary_location") or {}
    landing = (primary.get("landing_page_url") or "") + " " + (primary.get("pdf_url") or "")
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", landing)
    if m:
        return m.group(1)

    # 3. From DOI (10.48550/arXiv.XXXX.XXXXX)
    doi = work.get("doi") or ""
    m = re.search(r"10\.48550/arXiv\.(\d{4}\.\d{4,5})", doi, re.I)
    if m:
        return m.group(1)

    return None


def extract_arxiv_id_from_semantic(paper: dict) -> Optional[str]:
    """Try to pull a clean arXiv ID from a Semantic Scholar paper."""
    external = paper.get("externalIds") or {}
    arxiv_id = external.get("ArXiv") or external.get("arXiv")
    if arxiv_id:
        return re.sub(r"v\d+$", "", str(arxiv_id))

    # Fallback: look in url
    url = paper.get("url") or ""
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", url)
    if m:
        return m.group(1)

    return None


def save_generic_paper(
    *,
    arxiv_id: Optional[str],
    title: str,
    abstract: Optional[str],
    authors: list[str],
    published_date: Optional[str],
    doi: Optional[str],
    pdf_url: Optional[str],
    categories: Optional[list[str]],
    primary_category: Optional[str],
    seed_topic: str,
    source: str,
    extra: Optional[dict] = None,
):
    """
    Insert a paper into the papers table.
    If we have no arXiv ID we synthesize a stable key from the DOI or title hash
    so the rest of the pipeline still works.
    """
    if not arxiv_id:
        if doi:
            # Use a DOI-based surrogate key
            arxiv_id = "doi:" + re.sub(r"[^a-zA-Z0-9.]", "_", doi)[:80]
        else:
            # Last resort: short hash of title
            import hashlib
            h = hashlib.sha1((title or "").encode("utf-8")).hexdigest()[:12]
            arxiv_id = f"noarxiv:{h}"

    conn = get_db()

    conn.execute(
        """
        INSERT INTO papers (
            arxiv_id,
            title,
            abstract,
            authors_json,
            categories_json,
            primary_category,
            published_date,
            updated_date,
            journal_reference,
            comment,
            doi,
            arxiv_url,
            pdf_url,
            source_url,
            created_at,
            last_updated
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(arxiv_id) DO UPDATE SET
            title = excluded.title,
            abstract = excluded.abstract,
            authors_json = excluded.authors_json,
            categories_json = excluded.categories_json,
            primary_category = excluded.primary_category,
            published_date = excluded.published_date,
            doi = COALESCE(excluded.doi, papers.doi),
            pdf_url = COALESCE(excluded.pdf_url, papers.pdf_url),
            last_updated = excluded.last_updated
        """,
        (
            arxiv_id,
            clean_text(title),
            clean_text(abstract),
            json_dump(authors or []),
            json_dump(categories or []),
            primary_category,
            published_date,
            None,
            None,
            None,
            doi,
            f"https://arxiv.org/abs/{arxiv_id}" if not arxiv_id.startswith(("doi:", "noarxiv:")) else None,
            pdf_url,
            None,
            now(),
            now(),
        ),
    )

    conn.execute(
        """
        INSERT OR IGNORE INTO seed_topics
        VALUES (?, ?)
        """,
        (arxiv_id, seed_topic),
    )

    conn.commit()
    conn.close()
    return arxiv_id


def search_openalex(query: str, max_results: int = 50) -> list[dict]:
    """
    Search OpenAlex works by free-text query.
    Returns a list of OpenAlex work dicts.
    """
    if not OPENALEX_API_KEY:
        print(
            "WARNING: OPENALEX_API_KEY not set. "
            "OpenAlex will be heavily rate-limited.",
            file=sys.stderr,
        )

    results = []
    per_page = min(100, max_results)
    page = 1

    while len(results) < max_results:
        params = {
            "search": query,
            "per_page": per_page,
            "page": page,
            "sort": "relevance_score:desc",
        }
        if OPENALEX_API_KEY:
            params["api_key"] = OPENALEX_API_KEY

        try:
            data = openalex_get("works", params)
        except Exception as exc:
            print(f"  OpenAlex search error: {exc}", file=sys.stderr)
            break

        if not data:
            break

        batch = data.get("results") or []
        if not batch:
            break

        results.extend(batch)
        if len(batch) < per_page:
            break
        page += 1
        time.sleep(HTTP_DELAY)

    return results[:max_results]


def search_semantic_scholar(query: str, max_results: int = 50) -> list[dict]:
    """
    Search Semantic Scholar by free-text query.
    Returns a list of paper dicts.
    """
    results = []
    limit = min(100, max_results)
    offset = 0

    fields = ",".join([
        "paperId",
        "externalIds",
        "title",
        "abstract",
        "authors",
        "year",
        "citationCount",
        "influentialCitationCount",
        "referenceCount",
        "url",
        "venue",
        "publicationDate",
        "openAccessPdf",
    ])

    while len(results) < max_results:
        params = {
            "query": query,
            "limit": limit,
            "offset": offset,
            "fields": fields,
        }

        try:
            data = semantic_get("paper/search", params)
        except Exception as exc:
            print(f"  Semantic Scholar search error: {exc}", file=sys.stderr)
            break

        if not data:
            break

        batch = data.get("data") or []
        if not batch:
            break

        results.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
        time.sleep(SEMANTIC_SCHOLAR_RATE_LIMIT)

    return results[:max_results]


def run_search(
    query: Optional[str] = None,
    max_results: int = 70,
    backends: Optional[list[str]] = None,
):
    """
    Discover papers using OpenAlex and/or Semantic Scholar
    (arXiv Atom API is currently returning systemic 429s).
    """
    init_db()

    topics = [query] if query else SEED_TOPICS
    backends = backends or DEFAULT_SEARCH_BACKENDS

    print(f"Using backends: {backends}")
    if not OPENALEX_API_KEY and "openalex" in backends:
        print(
            "Tip: export OPENALEX_API_KEY=your_key  "
            "(free at https://openalex.org/settings/api)",
            file=sys.stderr,
        )
    if not SEMANTIC_SCHOLAR_API_KEY and "semantic_scholar" in backends:
        print(
            "Tip: export SEMANTIC_SCHOLAR_API_KEY=your_key  "
            "(free at https://www.semanticscholar.org/product/api)",
            file=sys.stderr,
        )

    for topic in topics:
        print()
        print("=" * 70)
        print(f"Searching: {topic}")
        print("=" * 70)

        seen_ids: set[str] = set()

        # ---------- OpenAlex ----------
        if "openalex" in backends:
            print("\n[OpenAlex]")
            try:
                works = search_openalex(topic, max_results=max_results)
            except Exception as exc:
                print(f"  OpenAlex failed: {exc}", file=sys.stderr)
                works = []

            print(f"  Retrieved {len(works)} works")

            for work in works:
                title = work.get("title") or work.get("display_name") or ""
                abstract = None
                # OpenAlex sometimes stores abstract as inverted index
                inv = work.get("abstract_inverted_index")
                if inv and isinstance(inv, dict):
                    # Reconstruct simple abstract
                    try:
                        positions = []
                        for word, idxs in inv.items():
                            for i in idxs:
                                positions.append((i, word))
                        positions.sort()
                        abstract = " ".join(w for _, w in positions)
                    except Exception:
                        abstract = None

                authors = []
                for a in work.get("authorships") or []:
                    name = (a.get("author") or {}).get("display_name")
                    if name:
                        authors.append(name)

                arxiv_id = extract_arxiv_id_from_openalex(work)
                doi = work.get("doi")
                if doi and doi.startswith("https://doi.org/"):
                    doi = doi.replace("https://doi.org/", "")

                published = work.get("publication_date") or (
                    str(work.get("publication_year"))
                    if work.get("publication_year")
                    else None
                )

                pdf_url = None
                primary = work.get("primary_location") or {}
                if primary.get("pdf_url"):
                    pdf_url = primary["pdf_url"]
                elif work.get("open_access", {}).get("oa_url"):
                    pdf_url = work["open_access"]["oa_url"]

                key = arxiv_id or doi or title
                if key in seen_ids:
                    continue
                seen_ids.add(key)

                saved_id = save_generic_paper(
                    arxiv_id=arxiv_id,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    published_date=published,
                    doi=doi,
                    pdf_url=pdf_url,
                    categories=None,
                    primary_category=None,
                    seed_topic=topic,
                    source="openalex",
                )
                print(f"  + {saved_id[:20]:20}  {title[:80]}")

                # Immediately store the OpenAlex enrichment we already have
                try:
                    enrich_openalex(saved_id, title, doi)
                except Exception:
                    pass

                time.sleep(0.3)

        # ---------- Semantic Scholar ----------
        if "semantic_scholar" in backends:
            print("\n[Semantic Scholar]")
            try:
                papers = search_semantic_scholar(topic, max_results=max_results)
            except Exception as exc:
                print(f"  Semantic Scholar failed: {exc}", file=sys.stderr)
                papers = []

            print(f"  Retrieved {len(papers)} papers")

            for paper in papers:
                title = paper.get("title") or ""
                abstract = paper.get("abstract")
                authors = [
                    a.get("name") for a in (paper.get("authors") or []) if a.get("name")
                ]
                arxiv_id = extract_arxiv_id_from_semantic(paper)
                external = paper.get("externalIds") or {}
                doi = external.get("DOI")

                published = paper.get("publicationDate")
                if not published and paper.get("year"):
                    published = str(paper["year"])

                pdf_url = None
                oa = paper.get("openAccessPdf") or {}
                if isinstance(oa, dict):
                    pdf_url = oa.get("url")

                key = arxiv_id or doi or title
                if key in seen_ids:
                    continue
                seen_ids.add(key)

                saved_id = save_generic_paper(
                    arxiv_id=arxiv_id,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    published_date=published,
                    doi=doi,
                    pdf_url=pdf_url,
                    categories=None,
                    primary_category=None,
                    seed_topic=topic,
                    source="semantic_scholar",
                )
                print(f"  + {saved_id[:20]:20}  {title[:80]}")

                # Store Semantic Scholar data we already fetched
                try:
                    enrich_semantic_scholar(saved_id)
                except Exception:
                    pass

                time.sleep(SEMANTIC_SCHOLAR_RATE_LIMIT)

        print(f"\nFinished topic: {topic}")


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Free-first scholarly "
            "literature collector."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    # --------------------------------------------------------
    # init
    # --------------------------------------------------------

    subparsers.add_parser(
        "init"
    )

    # --------------------------------------------------------
    # search
    # --------------------------------------------------------

    search_parser = subparsers.add_parser(
        "search"
    )

    search_parser.add_argument(
        "--query"
    )

    search_parser.add_argument(
        "--max-results",
        type=int,
        default=RESULTS_PER_SEED
    )

    search_parser.add_argument(
        "--backend",
        action="append",
        choices=["openalex", "semantic_scholar", "arxiv"],
        help=(
            "Search backend(s). Can be repeated. "
            "Default: openalex + semantic_scholar "
            "(arxiv Atom API is currently rate-limited globally)."
        ),
    )

    # --------------------------------------------------------
    # enrich
    # --------------------------------------------------------

    enrich_parser = subparsers.add_parser(
        "enrich",
        help="Fill OpenAlex / Semantic Scholar / OpenCitations for papers already in the DB",
    )
    enrich_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even if we already have a row for that source",
    )
    enrich_parser.add_argument(
        "--skip-openalex",
        action="store_true",
    )
    enrich_parser.add_argument(
        "--skip-semantic",
        action="store_true",
    )
    enrich_parser.add_argument(
        "--skip-opencitations",
        action="store_true",
    )

    # --------------------------------------------------------
    # download
    # --------------------------------------------------------

    subparsers.add_parser(
        "download"
    )

    # --------------------------------------------------------
    # stats
    # --------------------------------------------------------

    subparsers.add_parser(
        "stats"
    )

    # --------------------------------------------------------
    # list
    # --------------------------------------------------------

    list_parser = subparsers.add_parser(
        "list"
    )

    list_parser.add_argument(
        "--limit",
        type=int,
        default=50
    )

    # --------------------------------------------------------
    # export
    # --------------------------------------------------------

    export_parser = subparsers.add_parser(
        "export"
    )

    export_parser.add_argument(
        "output"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Execute
    # --------------------------------------------------------

    if args.command == "init":

        init_db()

        print(
            f"Initialized {DB_PATH}"
        )

    elif args.command == "search":

        run_search(
            query=args.query,
            max_results=args.max_results,
            backends=args.backend,
        )

    elif args.command == "enrich":

        init_db()

        enrich_all(
            only_missing=not args.force,
            skip_openalex=args.skip_openalex,
            skip_semantic=args.skip_semantic,
            skip_opencitations=args.skip_opencitations,
        )

    elif args.command == "download":

        download_pdfs()

    elif args.command == "stats":

        init_db()

        stats()

    elif args.command == "list":

        init_db()

        list_papers(
            args.limit
        )

    elif args.command == "export":

        init_db()

        export_jsonl(
            args.output
        )


if __name__ == "__main__":
    main()