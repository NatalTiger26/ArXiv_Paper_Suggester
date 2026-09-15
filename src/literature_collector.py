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

# Semantic Scholar API key (optional but recommended).
# With a key, the 1 request/sec limit is yours alone
# rather than shared across all unauthenticated users.

OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")

SEMANTIC_SCHOLAR_API_KEY = os.getenv(
    "SEMANTIC_SCHOLAR_API_KEY"
)

# Minimum seconds between Semantic Scholar requests.
SEMANTIC_SCHOLAR_RATE_LIMIT = 1.0

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
# ]

RESULTS_PER_SEED = 70

# arXiv API pacing.
ARXIV_DELAY = 5.0 # 3.0

# General HTTP pacing.
HTTP_DELAY = 1 # 0.5


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


def find_semantic_scholar(
    arxiv_id
):

    paper_id = (
        f"ARXIV:{arxiv_id}"
    )

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

    return semantic_get(
        f"paper/{paper_id}",
        {
            "fields": fields
        }
    )


def enrich_semantic_scholar(
    arxiv_id
):

    print(
        f"  Semantic Scholar: {arxiv_id}"
    )

    try:

        data = find_semantic_scholar(
            arxiv_id
        )

    except Exception as exc:

        print(
            f"  Semantic Scholar failed: {exc}"
        )

        return

    if not data:

        print(
            "  Semantic Scholar: not found"
        )

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

def enrich_all():

    conn = get_db()

    papers = conn.execute(
        """
        SELECT *
        FROM papers
        ORDER BY published_date DESC
        """
    ).fetchall()

    conn.close()

    print(
        f"Enriching {len(papers)} papers."
    )

    for index, paper in enumerate(
        papers,
        start=1
    ):

        print()
        print(
            f"[{index}/{len(papers)}] "
            f"{paper['title']}"
        )

        enrich_openalex(
            arxiv_id=paper["arxiv_id"],
            title=paper["title"],
            doi=paper["doi"],
        )

        time.sleep(HTTP_DELAY)

        enrich_semantic_scholar(
            arxiv_id=paper["arxiv_id"]
        )

        time.sleep(HTTP_DELAY)

        enrich_opencitations(
            arxiv_id=paper["arxiv_id"],
            doi=paper["doi"],
        )

        time.sleep(HTTP_DELAY)


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

def run_search(
    query=None,
    max_results=70
):

    init_db()

    client = make_arxiv_client()

    topics = (
        [query]
        if query
        else SEED_TOPICS
    )

    for topic in topics:

        print()
        print(
            "=" * 70
        )

        print(
            f"Searching: {topic}"
        )

        print(
            "=" * 70
        )

        try:

            results = search_arxiv(
                client,
                topic,
                max_results
            )

        except Exception as exc:

            print(
                f"Search failed: {exc}",
                file=sys.stderr
            )

            continue

        print(
            f"Found {len(results)} results."
        )

        for result in results:

            arxiv_id = get_arxiv_id(
                result
            )

            print(
                f"  {arxiv_id} "
                f"{result.title}"
            )

            save_arxiv_result(
                result,
                topic
            )


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

    # --------------------------------------------------------
    # enrich
    # --------------------------------------------------------

    subparsers.add_parser(
        "enrich"
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
            max_results=args.max_results
        )

    elif args.command == "enrich":

        init_db()

        enrich_all()

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