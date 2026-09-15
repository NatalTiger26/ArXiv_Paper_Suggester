#!/usr/bin/env python3
"""
Flexible arXiv research collector.
Designed for building an initial literature corpus from 2–3 seed topics,
but flexible enough to reuse for many kinds of searches.
Install:
    pip install arxiv
Examples:
    # Search default seed topics
    python arxiv_collector.py
    # Search custom topic
    python arxiv_collector.py \
        --query "neural quantum states" \
        --max-results 100
    # Search recent papers
    python arxiv_collector.py \
        --query "quantum many body chaos" \
        --sort submitted \
        --max-results 100
    # Search a specific category
    python arxiv_collector.py \
        --query "neural quantum states" \
        --category "quant-ph"
    # Search several categories
    python arxiv_collector.py \
        --query "interpretability" \
        --category "cs.AI" \
        --category "cs.LG"
    # Restrict by year
    python arxiv_collector.py \
        --query "generative models" \
        --year-from 2023 \
        --year-to 2026
    # Download PDFs
    python arxiv_collector.py \
        --query "neural quantum states" \
        --download-pdfs
    # Download LaTeX source as well
    python arxiv_collector.py \
        --query "neural quantum states" \
        --download-source
    # Save JSON instead of JSONL
    python arxiv_collector.py \
        --query "neural quantum states" \
        --output papers.json
    # Search several seed topics
    python arxiv_collector.py \
        --seed \
        --max-results 70
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional
import arxiv

# ============================================================
# DEFAULT RESEARCH CONFIGURATION
# ============================================================
DEFAULT_SEEDS = [
    "neural quantum states",
    "quantum many-body chaos random matrix theory",
    "interpretability generative models",
]
DEFAULT_RESULTS_PER_SEED = 70
DEFAULT_OUTPUT = "arxiv_candidates.jsonl"
DEFAULT_PDF_DIR = "papers"
DEFAULT_SOURCE_DIR = "sources"

# ============================================================
# CLIENT
# ============================================================
def make_client(
    page_size: int = 100,
    delay_seconds: float = 3.0,
    retries: int = 3,
) -> arxiv.Client:
    """
    Create a reusable arXiv client.
    Reusing the same client across searches is preferable to
    constructing a new client for every query.
    """
    return arxiv.Client(
        page_size=page_size,
        delay_seconds=delay_seconds,
        num_retries=retries,)

# ============================================================
# QUERY HELPERS
# ============================================================
def build_query(
    keywords: Optional[str] = None,
    title: Optional[str] = None,
    author: Optional[str] = None,
    category: Optional[list[str]] = None,
) -> str:
    """
    Build an arXiv query.
    Examples:
        keywords="neural quantum states"
        title="quantum"
        author="Preskill"
        category=["quant-ph", "cond-mat.stat-mech"]
    Multiple categories are OR'ed together.
    """
    parts = []
    if keywords:
        # Search broadly across arXiv metadata.
        parts.append(
            f'all:"{keywords}"')
    if title:
        parts.append(
            f'ti:"{title}"')
    if author:
        parts.append(
            f'au:"{author}"')
    if category:
        category_query = " OR ".join(
            f"cat:{cat}"
            for cat in category)
        parts.append(
            f"({category_query})")
    if not parts:
        return ""
    return " AND ".join(parts)

# ============================================================
# RESULT SERIALIZATION
# ============================================================
def clean_text(text: Optional[str]) -> Optional[str]:
    """
    Normalize whitespace while retaining the actual content.
    """
    if text is None:
        return None
    return re.sub(
        r"\s+",
        " ",
        text).strip()

def get_short_arxiv_id(result: arxiv.Result) -> str:
    """
    Return an arXiv identifier without the version suffix.
    Example:
        2401.12345v2 -> 2401.12345
    """
    arxiv_id = result.get_short_id()
    return re.sub(
        r"v\d+$",
        "",
        arxiv_id)

def result_to_dict(
    result: arxiv.Result,
    seed_topic: Optional[str] = None,
) -> dict:
    """
    Convert an arxiv.Result into a JSON-serializable dictionary.
    We intentionally keep more metadata than the minimum required,
    because it is cheap to preserve it now and useful later.
    """
    return {
        # ----------------------------
        # Identity
        # ----------------------------
        "arxiv_id": get_short_arxiv_id(result),
        "arxiv_id_version": result.get_short_id(),
        "entry_url": result.entry_id,
        # ----------------------------
        # Core bibliographic metadata
        # ----------------------------
        "title": clean_text(result.title),
        "abstract": clean_text(result.summary),
        "authors": [
            author.name
            for author in result.authors
        ],
        "categories": list(result.categories),
        "primary_category": result.primary_category,
        "published_date": (
            result.published.isoformat()
            if result.published
            else None),
        "updated_date": (
            result.updated.isoformat()
            if result.updated
            else None),
        # ----------------------------
        # Publication information
        # ----------------------------
        "comment": clean_text(
            result.comment),
        "journal_reference": clean_text(
            result.journal_ref),
        "doi": result.doi,
        # ----------------------------
        # Download information
        # ----------------------------
        "pdf_url": result.pdf_url,
        "source_url": (
            result.source_url()
            if hasattr(result, "source_url")
            else None),
        # ----------------------------
        # Research bookkeeping
        # ----------------------------
        "seed_topic": seed_topic,
        "year": (
            result.published.year
            if result.published
            else None),
    }

# ============================================================
# SEARCH
# ============================================================
def search(
    client: arxiv.Client,
    query: str,
    max_results: int,
    sort_by: str,
    sort_order: str,
) -> list[arxiv.Result]:
    """
    Execute one arXiv search.
    """
    if sort_by == "relevance":
        criterion = arxiv.SortCriterion.Relevance
    elif sort_by == "submitted":
        criterion = arxiv.SortCriterion.SubmittedDate
    elif sort_by == "updated":
        criterion = arxiv.SortCriterion.LastUpdatedDate
    else:
        raise ValueError(
            f"Unknown sort method: {sort_by}")
    if sort_order == "ascending":
        order = arxiv.SortOrder.Ascending
    else:
        order = arxiv.SortOrder.Descending
    search_request = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=criterion,
        sort_order=order,)
    results = []
    for result in client.results(search_request):
        results.append(result)
    return results

# ============================================================
# DATE FILTERING
# ============================================================
def filter_by_year(
    results: list[arxiv.Result],
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
) -> list[arxiv.Result]:
    filtered = []
    for result in results:
        if result.published is None:
            continue
        year = result.published.year
        if year_from is not None and year < year_from:
            continue
        if year_to is not None and year > year_to:
            continue
        filtered.append(result)
    return filtered

# ============================================================
# DEDUPLICATION
# ============================================================
def deduplicate(
    papers: list[dict],
) -> list[dict]:
    """
    Deduplicate by base arXiv ID.
    This means:
        2401.12345
        2401.12345v2
    are treated as the same paper.
    """
    unique = {}
    for paper in papers:
        arxiv_id = paper["arxiv_id"]
        if arxiv_id not in unique:
            unique[arxiv_id] = paper
        else:
            # Preserve all seed topics that found the paper.
            old_seed = unique[arxiv_id].get(
                "seed_topic")
            new_seed = paper.get(
                "seed_topic")
            seeds = []
            if isinstance(old_seed, list):
                seeds.extend(old_seed)
            elif old_seed:
                seeds.append(old_seed)
            if isinstance(new_seed, list):
                seeds.extend(new_seed)
            elif new_seed:
                seeds.append(new_seed)
            unique[arxiv_id]["seed_topic"] = sorted(
                set(seeds))
    return list(unique.values())

# ============================================================
# DOWNLOADS
# ============================================================
def safe_filename(name: str) -> str:
    """
    Make a filename safe for most operating systems.
    """
    name = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        name)
    name = re.sub(
        r"\s+",
        " ",
        name).strip()
    # Avoid ridiculously long filenames.
    return name[:180]

def download_paper(
    result: arxiv.Result,
    pdf_dir: Optional[Path] = None,
    source_dir: Optional[Path] = None,
):
    """
    Download PDF and/or source archive.
    """
    arxiv_id = get_short_arxiv_id(result)
    if pdf_dir:
        pdf_dir.mkdir(
            parents=True,
            exist_ok=True)
        print(
            f"    Downloading PDF: {arxiv_id}")
        try:
            result.download_pdf(
                dirpath=str(pdf_dir),
                filename=f"{arxiv_id}.pdf",)
        except Exception as exc:
            print(
                f"    PDF download failed: {exc}",
                file=sys.stderr)
    if source_dir:
        source_dir.mkdir(
            parents=True,
            exist_ok=True)
        print(
            f"    Downloading source: {arxiv_id}")
        try:
            result.download_source(
                dirpath=str(source_dir),
                filename=f"{arxiv_id}.tar",)
        except Exception as exc:
            print(
                f"    Source download failed: {exc}",
                file=sys.stderr)

# ============================================================
# OUTPUT
# ============================================================
def save_jsonl(
    papers: list[dict],
    path: Path,
):
    with path.open(
        "w",
        encoding="utf-8") as f:
        for paper in papers:
            f.write(
                json.dumps(
                    paper,
                    ensure_ascii=False,)
                + "\n")

def save_json(
    papers: list[dict],
    path: Path,
):
    with path.open(
        "w",
        encoding="utf-8") as f:
        json.dump(
            papers,
            f,
            indent=2,
            ensure_ascii=False,)

# ============================================================
# STATISTICS
# ============================================================
def print_statistics(
    papers: list[dict],
):
    print()
    print("=" * 70)
    print("COLLECTION STATISTICS")
    print("=" * 70)
    print(
        f"Unique papers: {len(papers)}")
    categories = defaultdict(int)
    for paper in papers:
        for category in paper["categories"]:
            categories[category] += 1
    print()
    print("Top categories:")
    for category, count in sorted(
        categories.items(),
        key=lambda x: x[1],
        reverse=True,)[:15]:
        print(
            f"  {category:20} {count}")
    authors = defaultdict(int)
    for paper in papers:
        for author in paper["authors"]:
            authors[author] += 1
    print()
    print("Most frequent authors:")
    for author, count in sorted(
        authors.items(),
        key=lambda x: x[1],
        reverse=True,)[:15]:
        print(
            f"  {author:40} {count}")

# ============================================================
# ARGUMENT PARSER
# ============================================================
def make_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Flexible arXiv research-paper collector."))
    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------
    parser.add_argument(
        "--query",
        help=(
            "Free-text arXiv query. "
            "Example: 'neural quantum states'"),)
    parser.add_argument(
        "--title",
        help="Search within paper titles.",)
    parser.add_argument(
        "--author",
        help="Search by author.",)
    parser.add_argument(
        "--category",
        action="append",
        help=(
            "Restrict to an arXiv category. "
            "Can be supplied multiple times."),)
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Run the configured DEFAULT_SEEDS.",)
    # --------------------------------------------------------
    # Quantity
    # --------------------------------------------------------
    parser.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_RESULTS_PER_SEED,
        help="Maximum results per search.",)
    # --------------------------------------------------------
    # Sorting
    # --------------------------------------------------------
    parser.add_argument(
        "--sort",
        choices=[
            "relevance",
            "submitted",
            "updated",
        ],
        default="relevance",)
    parser.add_argument(
        "--order",
        choices=[
            "ascending",
            "descending",
        ],
        default="descending",)
    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------
    parser.add_argument(
        "--year-from",
        type=int,
        help="Only keep papers from this year onward.",)
    parser.add_argument(
        "--year-to",
        type=int,
        help="Only keep papers up to this year.",)
    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output JSON/JSONL file.",)
    # --------------------------------------------------------
    # Downloads
    # --------------------------------------------------------
    parser.add_argument(
        "--download-pdfs",
        action="store_true",
        help="Download PDFs.",)
    parser.add_argument(
        "--download-source",
        action="store_true",
        help="Download LaTeX source archives.",)
    parser.add_argument(
        "--pdf-dir",
        default=DEFAULT_PDF_DIR,)
    parser.add_argument(
        "--source-dir",
        default=DEFAULT_SOURCE_DIR,)
    # --------------------------------------------------------
    # API client
    # --------------------------------------------------------
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Delay between API requests.",)
    parser.add_argument(
        "--retries",
        type=int,
        default=3,)
    return parser

# ============================================================
# MAIN
# ============================================================
def main():
    parser = make_parser()
    args = parser.parse_args()
    # --------------------------------------------------------
    # Validate search mode
    # --------------------------------------------------------
    if not args.seed and not any(
        [
            args.query,
            args.title,
            args.author,
            args.category,
        ]):
        # Convenient default for your research task.
        args.seed = True
    # --------------------------------------------------------
    # Client
    # --------------------------------------------------------
    client = make_client(
        delay_seconds=args.delay,
        retries=args.retries,)
    all_papers = []
    # --------------------------------------------------------
    # Build searches
    # --------------------------------------------------------
    searches = []
    if args.seed:
        for seed in DEFAULT_SEEDS:
            searches.append(
                {
                    "name": seed,
                    "query": build_query(
                        keywords=seed,
                        category=args.category,),
                })
    else:
        searches.append(
            {
                "name": args.query
                or args.title
                or args.author
                or "custom",
                "query": build_query(
                    keywords=args.query,
                    title=args.title,
                    author=args.author,
                    category=args.category,),
            })
    # --------------------------------------------------------
    # Execute searches
    # --------------------------------------------------------
    for search_info in searches:
        topic = search_info["name"]
        query = search_info["query"]
        print()
        print("=" * 70)
        print(f"SEARCH: {topic}")
        print("=" * 70)
        print(f"Query: {query}")
        try:
            results = search(
                client=client,
                query=query,
                max_results=args.max_results,
                sort_by=args.sort,
                sort_order=args.order,)
        except Exception as exc:
            print(
                f"Search failed: {exc}",
                file=sys.stderr)
            continue
        results = filter_by_year(
            results,
            year_from=args.year_from,
            year_to=args.year_to,)
        print(
            f"Results after filters: {len(results)}")
        for result in results:
            paper = result_to_dict(
                result,
                seed_topic=topic,)
            all_papers.append(paper)
            # Optional download.
            if (
                args.download_pdfs
                or args.download_source):
                download_paper(
                    result,
                    pdf_dir=(
                        Path(args.pdf_dir)
                        if args.download_pdfs
                        else None),
                    source_dir=(
                        Path(args.source_dir)
                        if args.download_source
                        else None),)
    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------
    papers = deduplicate(
        all_papers)
    # --------------------------------------------------------
    # Sort final corpus
    # --------------------------------------------------------
    papers.sort(
        key=lambda p: (
            p["published_date"] or ""),
        reverse=True,)
    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    output_path = Path(args.output)
    if output_path.suffix.lower() == ".json":
        save_json(
            papers,
            output_path,)
    else:
        save_jsonl(
            papers,
            output_path,)
    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------
    print_statistics(
        papers)
    print()
    print(
        f"Saved {len(papers)} unique papers to "
        f"{output_path}")

if __name__ == "__main__":
    main()
