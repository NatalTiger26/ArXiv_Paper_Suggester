import arxiv
import json
import os
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
QUERY = "large language models"
MAX_RESULTS = 50
OUTPUT_FILE = Path("data/raw_candidates.json")
TEMP_FILE = Path("data/raw_candidates.tmp.json")

# arXiv asks clients to avoid excessive requests.
# A conservative delay is better than getting rate-limited.
REQUEST_DELAY = 4.0

# Retry configuration
MAX_RETRIES = 8
INITIAL_BACKOFF = 5
MAX_BACKOFF = 300

# ============================================================
# HELPERS
# ============================================================
def load_existing_papers():
    """
    Load previously saved papers so the script can resume
    after a crash/interruption.
    """
    if not OUTPUT_FILE.exists():
        return []
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            print(f"Loaded {len(data)} existing papers.")
            return data
        print("Existing file is not a list. Starting fresh.")
        return []
    except (json.JSONDecodeError, OSError) as e:
        print(f"Could not read existing output: {e}")
        print("Starting with an empty dataset.")
        return []

def save_papers(papers):
    """
    Atomically save papers.
    We first write to a temporary file and then replace the
    real file. This prevents a crash during json.dump() from
    leaving the main JSON file half-written.
    """
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TEMP_FILE, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    # Atomic replacement
    os.replace(TEMP_FILE, OUTPUT_FILE)

def paper_to_dict(paper):
    """
    Convert an arxiv.Result object into a JSON-serializable dict.
    """
    return {
        "title": paper.title,
        "authors": [a.name for a in paper.authors],
        "abstract": paper.summary,
        "arxiv_id": paper.entry_id,
        "published": (paper.published.isoformat() if paper.published else None),
        "updated": (paper.updated.isoformat() if paper.updated else None),
        "categories": paper.categories,
        "primary_category": paper.primary_category,
        "pdf_url": paper.pdf_url,
        "doi": paper.doi,
        "journal": paper.journal_ref,
        "comments": paper.comment,
    }

# ============================================================
# MAIN
# ============================================================
def main():
    # --------------------------------------------------------
    # Load previous progress
    # --------------------------------------------------------
    papers = load_existing_papers()
    existing_ids = {paper["arxiv_id"] for paper in papers if "arxiv_id" in paper}
    print(f"Already have {len(existing_ids)} unique papers.")
    
    # --------------------------------------------------------
    # arXiv client
    # --------------------------------------------------------
    client = arxiv.Client(
        page_size=20,
        delay_seconds=REQUEST_DELAY,
        num_retries=0,
    )

    '''
    in query field
    ti:   title
    au:   author
    abs:  abstract
    cat:  category
    all:  all searchable fields
    submittedDate: [YYYYMMDDHHMM TO YYYYMMDDHHMM]
    The API supports AND, OR, and ANDNOT
    if searched by the id_list field
    id_list=["1706.03762"]

    we can also sort things for eg by 
    sort_by=arxiv.SortCriterion.SubmittedDate,
    EG - 
        Relevance = "relevance"
        LastUpdatedDate = "lastUpdatedDate"
        SubmittedDate = "submittedDate"

    sort_order=arxiv.SortOrder.Descending,
    '''

    search = arxiv.Search(
        query=QUERY,
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )
    try:
        results = client.results(search)
        for paper in results:
            arxiv_id = paper.entry_id
            # ----------------------------------------------
            # Skip duplicates when resuming
            # ----------------------------------------------
            if arxiv_id in existing_ids:
                print(f"Skipping existing: {arxiv_id}")
                continue
            # ----------------------------------------------
            # Convert paper
            # ----------------------------------------------
            try:
                record = paper_to_dict(paper)
                papers.append(record)
                existing_ids.add(arxiv_id)
                # ------------------------------------------
                # SAVE IMMEDIATELY
                # ------------------------------------------
                save_papers(papers)
                print(f"[{len(papers)}/{MAX_RESULTS}] Saved: {paper.title[:100]}")
            except Exception as e:
                print(f"Failed to process {arxiv_id}: {e}")
                # Don't kill the entire job because one
                # paper has malformed/unexpected data.
                continue
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        print(f"Progress saved: {len(papers)} papers.")
        save_papers(papers)
        return
    except Exception as e:
        print(f"\nRequest failed: {type(e).__name__}: {e}")
        print("Attempting to preserve current progress...")
        save_papers(papers)
        print(f"Progress saved: {len(papers)} papers.")
        raise
    # --------------------------------------------------------
    # Final save
    # --------------------------------------------------------
    save_papers(papers)
    print(f"\nDone. Saved {len(papers)} unique papers to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()