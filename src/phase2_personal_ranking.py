#!/usr/bin/env python3
"""
Phase 2 — Personal Ground-Truth Ranking

1) Sample 45–50 papers from data/corpus_100.json (stratified by year + topic)
2) Rank them interactively (1–5 scale) OR export a blank sheet to fill offline
3) Save rankings/personal_importance.csv  (DO NOT change later)
4) Also write rankings/personal_importance_binary.csv  (high = score >= threshold)
5) Print top-5 / bottom-5 for sanity check

Usage:
  # Step A — create the held-out ranking set
  python src/phase2_personal_ranking.py sample

  # Step B — rank interactively (recommended: one focused sitting)
  python src/phase2_personal_ranking.py rank

  # OR fill rankings/to_rank.csv offline, then:
  python src/phase2_personal_ranking.py import-scores rankings/to_rank.csv

  # Step C — freeze + binary + sanity check
  python src/phase2_personal_ranking.py finalize --threshold 4

  # Inspect anytime
  python src/phase2_personal_ranking.py show
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


CORPUS_PATH = Path("data/corpus_100.json")
RANK_DIR = Path("rankings")
SAMPLE_PATH = RANK_DIR / "ranking_sample.json"
BLANK_CSV = RANK_DIR / "to_rank.csv"
SCORES_CSV = RANK_DIR / "personal_importance.csv"
BINARY_CSV = RANK_DIR / "personal_importance_binary.csv"
NOTES_PATH = RANK_DIR / "ranking_notes.txt"

DEFAULT_N = 48
SEED = 42  # reproducible sample


def load_corpus(path: Path = CORPUS_PATH) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Corpus not found: {path}\nRun Phase 1 consolidate first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("corpus_100.json must be a JSON array")
    return data


def paper_key(p: dict) -> str:
    """Stable unique key for ranking file."""
    if p.get("arxiv_id"):
        return str(p["arxiv_id"])
    if p.get("doi"):
        return f"doi:{p['doi']}"
    if p.get("internal_id"):
        return str(p["internal_id"])
    title = re.sub(r"\s+", " ", (p.get("title") or "")[:80])
    return f"title:{title}"


def year_of(p: dict) -> int:
    d = p.get("published_date") or ""
    m = re.match(r"^(\d{4})", str(d))
    if m:
        return int(m.group(1))
    return 0


def topic_bucket(p: dict) -> str:
    blob = " ".join(
        [
            p.get("title") or "",
            " ".join(p.get("seed_topics") or []),
            " ".join(p.get("topics") or []),
        ]
    ).lower()
    if "mechanistic" in blob or "interpretab" in blob or "circuit" in blob:
        return "mechanistic_interpretability"
    if "random matrix" in blob or "marchenko" in blob or "eigenvalue" in blob:
        return "random_matrix"
    if "statistical mechanics" in blob or "stat-mech" in blob or "thermodynamic" in blob:
        return "stat_mech_learning"
    return "other"


def stratified_sample(papers: list[dict], n: int = DEFAULT_N, seed: int = SEED) -> list[dict]:
    """
    Stratify roughly by year band × topic bucket, then fill to n.
    """
    rng = random.Random(seed)

    def year_band(y: int) -> str:
        if y >= 2024:
            return "2024+"
        if y >= 2021:
            return "2021-2023"
        if y >= 2018:
            return "2018-2020"
        if y > 0:
            return "pre-2018"
        return "unknown"

    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in papers:
        buckets[(year_band(year_of(p)), topic_bucket(p))].append(p)

    # shuffle within buckets
    for b in buckets.values():
        rng.shuffle(b)

    # proportional allocation
    total = len(papers)
    alloc: dict[tuple[str, str], int] = {}
    for key, items in buckets.items():
        alloc[key] = max(1, round(n * len(items) / total)) if items else 0

    # adjust to exactly n
    chosen: list[dict] = []
    used_keys = set()

    def add_from(key, k):
        for p in buckets[key]:
            if len(chosen) >= n:
                break
            pk = paper_key(p)
            if pk in used_keys:
                continue
            chosen.append(p)
            used_keys.add(pk)
            k -= 1
            if k <= 0:
                break

    # first pass: proportional
    for key, k in sorted(alloc.items(), key=lambda x: -len(buckets[x[0]])):
        add_from(key, k)

    # fill remainder from leftover
    leftover = [p for p in papers if paper_key(p) not in used_keys]
    rng.shuffle(leftover)
    for p in leftover:
        if len(chosen) >= n:
            break
        chosen.append(p)
        used_keys.add(paper_key(p))

    rng.shuffle(chosen)
    return chosen[:n]


def cmd_sample(args: argparse.Namespace) -> None:
    RANK_DIR.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    sample = stratified_sample(corpus, n=args.n, seed=args.seed)

    SAMPLE_PATH.write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")

    # blank CSV for offline ranking
    with BLANK_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "paper_id",
                "arxiv_id",
                "title",
                "year",
                "topic_bucket",
                "score_1_to_5",
                "notes",
            ],
        )
        w.writeheader()
        for p in sample:
            w.writerow(
                {
                    "paper_id": paper_key(p),
                    "arxiv_id": p.get("arxiv_id") or "",
                    "title": p.get("title") or "",
                    "year": year_of(p) or "",
                    "topic_bucket": topic_bucket(p),
                    "score_1_to_5": "",  # YOU fill this
                    "notes": "",
                }
            )

    # stratification summary
    by_topic = defaultdict(int)
    by_year = defaultdict(int)
    for p in sample:
        by_topic[topic_bucket(p)] += 1
        by_year[year_of(p) or "unknown"] += 1

    print(f"Sampled {len(sample)} papers → {SAMPLE_PATH}")
    print(f"Blank ranking sheet     → {BLANK_CSV}")
    print()
    print("Topic mix:")
    for k, v in sorted(by_topic.items(), key=lambda x: -x[1]):
        print(f"  {k:30} {v}")
    print("Year mix:")
    for k, v in sorted(by_year.items(), key=lambda x: str(x[0])):
        print(f"  {k}: {v}")
    print()
    print("Next:")
    print("  python src/phase2_personal_ranking.py rank")
    print("  # or fill rankings/to_rank.csv then:")
    print("  python src/phase2_personal_ranking.py import-scores rankings/to_rank.csv")


def load_sample() -> list[dict]:
    if not SAMPLE_PATH.exists():
        raise SystemExit("No sample yet. Run: python src/phase2_personal_ranking.py sample")
    return json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))


def cmd_rank(args: argparse.Namespace) -> None:
    """
    Interactive 1–5 ranking in one sitting.
    Scale:
      5 = core / must-read for my research
      4 = high importance
      3 = useful / relevant
      2 = peripheral
      1 = low importance for me
    """
    sample = load_sample()
    print("=" * 70)
    print("PERSONAL IMPORTANCE RANKING")
    print("=" * 70)
    print(
        """
Scale (keep this fixed for the whole session):
  5 = Core / must-read for MY research
  4 = High importance
  3 = Useful / relevant
  2 = Peripheral
  1 = Low importance for me

Commands during ranking:
  1-5     assign score
  s       skip for now (come back later)
  n       short note then score prompt again
  q       quit and save progress
  ?       show this help
"""
    )

    # resume if partial scores exist
    existing: dict[str, dict] = {}
    if SCORES_CSV.exists():
        with SCORES_CSV.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[row["paper_id"]] = row
        print(f"Resuming — {len(existing)} already scored.\n")

    results = dict(existing)
    pending = [p for p in sample if paper_key(p) not in results or not results[paper_key(p)].get("score_1_to_5")]

    print(f"Remaining to score: {len(pending)} / {len(sample)}\n")

    notes_lines = []
    if NOTES_PATH.exists():
        notes_lines = NOTES_PATH.read_text(encoding="utf-8").splitlines()

    i = 0
    while i < len(pending):
        p = pending[i]
        pk = paper_key(p)
        abstract = (p.get("abstract") or "").strip().replace("\n", " ")
        if len(abstract) > 2000:
            abstract = abstract[:2000] + "…"

        print("-" * 70)
        print(f"[{i+1}/{len(pending)}]  {p.get('title') or '(no title)'}")
        print(f"  id:    {pk}")
        print(f"  year:  {year_of(p) or '?'}   topic: {topic_bucket(p)}")
        authors = p.get("authors") or []
        if authors:
            print(f"  authors: {', '.join(authors[:5])}{'…' if len(authors) > 5 else ''}")
        if abstract:
            print(f"  abs: {abstract}")
        cites = p.get("citation_count")
        if cites is not None:
            print(f"  cites: {cites}   fwci: {p.get('fwci')}")
        print()

        note = ""
        while True:
            ans = input("Score 1-5 (or s/n/q/?): ").strip().lower()
            if ans == "?":
                print("1-5 = score | s = skip | n = note | q = quit+save")
                continue
            if ans == "q":
                _save_progress(sample, results)
                print("Progress saved. Re-run `rank` to continue.")
                return
            if ans == "s":
                i += 1
                break
            if ans == "n":
                note = input("  Note (1 sentence): ").strip()
                continue
            if ans in {"1", "2", "3", "4", "5"}:
                results[pk] = {
                    "paper_id": pk,
                    "arxiv_id": p.get("arxiv_id") or "",
                    "title": p.get("title") or "",
                    "year": year_of(p) or "",
                    "topic_bucket": topic_bucket(p),
                    "score_1_to_5": ans,
                    "notes": note,
                }
                if note:
                    notes_lines.append(f"{pk}\t{note}")
                i += 1
                break
            print("  Invalid. Enter 1-5, s, n, q, or ?")

        # autosave every 5
        if len(results) % 5 == 0:
            _save_progress(sample, results)
            NOTES_PATH.write_text("\n".join(notes_lines) + ("\n" if notes_lines else ""), encoding="utf-8")

    _save_progress(sample, results)
    NOTES_PATH.write_text("\n".join(notes_lines) + ("\n" if notes_lines else ""), encoding="utf-8")
    print()
    print(f"All scored that were pending. Saved → {SCORES_CSV}")
    print("Next: python src/phase2_personal_ranking.py finalize")


def _save_progress(sample: list[dict], results: dict[str, dict]) -> None:
    RANK_DIR.mkdir(parents=True, exist_ok=True)
    # keep sample order
    rows = []
    for p in sample:
        pk = paper_key(p)
        if pk in results:
            rows.append(results[pk])
        else:
            rows.append(
                {
                    "paper_id": pk,
                    "arxiv_id": p.get("arxiv_id") or "",
                    "title": p.get("title") or "",
                    "year": year_of(p) or "",
                    "topic_bucket": topic_bucket(p),
                    "score_1_to_5": "",
                    "notes": "",
                }
            )
    with SCORES_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "paper_id",
                "arxiv_id",
                "title",
                "year",
                "topic_bucket",
                "score_1_to_5",
                "notes",
            ],
        )
        w.writeheader()
        w.writerows(rows)


def cmd_import_scores(args: argparse.Namespace) -> None:
    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"Not found: {path}")
    sample = load_sample()
    by_id = {paper_key(p): p for p in sample}

    imported = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = (row.get("paper_id") or "").strip()
            score = (row.get("score_1_to_5") or "").strip()
            if not pid:
                continue
            if score and score not in {"1", "2", "3", "4", "5"}:
                print(f"Skipping invalid score for {pid}: {score}")
                continue
            p = by_id.get(pid) or {}
            imported[pid] = {
                "paper_id": pid,
                "arxiv_id": row.get("arxiv_id") or p.get("arxiv_id") or "",
                "title": row.get("title") or p.get("title") or "",
                "year": row.get("year") or year_of(p) or "",
                "topic_bucket": row.get("topic_bucket") or topic_bucket(p) if p else "",
                "score_1_to_5": score,
                "notes": row.get("notes") or "",
            }

    _save_progress(sample, imported)
    scored = sum(1 for r in imported.values() if r.get("score_1_to_5"))
    print(f"Imported {scored} scores → {SCORES_CSV}")
    missing = len(sample) - scored
    if missing:
        print(f"Still missing scores for {missing} papers. Fill them then re-import or use `rank`.")
    else:
        print("All sample papers scored. Run: python src/phase2_personal_ranking.py finalize")


def cmd_finalize(args: argparse.Namespace) -> None:
    if not SCORES_CSV.exists():
        raise SystemExit("No scores file. Run `rank` or `import-scores` first.")

    rows = []
    with SCORES_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    missing = [r for r in rows if not (r.get("score_1_to_5") or "").strip()]
    if missing:
        print(f"WARNING: {len(missing)} papers still unscored.")
        for r in missing[:5]:
            print(f"  - {r.get('paper_id')}: {r.get('title', '')[:50]}")
        if not args.allow_partial:
            raise SystemExit("Score all papers (or pass --allow-partial).")

    scored = [r for r in rows if (r.get("score_1_to_5") or "").strip() in {"1", "2", "3", "4", "5"}]
    for r in scored:
        r["score_1_to_5"] = int(r["score_1_to_5"])

    # freeze main ranking (already on disk; rewrite sorted by score desc then title)
    scored_sorted = sorted(scored, key=lambda r: (-r["score_1_to_5"], r.get("title") or ""))
    with SCORES_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "paper_id",
                "arxiv_id",
                "title",
                "year",
                "topic_bucket",
                "score_1_to_5",
                "notes",
            ],
        )
        w.writeheader()
        for r in scored_sorted:
            w.writerow(
                {
                    "paper_id": r["paper_id"],
                    "arxiv_id": r.get("arxiv_id") or "",
                    "title": r.get("title") or "",
                    "year": r.get("year") or "",
                    "topic_bucket": r.get("topic_bucket") or "",
                    "score_1_to_5": r["score_1_to_5"],
                    "notes": r.get("notes") or "",
                }
            )

    # binary version
    thr = args.threshold
    with BINARY_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "paper_id",
                "arxiv_id",
                "title",
                "score_1_to_5",
                "high_importance",
            ],
        )
        w.writeheader()
        for r in scored_sorted:
            w.writerow(
                {
                    "paper_id": r["paper_id"],
                    "arxiv_id": r.get("arxiv_id") or "",
                    "title": r.get("title") or "",
                    "score_1_to_5": r["score_1_to_5"],
                    "high_importance": 1 if r["score_1_to_5"] >= thr else 0,
                }
            )

    high = sum(1 for r in scored_sorted if r["score_1_to_5"] >= thr)
    print(f"Frozen ranking → {SCORES_CSV}  ({len(scored_sorted)} papers)")
    print(f"Binary labels  → {BINARY_CSV}  (high_importance = score >= {thr}: {high} papers)")
    print()
    print("Sanity check — TOP 5 (highest personal importance):")
    for i, r in enumerate(scored_sorted[:5], 1):
        print(f"  {i}. [{r['score_1_to_5']}] {(r.get('title') or '')[:65]}")
    print()
    print("Sanity check — BOTTOM 5 (lowest personal importance):")
    for i, r in enumerate(scored_sorted[-5:], 1):
        print(f"  {i}. [{r['score_1_to_5']}] {(r.get('title') or '')[:65]}")
    print()
    print("Distribution:")
    for s in range(5, 0, -1):
        c = sum(1 for r in scored_sorted if r["score_1_to_5"] == s)
        print(f"  score {s}: {c}")
    print()
    print("*** Do not change rankings/personal_importance.csv after this. ***")
    print("Phase 2 exit criterion: fixed, reproducible personal ranking file exists.")


def cmd_show(args: argparse.Namespace) -> None:
    if not SCORES_CSV.exists():
        raise SystemExit("No rankings yet.")
    rows = list(csv.DictReader(SCORES_CSV.open(encoding="utf-8")))
    scored = [r for r in rows if (r.get("score_1_to_5") or "").strip()]
    print(f"{len(scored)} scored / {len(rows)} in file")
    for r in scored[:10]:
        print(f"  [{r.get('score_1_to_5')}] {(r.get('title') or '')[:65]}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2 personal ground-truth ranking")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sample = sub.add_parser("sample", help="Create stratified 45–50 paper sample")
    p_sample.add_argument("--n", type=int, default=DEFAULT_N)
    p_sample.add_argument("--seed", type=int, default=SEED)

    sub.add_parser("rank", help="Interactive 1–5 ranking session")

    p_imp = sub.add_parser("import-scores", help="Import filled to_rank.csv")
    p_imp.add_argument("csv", nargs="?", default=str(BLANK_CSV))

    p_fin = sub.add_parser("finalize", help="Freeze ranking + binary labels + sanity check")
    p_fin.add_argument("--threshold", type=int, default=4, help="high_importance if score >= this")
    p_fin.add_argument("--allow-partial", action="store_true")

    sub.add_parser("show", help="Show current scores")

    args = parser.parse_args()
    if args.cmd == "sample":
        cmd_sample(args)
    elif args.cmd == "rank":
        cmd_rank(args)
    elif args.cmd == "import-scores":
        cmd_import_scores(args)
    elif args.cmd == "finalize":
        cmd_finalize(args)
    elif args.cmd == "show":
        cmd_show(args)


if __name__ == "__main__":
    main()
