#!/usr/bin/env python3
"""
Derive paper importance / momentum / influence signals from data
already stored by literature_collector_v2.py.

No extra API calls required for the core metrics (reads SQLite only).

Usage:
    python src/compute_impact_scores.py
    python src/compute_impact_scores.py --out data/paper_scores.jsonl
    python src/compute_impact_scores.py --db data/papers.db --top 20
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


DB_PATH = Path("data/papers.db")
OUT_PATH = Path("data/paper_scores.jsonl")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = str(s).strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%Y",
    ):
        try:
            if fmt == "%Y":
                return datetime(int(s[:4]), 1, 1, tzinfo=timezone.utc)
            dt = datetime.strptime(s.replace("Z", "+00:00")[:19], fmt.replace("%z", ""))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            continue
    # fallback year
    m = re.match(r"^(\d{4})", s)
    if m:
        return datetime(int(m.group(1)), 1, 1, tzinfo=timezone.utc)
    return None


def months_between(a: datetime, b: datetime) -> float:
    days = abs((b - a).total_seconds()) / 86400.0
    return max(days / 30.44, 0.25)  # floor at ~1 week equivalent


def safe_div(n: float, d: float, default: float = 0.0) -> float:
    if d is None or d == 0:
        return default
    return n / d


def load_json(s: Optional[str]) -> Any:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def trajectory_label(counts_by_year: list[dict], total: int) -> str:
    """Classify citation trajectory."""
    if not counts_by_year or total is None:
        return "unknown"
    years = sorted(
        [(int(x["year"]), int(x.get("cited_by_count") or 0)) for x in counts_by_year],
        key=lambda t: t[0],
    )
    if len(years) < 2:
        if total and total > 20:
            return "growing"
        return "new_or_sparse"

    recent = years[-1][1]
    prev = years[-2][1] if len(years) >= 2 else 0
    older = sum(c for _, c in years[:-2]) if len(years) > 2 else 0

    if prev == 0 and recent > 5:
        return "exploding"
    if prev > 0 and recent / prev >= 2.0 and recent >= 5:
        return "exploding"
    if prev > 0 and recent / prev >= 1.3:
        return "growing"
    if recent >= prev * 0.85 and recent >= 3:
        return "stable"
    if recent < prev * 0.5 and older > recent:
        return "declining"
    if total < 5:
        return "dormant"
    return "stable"


def compute_for_paper(row: dict, oa: Optional[dict], ss: Optional[dict], oc: Optional[dict]) -> dict:
    title = row.get("title")
    arxiv_id = row.get("arxiv_id")
    doi = row.get("doi")
    published = parse_date(row.get("published_date"))
    age_months = months_between(published, now_utc()) if published else None
    age_days = (now_utc() - published).days if published else None

    # --- citation counts from multiple sources ---
    oa_cited = None
    ss_cited = None
    ss_infl = None
    ss_refs = None
    oc_cited = None

    oa_raw = None
    counts_by_year = []
    fwci = None
    cnp = None
    is_top_1 = None
    is_top_10 = None
    is_oa = None
    topics = []
    concepts = []
    countries = None
    institutions_count = None

    if oa:
        oa_cited = oa.get("cited_by_count")
        oa_raw = load_json(oa.get("raw_json"))
        if oa_raw:
            counts_by_year = oa_raw.get("counts_by_year") or []
            fwci = oa_raw.get("fwci")
            cnp = oa_raw.get("citation_normalized_percentile") or {}
            if isinstance(cnp, dict):
                is_top_1 = cnp.get("is_in_top_1_percent")
                is_top_10 = cnp.get("is_in_top_10_percent")
                cnp = cnp.get("value")
            else:
                cnp = None
            oa_info = oa_raw.get("open_access") or {}
            is_oa = oa_info.get("is_oa")
            topics = [
                t.get("display_name") or t.get("id")
                for t in (oa_raw.get("topics") or [])[:8]
            ]
            concepts = [
                c.get("display_name")
                for c in (oa_raw.get("concepts") or [])[:8]
                if c.get("display_name")
            ]
            # authorship institutions / countries
            insts = set()
            countries_set = set()
            for a in oa_raw.get("authorships") or []:
                for inst in a.get("institutions") or []:
                    if inst.get("id"):
                        insts.add(inst["id"])
                    if inst.get("country_code"):
                        countries_set.add(inst["country_code"])
            institutions_count = len(insts) or None
            countries = len(countries_set) or None

        # also from structured columns
        if not topics:
            tj = load_json(oa.get("topics_json")) or []
            topics = [t.get("display_name") or t.get("id") for t in tj[:8] if isinstance(t, dict)]
        if not concepts:
            cj = load_json(oa.get("concepts_json")) or []
            concepts = [c.get("display_name") for c in cj[:8] if isinstance(c, dict) and c.get("display_name")]
        oa_oa = load_json(oa.get("open_access_json")) or {}
        if is_oa is None:
            is_oa = oa_oa.get("is_oa")

    if ss:
        ss_cited = ss.get("citation_count")
        ss_infl = ss.get("influential_citation_count")
        ss_refs = ss.get("reference_count")

    if oc:
        oc_cited = oc.get("citation_count")

    # best available citation count
    citation_count = None
    for c in (ss_cited, oa_cited, oc_cited):
        if c is not None:
            citation_count = c if citation_count is None else max(citation_count, c)

    # --- derived metrics ---
    citation_velocity = None
    if citation_count is not None and age_months:
        citation_velocity = round(citation_count / age_months, 4)

    influence_ratio = None
    if ss_cited and ss_cited > 0 and ss_infl is not None:
        influence_ratio = round(ss_infl / ss_cited, 4)

    # growth from counts_by_year
    growth = None
    acceleration = None
    citations_last_year = None
    citations_prev_year = None
    if counts_by_year:
        years = sorted(
            [(int(x["year"]), int(x.get("cited_by_count") or 0)) for x in counts_by_year],
            key=lambda t: t[0],
        )
        if years:
            citations_last_year = years[-1][1]
        if len(years) >= 2:
            citations_prev_year = years[-2][1]
            if citations_prev_year > 0:
                growth = round(citations_last_year / citations_prev_year, 3)
            elif citations_last_year > 0:
                growth = 99.0  # infinite growth proxy
            # rough acceleration: last year velocity vs previous
            acceleration = round(citations_last_year - citations_prev_year, 2)

    traj = trajectory_label(counts_by_year, citation_count or 0)

    # simple composite scores (0–1-ish, heuristic — tune later)
    # Impact: mix FWCI + log citations + percentile
    impact = 0.0
    if fwci is not None:
        impact += min(fwci / 5.0, 1.0) * 0.45  # FWCI 5 ≈ strong
    if citation_count is not None:
        impact += min(math.log1p(citation_count) / math.log1p(500), 1.0) * 0.35
    if cnp is not None:
        impact += float(cnp) * 0.20
    impact = round(min(impact, 1.0), 4)

    # Momentum: velocity + growth + acceleration
    momentum = 0.0
    if citation_velocity is not None:
        momentum += min(citation_velocity / 5.0, 1.0) * 0.4  # 5 cites/month strong
    if growth is not None:
        momentum += min(growth / 3.0, 1.0) * 0.35
    if acceleration is not None and acceleration > 0:
        momentum += min(acceleration / 30.0, 1.0) * 0.25
    if traj == "exploding":
        momentum = min(momentum + 0.15, 1.0)
    elif traj == "growing":
        momentum = min(momentum + 0.08, 1.0)
    momentum = round(min(momentum, 1.0), 4)

    # Influence quality
    influence = 0.0
    if influence_ratio is not None:
        influence += min(influence_ratio / 0.3, 1.0) * 0.6
    if ss_infl is not None:
        influence += min(math.log1p(ss_infl) / math.log1p(50), 1.0) * 0.4
    influence = round(min(influence, 1.0), 4)

    # Recency (newer = higher)
    recency = 0.0
    if age_days is not None:
        # 0 days → 1.0, 5 years → ~0
        recency = round(max(0.0, 1.0 - (age_days / (5 * 365))), 4)

    # Accessibility
    accessibility = 1.0 if is_oa else 0.0

    # Breadth (weak proxy from author institutions/countries on the paper itself)
    breadth = 0.0
    if countries:
        breadth += min(countries / 4.0, 1.0) * 0.5
    if institutions_count:
        breadth += min(institutions_count / 5.0, 1.0) * 0.5
    breadth = round(min(breadth, 1.0), 4)

    overall = round(
        0.30 * impact
        + 0.25 * momentum
        + 0.20 * influence
        + 0.15 * recency
        + 0.05 * breadth
        + 0.05 * accessibility,
        4,
    )

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "doi": doi,
        "published_date": row.get("published_date"),
        "age_days": age_days,
        "age_months": round(age_months, 2) if age_months else None,
        # raw counts
        "citation_count": citation_count,
        "citation_count_openalex": oa_cited,
        "citation_count_semantic": ss_cited,
        "citation_count_opencitations": oc_cited,
        "influential_citation_count": ss_infl,
        "reference_count": ss_refs,
        # normalized impact
        "fwci": fwci,
        "citation_normalized_percentile": cnp,
        "is_top_1_percent": is_top_1,
        "is_top_10_percent": is_top_10,
        # trajectory
        "counts_by_year": counts_by_year,
        "citations_last_year": citations_last_year,
        "citations_prev_year": citations_prev_year,
        "citation_velocity": citation_velocity,
        "citation_growth": growth,
        "citation_acceleration": acceleration,
        "trajectory": traj,
        "influence_ratio": influence_ratio,
        # context
        "is_oa": is_oa,
        "topics": topics,
        "concepts": concepts,
        "author_institution_count": institutions_count,
        "author_country_count": countries,
        # scores
        "score_impact": impact,
        "score_momentum": momentum,
        "score_influence": influence,
        "score_recency": recency,
        "score_breadth": breadth,
        "score_accessibility": accessibility,
        "score_overall": overall,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute paper impact/momentum scores from local DB")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--top", type=int, default=15, help="Print top-N by overall score")
    args = parser.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"DB not found: {db}")

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row

    papers = conn.execute("SELECT * FROM papers").fetchall()
    oa_map = {
        r["arxiv_id"]: dict(r)
        for r in conn.execute("SELECT * FROM openalex").fetchall()
    }
    ss_map = {
        r["arxiv_id"]: dict(r)
        for r in conn.execute("SELECT * FROM semantic_scholar").fetchall()
    }
    oc_map = {
        r["arxiv_id"]: dict(r)
        for r in conn.execute("SELECT * FROM opencitations").fetchall()
    }
    conn.close()

    scores = []
    for p in papers:
        p = dict(p)
        aid = p["arxiv_id"]
        rec = compute_for_paper(
            p,
            oa_map.get(aid),
            ss_map.get(aid),
            oc_map.get(aid),
        )
        scores.append(rec)

    scores.sort(key=lambda x: (x.get("score_overall") or 0), reverse=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in scores:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Scored {len(scores)} papers → {out}")
    print()
    print(f"Top {args.top} by overall score:")
    print("-" * 100)
    for i, r in enumerate(scores[: args.top], 1):
        print(
            f"{i:2}. [{r.get('score_overall', 0):.3f}]  "
            f"imp={r.get('score_impact', 0):.2f} mom={r.get('score_momentum', 0):.2f} "
            f"inf={r.get('score_influence', 0):.2f} rec={r.get('score_recency', 0):.2f}  "
            f"cites={r.get('citation_count') or '?':>5}  "
            f"traj={r.get('trajectory') or '?':12}  "
            f"{(r.get('title') or '')[:55]}"
        )

    # coverage summary
    with_fwci = sum(1 for r in scores if r.get("fwci") is not None)
    with_vel = sum(1 for r in scores if r.get("citation_velocity") is not None)
    with_traj = sum(1 for r in scores if r.get("counts_by_year"))
    with_infl = sum(1 for r in scores if r.get("influential_citation_count") is not None)
    print()
    print("Coverage:")
    print(f"  FWCI available:              {with_fwci}/{len(scores)}")
    print(f"  Citation velocity:           {with_vel}/{len(scores)}")
    print(f"  counts_by_year trajectory:   {with_traj}/{len(scores)}")
    print(f"  Influential citations (S2):  {with_infl}/{len(scores)}")


if __name__ == "__main__":
    main()
