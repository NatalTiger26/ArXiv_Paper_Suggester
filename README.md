# ArXiv Paper Suggester

A multi-signal ranking system that recommends papers from a curated ~100-paper corpus given a free-text research interest. Built as a capstone on retrieval and personalization for scholarly search.

**Research seeds:** mechanistic interpretability · statistical mechanics of learning · random matrix theory & neural networks

---

## 1. Problem

Citation count alone favors older, highly cited work and ignores whether a paper matches *your* current interests. This project ranks papers for a natural-language query using:

- semantic similarity (embeddings)
- recency
- citation attention
- a light in-corpus author proxy
- optional LLM relevance judge

and evaluates the ranking against a **frozen personal importance list** (1–5 scores).

**Core claim (Phase 5):** the multi-signal ranker beats a pure citation baseline on personal ranking (Spearman ρ = 0.30 vs 0.20) and improves NDCG@10 (0.50 vs 0.04).

---

## 2. Data

| Path | Role |
|------|------|
| `data/corpus_100.json` | Curated 100-paper corpus (title, abstract, authors, year, DOI, citations, topics) |
| `data/chroma_corpus/` | Persistent Chroma index (Phase 3 embeddings) |
| `data/embeddings/` | Cached embedding vectors |
| `rankings/personal_importance.csv` | **Frozen** personal 1–5 ground truth (Phase 2) — do not edit |
| `results/phase5_results_table.md` | Metrics table |
| `results/phase5_error_analysis.md` | Underrank / overrank notes |

**Collection notes:** Papers were gathered with OpenAlex + Semantic Scholar (the arXiv Atom API returned HTTP 429 even for minimal queries). The corpus was deduplicated, cleaned, and curated to ~100 thematically relevant papers.

---

## 3. Method

### Phase 3 — Retrieval

- Model: `sentence-transformers/all-MiniLM-L6-v2` (runs on MPS/CPU)
- Index: Chroma, cosine space, metadata (year, arXiv id, citations)
- Input text: title (weighted) + abstract
- API: `retrieve(query, k)` → pool of candidates with similarity scores

### Phase 4 — Multi-signal scoring

| Signal | Definition | Default weight |
|--------|------------|----------------|
| **Embedding** | Cosine similarity from Phase 3 | 0.50 |
| **Recency** | `0.5 ** (age_years / half_life)` | 0.20 |
| **Citation** | `log1p(cites) / log1p(500)` (missing → 0.3) | 0.25 |
| **Author** | Max co-author frequency in this corpus, log-scaled | 0.05 |
| **LLM** (optional) | External judge on title/snippet; blended at weight 0.10 | — |

- Every result includes **signals** (raw 0–1 values) and **contrib** (`weight × signal`).
- `--temperature` enables MMR-style diversity (title-overlap penalty).
- LLM judge is non-blocking: missing key / API error → ranking continues without LLM.

### Phase 5 — Evaluation

Against `rankings/personal_importance.csv` (n = 48):

- Spearman ρ (system score vs personal 1–5 on intersection)
- NDCG@10, NDCG@20
- Citation-only and embedding-only baselines
- Ablations (zero one signal, renormalize weights)

### Phase 6 — Controllability

`src/recommend.py` wraps Phase 4 with optional filters: year range, arXiv-only, min citations, prefer-recent, demo traces.

---

## 4. Results (Phase 5)

**Eval setup:** broad multi-theme query · k = 40 · pool = 120 · half-life = 4 years · no LLM

| Method | Spearman ρ | NDCG@10 | NDCG@20 | Overlap |
|--------|------------|---------|---------|---------|
| citation_baseline | 0.204 | 0.041 | 0.252 | 19 |
| embedding_only | **0.678** | 0.352 | 0.385 | 17 |
| **full** | **0.298** | **0.499** | 0.462 | 16 |
| ablate_embedding | 0.547 | 0.371 | 0.548 | 16 |
| ablate_recency | 0.429 | 0.415 | 0.438 | 18 |
| ablate_citation | 0.146 | 0.470 | 0.483 | 21 |
| ablate_author | 0.531 | 0.496 | 0.474 | 16 |

**Reading the table**

- Full **beats citation** on Spearman and dominates on NDCG@10.
- **Embedding alone** best matches overall personal *order* (highest ρ).
- Full blend improves **top-of-list** quality (NDCG) by mixing recency and citations.
- Ablating citation hurts Spearman most among the “turn one off” runs in this setup.

Error analysis (underranks/overranks) lives in `results/phase5_error_analysis.md` (e.g. NTK highly valued personally but ranked low under the broad eval query).

---

## 5. Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install sentence-transformers chromadb openai

# Optional (collection / LLM only)
export OPENALEX_API_KEY=...
export SEMANTIC_SCHOLAR_API_KEY=...
export NVIDIA_API_KEY=...          # or GROQ_API_KEY / OPENROUTER_API_KEY
export NVIDIA_MODEL=openai/gpt-oss-20b   # if using NVIDIA
```

Expected layout after Phases 1–5:

```text
data/corpus_100.json
data/chroma_corpus/
rankings/personal_importance.csv
src/phase3_embeddings.py
src/phase4_scoring.py
src/phase5_evaluation.py
src/recommend.py
```

If the Chroma index is missing:

```bash
python src/phase3_embeddings.py build
```

---

## 6. How to run

### Recommend (Phase 6 — preferred entry point)

```bash
python src/recommend.py "mechanistic interpretability circuits" -k 8
python src/recommend.py "random matrix loss surfaces" --since 2020 -k 8
python src/recommend.py "statistical mechanics of deep learning" --prefer-recent -k 8
python src/recommend.py "mechanistic interpretability" --temperature 0.3 -k 8
python src/recommend.py --demo
python src/recommend.py "..." -k 5 --json
```

| Argument | Meaning |
|----------|---------|
| `query` | Free-text research interest |
| `-k` | Number of papers to return (default 10) |
| `--since YEAR` | Keep papers with year ≥ YEAR |
| `--until YEAR` | Keep papers with year ≤ YEAR |
| `--prefer-recent` | Stronger recency (shorter half-life) |
| `--half-life Y` | Recency half-life in years (default 4) |
| `--temperature T` | MMR diversity in [0, 1] (0 = pure score order) |
| `--weights E,R,C,A` | Weights for embedding, recency, citation, author |
| `--arxiv-only` | Drop DOI-only / synthetic ids |
| `--min-citations N` | Minimum citation count |
| `--llm` | Enable optional LLM judge (needs API key) |
| `--json` | Machine-readable output |
| `--demo` | Four example recommendation traces |

Filters are applied **after** scoring. The wrapper over-fetches candidates so filters can still return up to `k` papers.

### Phase 4 scoring (same ranker, no post-filters)

```bash
python src/phase4_scoring.py rank "mechanistic interpretability circuits" -k 10
python src/phase4_scoring.py rank "..." --weights 0.6,0.2,0.15,0.05 --half-life 3
python src/phase4_scoring.py rank "..." --temperature 0.35
python src/phase4_scoring.py rank "..." --llm -k 8
python src/phase4_scoring.py rank "..." -k 5 --json
```

### Phase 3 retrieval only

```bash
python src/phase3_embeddings.py build
python src/phase3_embeddings.py query "random matrix loss surfaces" -k 10
python src/phase3_embeddings.py demo
python src/phase3_embeddings.py interactive
```

### Phase 5 evaluation

```bash
python src/phase5_evaluation.py run
python src/phase5_evaluation.py run --llm
python src/phase5_evaluation.py run --query "mechanistic interpretability circuits"
python src/phase5_evaluation.py errors --top 12
```

Regenerates `results/phase5_results_table.md` and `results/phase5_results.json`.

### Smoke test (ranking + eval)

```bash
python src/phase3_embeddings.py query "random matrix loss surfaces" -k 5 \
  && python src/phase4_scoring.py rank "random matrix loss surfaces" -k 5 \
  && python src/phase5_evaluation.py run
```

Rebuilding the full corpus from OpenAlex/S2 is **not** required to reproduce ranking or evaluation if `data/` and `rankings/` are present.

---

## 7. Example behavior

**Query:** `mechanistic interpretability circuits`  
Top hits typically include *Open Problems in Mechanistic Interpretability*, *Automated Circuit Discovery*, and MI safety/survey papers, with explicit `emb` / `rec` / `cit` / `auth` contributions.

**Query:** `random matrix loss surfaces --since 2020`  
Keeps post-2020 RMT / loss-surface papers (e.g. universal loss-surface characteristics, RMT in deep learning).

Each printed result shows:

- `final_score`
- `signals` — raw signal values in [0, 1]
- `contrib` — how much each signal added to the final score

---

## 8. Limitations

1. **Small corpus (100) and partial GT overlap (≈16–21 / 48)** — metrics are indicative, not large-scale IR estimates.
2. **Personal labels are subjective** and frozen at one point in time.
3. **Author signal** is in-corpus frequency only, not prestige or h-index.
4. **Citation signal** favors older papers that had more time to accumulate citations.
5. **Embedding-only** can beat full multi-signal on Spearman; extra signals do not improve every metric.
6. **LLM judge** depends on an external API and a live model name; failures are silent fallbacks.
7. **One broad eval query** mixes three themes; topic-specific queries change absolute numbers.
8. Post-hoc filters in `recommend.py` can interact with MMR (`--temperature`) because diversity is computed on the pre-filter pool.

---

## 9. Future work

1. Larger corpus and higher overlap with personal labels
2. Learn signal weights from a held-out slice of the personal ranking
3. Cross-encoder re-rank on the top-20
4. Practical-impact features (GitHub stars, Papers with Code, HF downloads)
5. Per-seed evaluation (separate ρ for MI / stat-mech / RMT)
6. Stronger embeddings or domain-adapted models
7. Simple web or notebook UI for constraints

---

## 10. Project layout

```text
.
├── README.md
├── data/
│   ├── corpus_100.json
│   ├── chroma_corpus/
│   └── embeddings/
├── rankings/
│   └── personal_importance.csv      # FROZEN ground truth
├── results/
│   ├── phase5_results_table.md
│   ├── phase5_results.json
│   └── phase5_error_analysis.md
└── src/
    ├── phase3_embeddings.py         # embed + Chroma + retrieve
    ├── phase4_scoring.py            # multi-signal score_and_rank
    ├── phase5_evaluation.py         # Spearman, NDCG, ablations
    └── recommend.py                 # Phase 6 CLI + constraints
```

Supporting scripts from earlier phases (collection, consolidation, personal ranking UI) may also live under `src/` but are not required to run recommendation or evaluation once artifacts exist.

---

## 11. License / course note

Academic capstone project for evaluation. Respect terms of use for arXiv, OpenAlex, Semantic Scholar, and any LLM provider. No separate open-source license is declared.
