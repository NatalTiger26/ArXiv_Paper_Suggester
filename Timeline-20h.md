# ArXiv Paper Suggester — Detailed 20-Hour Execution Timeline

**Total budget:** 20 hours (hard upper limit for the graded deliverable)  
**Order is strict:** later phases assume earlier phases are complete and working.  
**Principle:** Get a minimal end-to-end pipeline running as early as possible, then enrich.

---

## Phase 0 — Project Skeleton & Environment (1.0 h)

| Subtask | Approx. Time | Notes |
|---------|--------------|-------|
| Create project folder structure (`data/`, `src/`, `notebooks/`, `results/`, `rankings/`) | 10 min | Keep it flat and simple |
| Set up clean virtual environment + install core libraries (sentence-transformers, numpy, pandas, scikit-learn, requests, tqdm, chromadb or faiss-cpu, matplotlib) | 25 min | Prefer CPU / MPS-friendly versions |
| Create a basic `README.md` and `requirements.txt` | 15 min | Write the high-level goal in 3–4 sentences |
| Quick test that embeddings and a tiny FAISS/Chroma index work on dummy data | 10 min | Catch environment issues immediately |

**Exit criterion:** You can embed 5 dummy sentences and retrieve the nearest neighbour.

---

## Phase 1 — Corpus Construction (3.5 h)

| Subtask | Approx. Time | Notes |
|---------|--------------|-------|
| Decide 2–3 seed topics tightly aligned with your interests (e.g. neural quantum states, quantum many-body chaos / RMT, interpretability of generative models, AI-for-science) | 15 min | Write them down; they drive everything |
| Write a small arXiv API script to pull papers (title, abstract, authors, categories, published date, arXiv ID) for the seed topics | 45 min | Use `arxiv` Python package or raw API; aim for 150–200 raw candidates |
| Deduplicate and basic cleaning (remove withdrawn, very short abstracts, obvious noise) | 30 min | |
| Manually curate down to ≈ 100 papers that feel thematically coherent but vary in perceived importance | 60 min | This is subjective but necessary; keep notes on borderline decisions |
| Save final corpus as a clean CSV / JSON (`data/corpus_100.json`) with consistent fields | 15 min | |
| Optional light enrichment: pull citation counts or attention scores via Semantic Scholar or OpenAlex free API for as many papers as possible | 45 min | Make this signal completely optional; system must work without it |

**Exit criterion:** A clean 100-paper corpus file exists and you can load it in one line of code.

---

## Phase 2 — Personal Ground-Truth Ranking (2.0 h)

| Subtask | Approx. Time | Notes |
|---------|--------------|-------|
| Randomly sample or strategically select 45–50 papers from the corpus for the held-out ranking set | 15 min | Stratify roughly by year or topic if possible |
| Rank them by personal importance (1–5 scale or pure ordinal ranking) in one focused sitting | 70 min | Do this in a single session to keep the criteria consistent; write 1-sentence notes for papers you are unsure about |
| Save the ranking as a simple CSV (`rankings/personal_importance.csv`) with arXiv ID and score/rank | 10 min | |
| Create a small “high-importance” binary threshold version of the same ranking (useful later for Precision@k) | 15 min | |
| Quick sanity check: look at top-5 and bottom-5 to confirm the ranking feels meaningful to you | 10 min | |

**Exit criterion:** A fixed, reproducible personal ranking file exists. Do **not** change it later.

---

## Phase 3 — Embedding & Retrieval Backbone (2.5 h)

| Subtask | Approx. Time | Notes |
|---------|--------------|-------|
| Choose and load a small, strong sentence-transformer (e.g. `all-MiniLM-L6-v2` or a slightly stronger but still fast model) | 15 min | Prefer models that run well on MPS / CPU |
| Embed all 100 titles + abstracts; cache the embeddings to disk | 30 min | |
| Build a simple persistent vector index (Chroma or FAISS) with metadata (year, categories, arXiv ID) | 40 min | |
| Implement a clean retrieval function: given a text query → top-k papers with scores | 30 min | |
| Test retrieval with 4–5 different interest descriptions and inspect results qualitatively | 35 min | |

**Exit criterion:** You can type an interest description and immediately get a sensible ranked list from the 100-paper corpus.

---

## Phase 4 — Multi-Signal Scoring (3.5 h)

| Subtask | Approx. Time | Notes |
|---------|--------------|-------|
| Implement pure embedding-similarity baseline scorer | 20 min | This is your simplest method |
| Add recency signal (simple function of publication year; make the decay controllable) | 25 min | |
| Add citation / attention signal (if you obtained the data); normalise it robustly | 30 min | Handle missing values gracefully |
| Optional: simple author-level proxy (e.g. number of papers by the same authors in the corpus, or a crude productivity signal) | 25 min | Keep it lightweight |
| Optional: free-tier LLM-as-judge score on a small candidate set only (Groq / Together / OpenRouter) | 40 min | Make this completely skippable; do not block on it |
| Design a transparent weighted combination of the signals (start with hand-chosen weights) | 30 min | Log the contribution of each signal for every paper |
| Expose a simple “temperature / diversity” parameter that affects re-ranking or sampling | 25 min | Even a basic MMR-style or score-perturbation approach is fine |

**Exit criterion:** A single function that, given a query + optional constraints, returns a ranked list plus a per-paper breakdown of which signals drove the score.

---

## Phase 5 — Evaluation Suite (3.0 h)

| Subtask | Approx. Time | Notes |
|---------|--------------|-------|
| Implement Spearman rank correlation against the personal ranking | 30 min | Primary metric |
| Implement the pure citation-count (or attention) baseline and compute its Spearman ρ | 25 min | Mandatory comparison |
| Add NDCG@10 and NDCG@20 | 25 min | |
| Implement simple ablation: turn each signal off one at a time and re-measure Spearman | 40 min | |
| Produce a clean results table (method vs baseline vs ablations) | 20 min | |
| Qualitative error analysis: inspect the biggest disagreements between your ranking and the system’s ranking; write short notes | 40 min | This is often the most insightful part |

**Exit criterion:** You have a results table and can state in one sentence whether (and by how much) your method beats the citation baseline on your personal ranking.

---

## Phase 6 — Controllability, Polish & Write-up (4.5 h)

| Subtask | Approx. Time | Notes |
|---------|--------------|-------|
| Wire optional user constraints (recency preference, max papers, simple interest description) into the main function | 40 min | |
| Create 3–4 polished example recommendation traces with signal breakdowns | 40 min | |
| Write a clear technical report / README covering: problem, data, method, metric, results, limitations, and explicit future-work roadmap | 90 min | |
| Add a short “how to reproduce” section and make sure the whole pipeline runs from a single command or notebook top-to-bottom | 30 min | |
| Final self-review: check that the core claim is supported by the numbers and that nothing critical is only in your head | 30 min | |
| Buffer / contingency time for unexpected bugs or small improvements | 40 min | Use only if needed; otherwise stop early |

**Exit criterion:** A complete, reproducible project that can be handed to the instructor without additional verbal explanation of how to run it or what the numbers mean.

---

## Time Allocation Summary (20 h)

| Phase | Hours | Cumulative |
|-------|-------|------------|
| 0. Skeleton & environment | 1.0 | 1.0 |
| 1. Corpus construction | 3.5 | 4.5 |
| 2. Personal ground-truth ranking | 2.0 | 6.5 |
| 3. Embedding & retrieval backbone | 2.5 | 9.0 |
| 4. Multi-signal scoring | 3.5 | 12.5 |
| 5. Evaluation suite | 3.0 | 15.5 |
| 6. Controllability, polish & write-up | 4.5 | 20.0 |

---

## Working Rules for the 20 Hours

1. **Never start Phase N+1 until the exit criterion of Phase N is met.**
2. Protect the personal ranking: once written, treat it as frozen golden data.
3. Every external signal (citations, LLM judge, author proxies) must be optional; the pure embedding + recency system must remain fully functional.
4. Prefer simple, transparent methods over clever ones. Clarity of evaluation beats marginal ranking gains.
5. If you fall behind, cut optional enrichment and LLM-as-judge first; never cut the personal ranking or the Spearman evaluation.
6. Log decisions (why a paper was included/excluded, why a weight was chosen) in a simple `decisions.md` or in notebook markdown cells.

---

**End of timeline document.**  
This is the single source of truth for execution order and time budgeting. Follow the phases in sequence.