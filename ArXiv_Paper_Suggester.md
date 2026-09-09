# ArXiv Paper Suggester  
**Capstone Project – Python Programming**  
**Student:** Ajay Sharma  
**Hard limits:** 12 h or 20 h versions (see timelines below)  
**Primary hardware:** M4 MacBook Air (24 GB) · Optional overnight: RTX 3050 laptop

---

## 1. Project Overview & Motivation

The goal is to build a lightweight, controllable paper recommender that, given a short natural-language description of a researcher’s interests (plus optional constraints such as time budget, recency preference, and past reading history), returns a ranked list of arXiv papers judged to be *important* rather than merely topically similar or highly cited.

“Importance” is treated as a multi-signal quantity that can include:
- topical relevance (embeddings),
- citation / attention signals (when available),
- recency and venue proxies,
- author-level indicators,
- and, optionally, lightweight LLM-as-judge scores.

The system is deliberately scoped so that a working, evaluable version can be finished inside a 12- or 20-hour window, while the architecture remains open for substantial post-course extension.

This project draws directly on material already covered:
- Embeddings, cosine similarity, and vector search (Python Programming labs + Introduction to LLMs & GenAI)
- Vector databases and similarity search (Database Concepts)
- Evaluation-driven development, golden sets, and ranking metrics (AI in Practice)
- Systematic experimentation and baseline comparison (Machine Learning Evaluation & Optimisation)
- Linear-algebra fluency and matrix operations (Introduction to Data Science – Mathematics & Statistics + prior physics training)

---

## 2. Concrete Problem Statement

**Input**
- A free-text interest description (e.g. “neural quantum states for spin chains, variational Monte Carlo, interpretability of generative models”)
- Optional structured constraints:
  - maximum papers the user can read per week,
  - earliest acceptable publication year,
  - preference for survey vs. technical papers,
  - list of previously read paper IDs or titles,
  - temperature / diversity parameter.

**Output**
- A ranked list of 10–20 candidate papers from a fixed working corpus,
- each accompanied by a short justification (which signals contributed most to the score),
- and aggregate evaluation numbers against a held-out personal ranking and a pure citation baseline.

**Minimum viable claim the project must support**  
“On a held-out set of papers that I ranked by personal importance, the system’s ranking correlates more strongly with my judgments than a pure citation-count ranking does.”

---

## 3. Data Strategy (Instructor-Aligned)

Following the instructor’s guidance, the core evaluation corpus will be a manageable collection of approximately 100 papers. This keeps everything reproducible and inspectable.

**Working corpus construction**
1. Choose 2–3 seed topics aligned with the student’s research interests (e.g. neural quantum states, quantum chaos / RMT in many-body systems, interpretability of generative models, or a broader “AI for science” slice).
2. Retrieve candidate papers via the arXiv API (title, abstract, authors, categories, publication date, arXiv ID).
3. Optionally enrich with:
   - citation counts or attention scores from OpenAlex / Semantic Scholar API (free tiers),
   - simple altmetric-style signals if an open endpoint is available,
   - author h-index or recent productivity proxies (again via open scholarly APIs).
4. Manually curate down to ~100 papers that are thematically coherent yet vary in perceived importance.

**Personal ground-truth ranking (essential)**  
The student will rank a held-out subset (30–50 papers) by personal importance on a simple ordinal or Likert scale. This ranking is the primary evaluation target and must be created early.

---

## 4. System Architecture (Modular)

```
User prompt + constraints
        ↓
[1] Query embedding (sentence-transformer)
        ↓
[2] Candidate retrieval (vector DB or brute-force cosine on 100 papers)
        ↓
[3] Feature assembly
    - embedding similarity
    - recency score
    - citation / attention score (if available)
    - author-level proxy
    - optional LLM-as-judge score (free API)
        ↓
[4] Weighted scoring / simple learned ranker
        ↓
[5] Diversity re-ranking (optional, temperature-controlled)
        ↓
Ranked list + per-paper signal breakdown
```

All components are deliberately simple so that the 12 h and 20 h versions differ mainly in feature richness and evaluation depth, not in architectural complexity.

---

## 5. Evaluation Design (Most Important Section)

Because “importance” is subjective, the project stands or falls on evaluation discipline.

**Primary metric**
- Spearman rank correlation (ρ) between the system’s ranking and the student’s held-out personal importance ranking.

**Mandatory baseline**
- Pure citation-count ranking (or attention-score ranking) evaluated with the same Spearman ρ.

**Secondary metrics (20 h version)**
- NDCG@10 / NDCG@20
- Precision@k against a binary “high-importance” threshold
- Ablation: performance when individual signals are removed

**Success criterion for the course**  
The system should at least match, and ideally exceed, the citation baseline on the personal ranking. Even a negative result (“citations already capture most of what I care about”) is acceptable if cleanly measured and discussed.

---

## 6. Two Concrete Timelines

### 12-Hour Version (Minimum Viable, High Reliability)

| Phase | Hours | Tasks |
|-------|-------|-------|
| 1. Corpus & ground truth | 3.0 | Select seed topics, pull ~120 papers via arXiv API, manually curate to 100, create personal ranking of 40 papers |
| 2. Embeddings & retrieval | 2.5 | Embed titles+abstracts with a small sentence-transformer, store in a simple FAISS / Chroma / NumPy index |
| 3. Scoring & ranking | 2.5 | Implement cosine baseline + one or two extra signals (recency, citation count if easily obtained) |
| 4. Evaluation | 2.0 | Compute Spearman ρ vs personal ranking and vs citation baseline; produce a results table |
| 5. Write-up & demo | 2.0 | Short report, example recommendation traces, clear statement of metric and limitations |

**Deliverables at 12 h**
- Working script that takes a text query and returns a ranked list from the 100-paper corpus
- Personal ranking file
- Spearman ρ numbers for the method and the citation baseline
- Short README explaining design choices and how to reproduce

### 20-Hour Version (Richer Features + Stronger Evaluation)

| Phase | Hours | Tasks |
|-------|-------|-------|
| 1. Corpus, enrichment & ground truth | 4.0 | Same as 12 h + enrich with OpenAlex / Semantic Scholar citation or attention data; expand personal ranking to 50 papers |
| 2. Robust embedding & vector store | 3.0 | Clean embedding pipeline, persistent vector DB (Chroma or FAISS), basic metadata filtering (year, category) |
| 3. Multi-signal scorer | 4.0 | Embedding similarity + recency + citation/attention + simple author proxy; optional free LLM-as-judge (e.g. via Groq / Together / OpenRouter free tier) for a subset |
| 4. Controllable ranking | 3.0 | Expose temperature / diversity parameter and simple dynamic weights; implement a minimal re-ranker |
| 5. Thorough evaluation & ablation | 3.0 | Spearman + NDCG, ablation study, qualitative error analysis on top disagreements with personal ranking |
| 6. Write-up, demo notebook & future-work section | 3.0 | Polished report, interactive example, explicit roadmap for post-course extension |

**Deliverables at 20 h**
- Everything in the 12 h version, plus
- Multi-signal scoring with ablations
- Controllable parameters (recency preference, diversity)
- Stronger quantitative evaluation and qualitative discussion
- Clear extension roadmap

---

## 7. Post-Course Extension Roadmap (Designed In from the Start)

The architecture is intentionally modular so that any of the following can be added later without rewriting the core:

- Scale corpus from 100 → several thousand papers while keeping the same evaluation protocol
- Replace hand-tuned weights with a small learning-to-rank model (LambdaMART / simple neural ranker)
- Incorporate richer graph signals (co-citation, bibliographic coupling) via OpenAlex
- Add a lightweight conversational interface that maintains a running user model
- Continuous evaluation against an expanding personal “read / want-to-read” list
- Domain-specific importance proxies (e.g. for quantum many-body or AI-for-science slices)
- Integration of altmetrics or Dimensions-style attention data when stable free access is available
- User study with 3–5 peers ranking the same corpus to measure inter-annotator agreement

---

## 8. Risks & Explicit Mitigations

| Risk | Mitigation |
|------|----------|
| Personal ranking is noisy or inconsistent | Create the ranking in one focused sitting; record brief notes for borderline papers; treat it as a fixed golden set |
| Citation / attention APIs are rate-limited or incomplete | Make every external signal optional; the pure embedding + recency system must still run and be evaluable |
| 100-paper corpus feels too small | Accept the instructor’s guidance for the graded deliverable; treat larger scale as the first post-course extension |
| LLM-as-judge is slow or expensive | Use it only on a small candidate set or skip it entirely in the 12 h version |
| Scope creep into full RAG / agent system | Freeze the pipeline at ranking + signal breakdown; no multi-hop retrieval inside the course window |

---

## 9. Alignment with Student Background

- Prior research on quantum chaos and RMT provides comfort with “importance” judgments inside a scientific literature and with spectral / statistical thinking.
- Physics + Data Science training supplies the linear-algebra and probability foundations used in embeddings and ranking metrics.
- Recent coursework supplies ready-to-use tools: sentence embeddings, vector databases, evaluation methodology, and systematic experimentation practices.
- The project therefore sits at a natural intersection of the student’s existing research taste and the concrete skills developed in the Plaksha courses.

---

## 10. Final Deliverables Checklist (Both Versions)

- [ ] Fixed ~100-paper corpus with metadata
- [ ] Personal importance ranking (held-out)
- [ ] Reproducible recommendation script / notebook
- [ ] Primary metric: Spearman ρ vs personal ranking
- [ ] Mandatory baseline: citation-count (or attention) ranking
- [ ] Short technical report (design, metric, results, limitations, future work)
- [ ] Example recommendation traces with signal breakdown
- [ ] Clear statement of what was deliberately left for post-course work

---

**Document status:** Ready for instructor discussion and for immediate execution under either the 12-hour or 20-hour plan.  
The 12-hour version prioritises a clean, defensible evaluation; the 20-hour version adds feature richness and ablation depth while preserving the same core claim.