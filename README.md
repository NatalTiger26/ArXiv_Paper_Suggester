# ArXiv_Paper_Suggester
This project is for suggesting a Paper based on the Inputs from the user about their needs etc and also taking into account the similarity with the things they want to read.


Will be using chromadb in case I expand the number of papers later by a lot make use FAISS




# ArXiv Paper Suggester

A lightweight, controllable paper recommender. Given a short natural-language description of a researcher's interests, plus optional constraints (time budget, recency preference, past reading history), it returns a ranked list of arXiv papers judged to be important rather than merely topically similar or highly cited.

Built as a capstone project for Python Programming at Plaksha University.

## Goal

Given a free-text interest description, rank a fixed corpus of ~100 arXiv papers by predicted importance to that researcher. Importance combines embedding similarity, recency, citation/attention signals, and optional author-level or LLM-judge signals. The system is evaluated against a personal held-out importance ranking, with a pure citation-count ranking as the mandatory baseline.

**Core claim to support:** on a held-out set of papers ranked by personal importance, the system's ranking correlates more strongly with those judgments than a pure citation-count ranking does.

## Project Structure

```
data/           raw and curated corpus (corpus_100.json)
rankings/       personal importance ranking (frozen golden set)
src/            embedding, retrieval, scoring, evaluation code
notebooks/      exploration and demo notebooks
results/        evaluation tables, ablations, example traces
decisions.md    log of curation and weighting decisions
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Core dependencies: `sentence-transformers`, `numpy`, `pandas`, `scikit-learn`, `requests`, `tqdm`, `chromadb` or `faiss-cpu`, `matplotlib`.

## Usage

```bash
python src/recommend.py --query "neural quantum states for spin chains, variational Monte Carlo, interpretability of generative models"
```

Returns a ranked list of papers from the corpus, each with a per-signal score breakdown.

## Method

1. Embed the query and the corpus (title + abstract) with a sentence-transformer.
2. Retrieve candidates by cosine similarity (vector index).
3. Assemble features: embedding similarity, recency, citation/attention (if available), author proxy, optional LLM-as-judge score.
4. Combine via transparent weighted scoring.
5. Optionally re-rank for diversity using a temperature parameter.

## Evaluation

- **Primary metric:** Spearman rank correlation (ρ) between the system's ranking and the personal held-out importance ranking.
- **Baseline:** pure citation-count (or attention-score) ranking, evaluated the same way.
- **Secondary (20h version):** NDCG@10 / NDCG@20, Precision@k, per-signal ablations.

Results are reported in `results/`, including qualitative analysis of the largest disagreements between the system's ranking and the personal ranking.

## Status / Limitations

- Corpus is fixed at ~100 papers by design, for reproducibility within the course window.
- The personal ranking is a single-annotator golden set, frozen after creation.
- External signals (citations, LLM-as-judge) are optional and degrade gracefully when unavailable.

## Future Work

Scaling the corpus, replacing hand-tuned weights with a learned ranker, richer citation-graph signals, a conversational interface with a persistent user model, and inter-annotator agreement studies. See the full roadmap in the project spec.