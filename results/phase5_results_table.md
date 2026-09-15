# Phase 5 Evaluation Results

**Query:** mechanistic interpretability circuits random matrix theory neural networks loss landscapes statistical mechanics representation learning world models

**Ground truth:** `/Users/nataltiger/Documents/Plaksha Course/Python Programming/Capstone_Project/ArXiv_Paper_Suggester/rankings/personal_importance.csv`

**Ground-truth papers:** 48

**Evaluation k:** 40

**Retrieval pool:** 120

**Recency half-life:** 4.0

**LLM enabled:** no

## Metrics

| Method | Spearman ρ | NDCG@10 | NDCG@20 | Intersection |
|---|---:|---:|---:|---:|
| `citation_baseline` | 0.2042 | 0.0407 | 0.2523 | 19 |
| `embedding_only` | 0.6776 | 0.3517 | 0.3846 | 17 |
| `full` | 0.2984 | 0.4993 | 0.4624 | 16 |
| `ablate_embedding` | 0.5470 | 0.3708 | 0.5482 | 16 |
| `ablate_recency` | 0.4294 | 0.4150 | 0.4379 | 18 |
| `ablate_citation` | 0.1463 | 0.4697 | 0.4829 | 21 |
| `ablate_author` | 0.5312 | 0.4960 | 0.4736 | 16 |

## Weights

| Signal | Default weight |
|---|---:|
| `embedding` | 0.5000 |
| `recency` | 0.2000 |
| `citation` | 0.2500 |
| `author` | 0.0500 |

## Conclusion

The multi-signal ranker beats the citation baseline on my personal ranking (Spearman ρ = 0.2984 vs 0.2042).
