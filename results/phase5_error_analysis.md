# Phase 5 Error Analysis

**Query:** mechanistic interpretability circuits random matrix theory neural networks loss landscapes statistical mechanics representation learning world models

Rank gap is defined as `system rank - personal rank`.
Positive values indicate papers that the system **underranked** relative to the personal ranking.
Negative values indicate papers that the system **overranked** relative to the personal ranking.

## Biggest Underranks

| Paper | Personal score | Personal rank | System rank | Gap | Final score | Citations |
|---|---:|---:|---:|---:|---:|---:|
| Neural Tangent Kernel: Convergence and Generalization in Neural Networks | 5.0 | 2.5 | 39 | +36.5 | 0.5023 | 4273 |
| Rethinking generalization requires revisiting old ideas: statistical mechanics approaches and complex learning behavior | 4.0 | 9.0 | 37 | +28.0 | 0.5032 | 69 |
| Survey on the Role of Mechanistic Interpretability in Generative AI | 5.0 | 2.5 | 20 | +17.5 | 0.5379 | 22 |
| Random matrix analysis of deep neural network weight matrices | 4.0 | 9.0 | 26 | +17.0 | 0.5289 | 35 |
| Opening the AI black box: program synthesis via mechanistic interpretability | 4.0 | 9.0 | 24 | +15.0 | 0.5368 | 25 |
| Dissecting and Mitigating Diffusion Bias via Mechanistic Interpretability | 3.0 | 22.0 | 31 | +9.0 | 0.5160 | 32 |
| Random Matrix Theory for Deep Learning: Beyond Eigenvalues of Linear Models [Special Issue on the Mathematics of Deep Learning] | 3.0 | 22.0 | 29 | +7.0 | 0.5222 | 2 |
| Statistical Mechanics of Deep Learning | 5.0 | 2.5 | 9 | +6.5 | 0.5751 | 251 |
| Towards Mechanistic Interpretability of Graph Transformers via Attention Graphs | 3.0 | 22.0 | 28 | +6.0 | 0.5224 | 14 |
| Learning Rates as a Function of Batch Size: A Random Matrix Theory Approach to Neural Network Training | 2.0 | 36.0 | 38 | +2.0 | 0.5031 | 76 |
| Identifying and attacking the saddle point problem in high-dimensional non-convex optimization | 3.0 | 22.0 | 22 | +0.0 | 0.5373 | 1558 |
| Understanding polysemanticity in neural networks through coding theory | 3.0 | 22.0 | 21 | +-1.0 | 0.5377 | 19 |

## Biggest Overranks

| Paper | Personal score | Personal rank | System rank | Gap | Final score | Citations |
|---|---:|---:|---:|---:|---:|---:|
| Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability | 3.0 | 22.0 | 4 | -18.0 | 0.6427 | 198 |
| Bridging the Black Box: A Survey on Mechanistic Interpretability in AI | 4.0 | 9.0 | 5 | -4.0 | 0.6325 | 22 |
| Seeing Is Believing: Brain-Inspired Modular Training for Mechanistic Interpretability | 4.0 | 9.0 | 6 | -3.0 | 0.6173 | 63 |
| Open Problems in Mechanistic Interpretability | 5.0 | 2.5 | 1 | -1.5 | 0.7292 | 197 |
| Understanding polysemanticity in neural networks through coding theory | 3.0 | 22.0 | 21 | -1.0 | 0.5377 | 19 |
| Identifying and attacking the saddle point problem in high-dimensional non-convex optimization | 3.0 | 22.0 | 22 | 0.0 | 0.5373 | 1558 |
| Learning Rates as a Function of Batch Size: A Random Matrix Theory Approach to Neural Network Training | 2.0 | 36.0 | 38 | 2.0 | 0.5031 | 76 |
| Towards Mechanistic Interpretability of Graph Transformers via Attention Graphs | 3.0 | 22.0 | 28 | 6.0 | 0.5224 | 14 |
| Statistical Mechanics of Deep Learning | 5.0 | 2.5 | 9 | 6.5 | 0.5751 | 251 |
| Random Matrix Theory for Deep Learning: Beyond Eigenvalues of Linear Models [Special Issue on the Mathematics of Deep Learning] | 3.0 | 22.0 | 29 | 7.0 | 0.5222 | 2 |
| Dissecting and Mitigating Diffusion Bias via Mechanistic Interpretability | 3.0 | 22.0 | 31 | 9.0 | 0.5160 | 32 |
| Opening the AI black box: program synthesis via mechanistic interpretability | 4.0 | 9.0 | 24 | 15.0 | 0.5368 | 25 |

## Interpretation Notes

- **Underrank:** a paper the personal ranking considers important but the system placed relatively low.
- **Overrank:** a paper the system placed relatively high despite a lower personal importance score.
- These disagreements are useful for diagnosing whether the embedding, recency, citation, or author signals are misaligned with personal research interests.
