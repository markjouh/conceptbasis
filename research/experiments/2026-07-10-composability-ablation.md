# Compositional retrieval ablation — 2026-07-10

Status: preliminary, single seed. This is the canonical human-readable record
for the SigLIP2-B composability experiment. The complete machine-readable output
is `research/results/composability_ablation_k14.json`.

Project decision after this ablation: identification was removed from the
active training objective. V4 is retained only as a legacy comparison. The
active flagship is contrastive + smooth correlation-weighted orthogonality
(`tau=.15`, power 4, floor `.01`, `lambda_orth=8`).

## Main result

Conditional orthogonality accounts for almost all of the compositional retrieval
gain of the full v4 objective. At `k=14`, contrastive + conditional orthogonality
matches v4's median source rank (11 versus 11) and nearly matches R@10
(48.50% versus 49.42%), R@20 (61.33% versus 62.21%), and R@50
(76.36% versus 76.19%). Contrastive-only is much weaker (median rank 39,
R@10 26.25%).

Unconditional orthogonality is still strong at high `k`, but is consistently
worse than conditional orthogonality. It reaches median rank 13 and R@10
45.87% at `k=14`, while badly degrading one-attribute retrieval and concept
AUROC. This supports the conditional exemption: aggressively orthogonalizing
naturally correlated concepts is harmful.

A first smooth correlation-weighted run used
`w = 0.05 + 0.95 exp(-(|corr| / 0.15)^2)`. It also reaches median rank 11 at
`k=14` and slightly improves over the hard conditional model on MRR, R@1,
R@10, R@20, and R@50. It is worse at low `k` and has lower concept AUROC,
revealing a readability/composability tradeoff that requires further sweeps.

## Objectives

All learned models use the same 768-dimensional SigLIP2-B frozen features, the
same 320-dimensional two-layer image/text adapters, seed 0, 60 epochs, batch
size 1024, AdamW, cosine schedule, and symmetric image-caption InfoNCE.

| name | identification | orthogonality | correlation exemption |
|:--|--:|--:|:--|
| Contrastive only | 0 | 0 | none |
| Unconditional orth | 0 | 5 | none |
| Conditional orth | 0 | 5 | exempt pairs with `abs(label correlation) >= 0.15` |
| Smooth orth | 0 | 5 | `0.05 + 0.95 exp(-(|corr| / 0.15)^2)` |
| V4 | 1 | 5 | exempt pairs with `abs(label correlation) >= 0.15` |

The concept directions used by the two concept losses are differentiable soft
class-mean differences:

`d_k = normalize(mu_positive_k - mu_negative_k)`.

The identification term is a calibrated BCE predicting each soft concept label
from projection onto `d_k`. The orthogonality term is the mean squared cosine
between penalized pairs of directions.

## Data and benchmark protocol

- Training corpus: 26,107 THINGS images.
- Backbone cache: `archive/data/image_embeddings_siglip2.npy`.
- Caption cache: `archive/data/caption_embeddings_siglip2.npy`.
- Concept labels were reconstructed by GMM-calibrating projections onto
  `archive/data/concept_directions_siglip2_final.npy`.
- The reconstructed table is
  `outputs/evals/labels_siglip2_final_reconstructed.parquet`.
- The reconstructed labels reproduce the audited frozen group-mean profiles
  with RMSE `1.78e-7`, correlation effectively 1.0, and maximum absolute error
  `1.67e-6`.
- Evaluation gallery: 1,854 separate CC0 images.
- Fixed high-attribute cohort: 392 images with at least 14 mapped, non-flagged
  dictionary concepts.
- Each image receives 24 fixed-seed random attribute permutations. Prefixes at
  `k = 1, 2, 4, 6, 8, 10, 12, 14` form nested queries.
- Query score is the sum of the selected standardized group-mean concept
  profiles, matching the playground operation.
- Target rank is the average-tie rank of the query image among all 1,854 gallery
  images. Lower median rank is better.
- The same cohort, permutations, and queries are used for every model.

## Final training diagnostics

| model | caption R@1 | concept AUROC | direction orth-RMS |
|:--|--:|--:|--:|
| Contrastive only | 95.25% | 0.8474 | 0.2359 |
| Conditional orth | 94.90% | 0.8715 | 0.2056 |
| Smooth orth | 94.60% | 0.8150 | 0.1548 |
| Unconditional orth | 94.50% | 0.7684 | 0.1334 |
| V4 | 94.85% | 0.8837 | 0.2141 |

Unconditional orthogonality achieves the lowest raw overlap but the worst
concept AUROC. Conditional orthogonality preserves semantic correlation while
delivering essentially all of v4's compositional gain.

## Correlation graph is not clusterable

The conditional exemption is an edge-level relation, not an equivalence
relation. On the 256 reconstructed training-label columns, the graph containing
an edge whenever `abs(correlation) >= threshold` has the following structure:

| threshold | edges | density | components | largest component | median degree |
|--:|--:|--:|--:|--:|--:|
| 0.10 | 20,221 | 62.0% | 1 | 256 | 160.0 |
| 0.15 | 15,310 | 46.9% | 1 | 256 | 122.0 |
| 0.20 | 11,412 | 35.0% | 1 | 256 | 90.5 |
| 0.25 | 8,298 | 25.4% | 2 | 255 | 63.0 |
| 0.30 | 6,011 | 18.4% | 5 | 252 | 41.0 |
| 0.40 | 2,999 | 9.2% | 14 | 242 | 15.5 |
| 0.50 | 1,379 | 4.2% | 51 | 193 | 5.0 |

At the actual threshold of 0.15, the graph is fully connected. Positive edges
alone also connect all 256 concepts. Therefore, connected-component clustering
would merge the entire dictionary, and even much higher thresholds retain one
giant component.

The relation is strongly non-transitive. Example triples include:

- `tool`–`bumpy`: -0.597; `bumpy`–`grounded`: +0.583; but
  `tool`–`grounded`: +0.004.
- `rough`–`biodegradable`: +0.579; `biodegradable`–`spreadable`: +0.616;
  but `rough`–`spreadable`: +0.005.
- `animal`–`domesticated`: +0.584; `domesticated`–`consumable`: +0.563;
  but `animal`–`consumable`: -0.001.

The current loss does not propagate exemptions through paths. If A–B and B–C
are exempt, A–C is still penalized whenever its own correlation falls below the
threshold. Any dictionary integration must preserve this pairwise graph rather
than turn correlations into clusters. Synonym/equivalence clustering and
empirical relationship edges should remain separate operations.

## Geometry on penalized versus exempt pairs

The all-pairs orth-RMS obscures what the conditional objective is doing. Using
global train-split group-mean directions and the fixed 0.15 mask:

| model | all-pair RMS | penalized RMS | exempt RMS | penalized median abs cosine | exempt median abs cosine | penalized p90 | exempt p90 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Frozen | 0.4068 | 0.2361 | 0.5383 | 0.1794 | 0.5061 | 0.3828 | 0.7539 |
| Contrastive only | 0.2457 | 0.1325 | 0.3299 | 0.0873 | 0.2673 | 0.2188 | 0.5109 |
| Unconditional orth | 0.1285 | 0.1021 | 0.1530 | 0.0627 | 0.0890 | 0.1659 | 0.2380 |
| Smooth orth | 0.1419 | 0.0712 | 0.1929 | 0.0434 | 0.1096 | 0.1163 | 0.3070 |
| Conditional orth | 0.2001 | 0.0642 | 0.2841 | 0.0418 | 0.2043 | 0.1052 | 0.4428 |
| V4 | 0.2091 | 0.0667 | 0.2969 | 0.0442 | 0.2181 | 0.1092 | 0.4606 |

Thus conditional v4 is strongly orthogonal where the loss applies: 90% of
penalized pairs have absolute cosine at most 0.109. However, 46.91% of all pairs
are exempt and retain substantial overlap. The defensible current claim is
"correlation-aware selective orthogonalization," not that the complete
dictionary forms an orthogonal basis.

For the smooth objective's own weights, weighted RMS cosine is 0.0832, versus
0.0945 for the hard-conditional checkpoint and 0.1055 for the unconditional
checkpoint under the same weights. The smooth run's weight distribution has
mean 0.4825, median 0.4533, 10th percentile 0.0512, and 90th percentile 0.9752.

## Complete smooth-weighting results

| k | median | MRR | R@1 | R@5 | R@10 | R@20 | R@25 | R@50 |
|---:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 347.0 | 0.0215 | 0.58% | 2.23% | 3.99% | 7.48% | 8.69% | 15.72% |
| 2 | 210.0 | 0.0414 | 1.35% | 5.19% | 8.47% | 13.59% | 16.02% | 24.50% |
| 4 | 96.0 | 0.0778 | 3.35% | 10.16% | 16.14% | 23.75% | 26.68% | 37.82% |
| 6 | 52.0 | 0.1216 | 5.87% | 16.83% | 24.32% | 34.12% | 37.63% | 49.42% |
| 8 | 31.0 | 0.1598 | 8.01% | 22.99% | 32.24% | 43.09% | 46.67% | 58.33% |
| 10 | 21.0 | 0.2013 | 10.92% | 28.91% | 38.49% | 49.86% | 53.35% | 65.54% |
| 12 | 15.0 | 0.2418 | 13.99% | 33.86% | 44.05% | 56.46% | 60.59% | 71.14% |
| 14 | 11.0 | 0.2756 | 17.53% | 36.60% | 48.77% | 61.85% | 65.65% | 76.93% |

## Complete compositional retrieval results

| k | model | median | MRR | R@1 | R@5 | R@10 | R@20 | R@25 | R@50 |
|---:|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | Frozen | 345.0 | 0.0155 | 0.31% | 1.46% | 2.61% | 4.82% | 5.82% | 11.04% |
| 1 | CLIP | 334.0 | 0.0224 | 0.71% | 2.29% | 4.22% | 6.83% | 8.35% | 14.50% |
| 1 | CLIP + conditional orth | 306.0 | 0.0220 | 0.53% | 2.19% | 3.99% | 7.72% | 9.12% | 15.86% |
| 1 | CLIP + unconditional orth | 413.0 | 0.0204 | 0.55% | 2.05% | 3.71% | 6.75% | 8.25% | 14.70% |
| 1 | V4: CLIP + ID + conditional orth | 294.0 | 0.0219 | 0.47% | 2.22% | 4.15% | 7.96% | 9.35% | 16.27% |
| 2 | Frozen | 210.0 | 0.0295 | 0.80% | 3.28% | 5.61% | 9.62% | 11.50% | 19.27% |
| 2 | CLIP | 207.0 | 0.0381 | 1.26% | 4.70% | 7.61% | 12.38% | 14.57% | 22.83% |
| 2 | CLIP + conditional orth | 176.0 | 0.0433 | 1.47% | 5.34% | 8.95% | 14.49% | 16.92% | 26.32% |
| 2 | CLIP + unconditional orth | 269.0 | 0.0371 | 1.26% | 4.78% | 7.39% | 12.00% | 13.89% | 21.43% |
| 2 | V4: CLIP + ID + conditional orth | 169.0 | 0.0439 | 1.54% | 5.39% | 9.11% | 14.83% | 17.27% | 27.01% |
| 4 | Frozen | 123.0 | 0.0508 | 1.79% | 6.29% | 10.13% | 16.46% | 19.28% | 30.62% |
| 4 | CLIP | 112.0 | 0.0658 | 2.77% | 8.45% | 13.12% | 19.75% | 22.81% | 33.72% |
| 4 | CLIP + conditional orth | 78.0 | 0.0818 | 3.50% | 10.64% | 17.02% | 25.45% | 28.95% | 41.15% |
| 4 | CLIP + unconditional orth | 135.0 | 0.0687 | 2.97% | 9.05% | 13.93% | 20.74% | 23.36% | 32.63% |
| 4 | V4: CLIP + ID + conditional orth | 74.0 | 0.0830 | 3.51% | 10.76% | 17.17% | 26.37% | 29.70% | 42.18% |
| 6 | Frozen | 88.0 | 0.0659 | 2.26% | 8.54% | 14.13% | 21.79% | 25.18% | 38.15% |
| 6 | CLIP | 78.0 | 0.0896 | 3.96% | 11.90% | 18.41% | 26.71% | 29.84% | 41.53% |
| 6 | CLIP + conditional orth | 44.0 | 0.1264 | 6.03% | 17.50% | 25.41% | 36.36% | 39.97% | 52.76% |
| 6 | CLIP + unconditional orth | 75.0 | 0.1049 | 4.94% | 14.73% | 20.95% | 29.05% | 32.33% | 43.08% |
| 6 | V4: CLIP + ID + conditional orth | 42.0 | 0.1282 | 5.96% | 17.64% | 26.17% | 36.88% | 40.77% | 53.88% |
| 8 | Frozen | 72.0 | 0.0786 | 2.95% | 10.30% | 16.67% | 25.78% | 29.53% | 42.37% |
| 8 | CLIP | 61.0 | 0.1074 | 4.94% | 14.68% | 21.66% | 30.78% | 33.92% | 46.35% |
| 8 | CLIP + conditional orth | 28.0 | 0.1624 | 8.16% | 22.58% | 32.63% | 44.38% | 48.53% | 61.15% |
| 8 | CLIP + unconditional orth | 45.0 | 0.1433 | 7.49% | 19.75% | 28.15% | 37.61% | 41.03% | 51.53% |
| 8 | V4: CLIP + ID + conditional orth | 27.0 | 0.1656 | 8.42% | 23.02% | 33.09% | 45.26% | 49.16% | 62.67% |
| 10 | Frozen | 64.0 | 0.0884 | 3.44% | 11.80% | 18.38% | 28.53% | 32.23% | 45.29% |
| 10 | CLIP | 50.0 | 0.1217 | 5.98% | 16.48% | 23.92% | 33.17% | 36.82% | 50.15% |
| 10 | CLIP + conditional orth | 19.5 | 0.1991 | 10.64% | 28.23% | 38.83% | 51.18% | 55.39% | 68.11% |
| 10 | CLIP + unconditional orth | 29.0 | 0.1814 | 9.97% | 25.41% | 34.30% | 44.71% | 47.98% | 59.62% |
| 10 | V4: CLIP + ID + conditional orth | 18.0 | 0.2028 | 10.87% | 28.68% | 39.48% | 52.05% | 56.32% | 69.11% |
| 12 | Frozen | 57.0 | 0.0969 | 4.03% | 12.97% | 20.05% | 30.23% | 34.09% | 47.09% |
| 12 | CLIP | 44.0 | 0.1340 | 6.73% | 18.36% | 25.62% | 35.89% | 39.54% | 53.01% |
| 12 | CLIP + conditional orth | 14.0 | 0.2309 | 12.69% | 32.89% | 44.33% | 57.12% | 60.71% | 72.11% |
| 12 | CLIP + unconditional orth | 19.0 | 0.2214 | 12.81% | 31.23% | 40.60% | 51.69% | 54.90% | 66.27% |
| 12 | V4: CLIP + ID + conditional orth | 13.0 | 0.2352 | 12.95% | 33.77% | 45.08% | 57.74% | 61.72% | 73.08% |
| 14 | Frozen | 55.0 | 0.1043 | 4.24% | 13.36% | 22.05% | 32.78% | 35.24% | 48.20% |
| 14 | CLIP | 39.0 | 0.1403 | 6.69% | 19.10% | 26.25% | 37.70% | 40.64% | 55.62% |
| 14 | CLIP + conditional orth | 11.0 | 0.2646 | 15.80% | 36.71% | 48.50% | 61.33% | 65.67% | 76.36% |
| 14 | CLIP + unconditional orth | 13.0 | 0.2567 | 14.81% | 36.07% | 45.87% | 57.93% | 61.51% | 72.74% |
| 14 | V4: CLIP + ID + conditional orth | 11.0 | 0.2654 | 15.42% | 38.89% | 49.42% | 62.21% | 67.85% | 76.19% |

## Paired comparisons at k=14

- V4 versus conditional orth: v4 ranks the target better on 39.96% of rollout
  queries, ties on 36.98%, and is worse on 23.07%. Mean v4-minus-conditional
  rank is -3.58.
- V4 versus unconditional orth: v4 is better on 48.49%, ties on 11.59%, and is
  worse on 39.92%. Mean v4-minus-unconditional rank is -19.30, indicating a
  substantially worse tail for unconditional orth despite its median rank of 13.

## Interpretation

1. Contrastive adapter training alone gives only a modest improvement over the
   frozen encoder.
2. Conditional orthogonality without identification recovers essentially the
   full v4 composability curve. This is the cleanest evidence so far that the
   orthogonality objective causes additive concept composability.
3. Identification adds little to this retrieval benchmark, but improves concept
   AUROC from 0.8715 to 0.8837. Its role appears to be concept readability rather
   than composability.
4. Unconditional orthogonality demonstrates that lower Gram overlap is not
   automatically better. It reaches orth-RMS 0.1334 but damages AUROC and is
   worse than conditional orthogonality across the composability curve.
5. The widening difference from contrastive-only as `k` grows is consistent
   with reduced cross-concept interference rather than a constant retrieval
   offset.
6. Smooth weighting improves high-order composition over the hard mask on most
   metrics, but sacrifices low-k retrieval and AUROC. One arbitrary smooth
   function is therefore promising but not yet a principled replacement.

## Caveats and required follow-ups

- These are single-seed results. Repeat at no fewer than three seeds.
- Add bootstrap confidence intervals over source images, not over correlated
  rollouts.
- Repeat on the new backbone only after the dictionary and label construction
  are frozen.
- Consider held-out concept combinations in addition to held-out CC0 images.
- Report both compositional retrieval and semantic readability; unconditional
  orth illustrates why either metric alone is incomplete.

## Smooth-weight parameter screen and MPS optimization

The original runs silently used CPU because the execution sandbox hides the
Metal device. Outside the sandbox, MPS is available on the 40-core Apple GPU.
The concept loss was vectorized from 256 Python iterations into matrix
multiplications; the vectorized ID loss, directions, and orthogonality loss
match the original implementation to numerical precision. A five-epoch MPS run
including evaluation and checkpointing completed in 4.6 seconds, and a full
60-epoch run completed in roughly 10–14 seconds. The trainer now accepts
`--device mps` and fails explicitly if it is unavailable.

Fourteen configurations were screened for 30 epochs on MPS. The exploratory
ranking score was `MRR@k14 * validation AUROC`; its components are reported
separately below.

| configuration | tau | power | floor | lambda | AUROC | orth-RMS | median k1 | median k14 | MRR k14 | R@10 k14 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `t015_p4_f005` | 0.15 | 4 | 0.05 | 5 | 0.853 | 0.185 | 305 | 8 | 0.300 | 55.1% |
| `t015_p4_f001_l8` | 0.15 | 4 | 0.01 | 8 | 0.880 | 0.208 | 292 | 9 | 0.286 | 52.7% |
| `t010_p2_f000` | 0.10 | 2 | 0.00 | 5 | 0.908 | 0.238 | 277 | 10 | 0.272 | 50.0% |
| `t015_p2_f000` | 0.15 | 2 | 0.00 | 5 | 0.873 | 0.205 | 296 | 9 | 0.280 | 53.0% |
| `t010_p4_f000` | 0.10 | 4 | 0.00 | 5 | 0.917 | 0.250 | 273 | 12 | 0.265 | 47.9% |
| `t015_p4_f001` | 0.15 | 4 | 0.01 | 5 | 0.882 | 0.212 | 285 | 9 | 0.275 | 52.9% |
| `t015_p8_f000` | 0.15 | 8 | 0.00 | 5 | 0.890 | 0.222 | 283 | 10 | 0.270 | 50.7% |
| `t015_p4_f000` | 0.15 | 4 | 0.00 | 5 | 0.888 | 0.220 | 284 | 10 | 0.269 | 52.3% |
| `t015_p4_f001_l3` | 0.15 | 4 | 0.01 | 3 | 0.884 | 0.217 | 284 | 10 | 0.270 | 51.0% |
| `t020_p4_f001` | 0.20 | 4 | 0.01 | 5 | 0.851 | 0.189 | 308 | 9 | 0.278 | 52.9% |
| `hard_t015` | 0.15 | — | — | 5 | 0.886 | 0.220 | 284 | 10 | 0.267 | 51.3% |
| `t020_p4_f000` | 0.20 | 4 | 0.00 | 5 | 0.856 | 0.193 | 302 | 10 | 0.275 | 52.7% |
| `t020_p2_f000` | 0.20 | 2 | 0.00 | 5 | 0.841 | 0.180 | 319 | 9 | 0.278 | 52.9% |
| `t020_p8_f000` | 0.20 | 8 | 0.00 | 5 | 0.858 | 0.196 | 304 | 10 | 0.271 | 51.1% |

Two finalists and a fresh hard-mask control were then trained for 60 epochs on
MPS. The more balanced smooth model is the current best candidate:

| model | AUROC | orth-RMS | median k1 | median k4 | median k8 | median k14 | MRR k14 | R@1 k14 | R@10 k14 | R@50 k14 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Hard mask, lambda 5 | 0.871 | 0.206 | 306 | 78 | 28 | 11 | 0.265 | 15.80% | 48.52% | 76.36% |
| Smooth, floor .05, p4, lambda 5 | 0.830 | 0.165 | 331.5 | 89 | 28 | 11 | **0.292** | **19.38%** | 49.61% | **78.85%** |
| Smooth, floor .01, p4, lambda 8 | **0.865** | **0.193** | **311** | **77** | **26** | **10** | 0.281 | 17.23% | **50.50%** | 76.79% |
| Full v4 | 0.884 | 0.214 | 294 | 74 | 27 | 11 | 0.265 | 15.42% | 49.42% | 76.19% |

The balanced smooth candidate improves composition over the hard mask while
losing only 0.006 AUROC. However, the sweep used the same CC0 cohort later used
for reporting, so these gains are exploratory and selection-biased. A paper
result requires a held-out tuning split or a separate evaluation set, plus
multiple seeds.

## Reproducibility artifacts

- Training checkpoints:
  - `outputs/checkpoints/ablation_clip_only_siglip2b/`
  - `outputs/checkpoints/ablation_clip_orth_siglip2b/`
  - `outputs/checkpoints/ablation_clip_orth_smooth_siglip2b/`
  - `outputs/checkpoints/ablation_clip_orth_smooth_p4_f005_mps/`
  - `outputs/checkpoints/ablation_clip_orth_smooth_p4_f001_l8_mps/`
  - `outputs/checkpoints/ablation_clip_orth_unconditional_siglip2b/`
  - `outputs/checkpoints/adapter_v4/`
- Reconstructed labels:
  `outputs/evals/labels_siglip2_final_reconstructed.parquet`
- Matched profiles:
  `outputs/evals/siglip2_ablation_groupmean_profiles.npz`
- Full metrics:
  `research/results/composability_ablation_k14.json`
- Evaluation code:
  `scripts/evaluation/eval_playground_subset_composability.py`
- Profile construction:
  `scripts/evaluation/build_groupmean_profiles.py`
- Label reconstruction:
  `scripts/evaluation/rebuild_soft_labels_from_directions.py`
