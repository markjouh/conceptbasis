# Tracked results

- `composability_ablation_k14.json` — frozen, contrastive-only,
  hard/unconditional orthogonality, smooth orthogonality, and legacy v4
  comparison across attribute-subset sizes through `k=14`.
- `smooth_orth_finalists_k14.json` — final smooth-weighting candidates.
- `smooth_orth_sweep_summary.json` — compact hyperparameter sweep summary.
- `reverse_ridge_dev_results.json` — compact pure reverse-ridge sweep and
  class-disjoint compositional-retrieval results. These are development-only;
  the sealed test split remains unused.
- `three_model_dev_composability.json` — the README comparison of contrastive,
  group-mean, and reverse-ridge directions. Each attribute count uses every
  eligible development class and approximately 6,700 sampled queries.

These files are small enough to review and version. Large arrays and model
weights used to produce them are intentionally excluded from Git.
