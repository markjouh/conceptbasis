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
- `matched_retrieval_dev.json` — five-seed matched ordinary image–text
  retrieval results for the three objectives shown in the README chart.
- `direct_composition_baseline.json` — exploratory seed-0 comparison of direct
  compositional-retrieval training against contrastive-only and reverse-ridge
  orthogonality adapters on both train and class-disjoint development splits.
- `positive_only_dictionary_edge_average_085.json` — prior working
  dictionary construction: SigLIP 2 Giant cosine proposals, Gemma 4 26B NVFP4
  positive-family edge review, and weighted average-linkage clustering into
  256 nonnegative concepts.
- `positive_only_dictionary_labels_leaders_v2.json` — complete Gemma 4 26B
  NVFP4 canonical-leader labeling pass over every train and development image,
  including throughput, coverage, output hashes, and integrity checks.
- `siglip2_positive_dictionary_adapter_dev.json` — matched seed-0 contrastive,
  group-mean, and reverse-ridge adapters using SigLIP2 Giant and the accepted
  positive-only dictionary, including retrieval, composability, support
  diagnostics, and checkpoint hashes.
- `siglip2_positive_dictionary_adapter_interim_v9_dev.json` — provisional
  matched rerun after the exhaustive mixed-prompt train relabel, including the
  old-versus-new reverse-ridge comparison. The test split remains sealed.
- `siglip2_positive_dictionary_adapter_interim_v9_sweep_dev.json` — broad and
  focused reverse-ridge regularization sweep on the exhaustive labels, plus a
  five-seed confirmation of the selected development configuration.
- `siglip2_positive_dictionary_exhaustive_cc0_dev_k20.json` — five-seed
  contrastive, group-mean, and tuned reverse-ridge comparison using exhaustive
  fixed-dictionary labels on all 278 development CC0 images through `k=20`.
- `siglip2_usage_profile_v8_v11_exhaustive_cc0_dev_k20.json` — selected
  five-seed comparison using the usage-profile-v8 dictionary, grounded-v11
  fixed labels, and Gemma grounded captions. All 278 development images are
  eligible through `k=10`; the file records the eligible cohort at every
  longer query length.

These files are small enough to review and version. Large arrays and model
weights used to produce them are intentionally excluded from Git.
