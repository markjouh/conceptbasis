# Research records

This directory contains the durable, reviewable record of experiments that
inform the project. Keep narrative decisions in `experiments/` and compact,
machine-readable final metrics in `results/`.

Large checkpoints, embedding profiles, reconstructed labels, and generated
HTML remain under the ignored `outputs/` tree. A result promoted into a paper
table should have both a tracked summary here and enough provenance in its
experiment note to regenerate the large intermediates.

`related-work-splice.md` is topical paper-positioning material rather than an
experiment record. Dated files under `experiments/` are chronological and may
describe superseded methods; their filenames and commands remain stable so
the historical record stays auditable.

The direct-composition reviewer baseline is recorded in
`experiments/2026-07-18-direct-composition-baseline.md`, with compact metrics in
`results/direct_composition_baseline.json`. Its one-off runner and large label
payload were retired when the repository standardized on one CUDA training
entry point.

The positive-only dictionary construction experiments are recorded in
`experiments/2026-07-19-positive-only-dictionary-construction.md`. Its compact
metrics and artifact hashes are in
`results/positive_only_dictionary_edge_average_085.json`; large reviewed-edge
and preview artifacts remain under `outputs/`. The selected, simpler
usage-profile-v8 dictionary and its provenance are promoted under `data/` so
downstream defaults do not depend on ignored output paths.

The completed canonical-leader train/development labeling pass is summarized
in `results/positive_only_dictionary_labels_leaders_v2.json` and discussed in
the same dictionary-construction experiment note.

The matched adapter experiments on SigLIP2 Giant are recorded in
`experiments/2026-07-19-siglip2-positive-dictionary-adapters.md`, with compact
development metrics in
`results/siglip2_positive_dictionary_adapter_dev.json`. That note also records
the exhaustive mixed-prompt relabel, the selected regularization sweep, and the
five-seed fixed-dictionary evaluation through 20 attributes.
The selected usage-profile-v8/object-grounded-v11 rerun is summarized in
`experiments/2026-07-21-usage-profile-v8-adapters.md` and
`results/siglip2_usage_profile_v8_v11_exhaustive_cc0_dev_k20.json`.
