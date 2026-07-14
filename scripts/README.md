# Workflow entry points

Scripts are thin command-line entry points grouped by research stage:

- `data/` prepares annotations, frozen embeddings, and initial labels.
- `dictionary/` constructs the concept vocabulary, verifies anchors, and
  freezes concept directions.
- `evaluation/` builds benchmark inputs and computes reported metrics.
- `visualization/` renders static demos and inspection galleries.
- `sweeps/` launches named experiment grids.
- `python -m conceptbasis.train` is the single adapter-training entry point.
  Its explicit `--objective` modes preserve the progression from
  `contrastive`, to `group-mean`, to the selected `reverse-ridge` objective.
- `sweeps/sweep_reverse_ridge.py` defaults to the pure-reverse orthogonality
  sweep and accepts `--seed` for confirmation runs.

Reusable neural-network components and losses belong in `conceptbasis/`, not
here. Conversely, dataset paths, sweep grids, and artifact naming are workflow
policy and should remain in scripts.

Every executable script must use `argparse`, even if it currently has no
options, so `python path/to/script.py --help` is guaranteed to be side-effect
free.
