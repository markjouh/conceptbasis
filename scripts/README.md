# Workflow entry points

Scripts are thin command-line entry points grouped by research stage:

- `data/` prepares annotations, frozen embeddings, and initial labels.
- `dictionary/` constructs the concept vocabulary, verifies anchors, and
  freezes concept directions.
- `evaluation/` builds benchmark inputs and computes reported metrics.
- `visualization/` renders static demos and inspection galleries.
- `sweeps/` launches named experiment grids.

Reusable neural-network components and losses belong in `conceptbasis/`, not
here. Conversely, dataset paths, one-off sweep grids, and artifact naming are
workflow policy and should remain in scripts or experiment configuration.

Every executable script must use `argparse`, even if it currently has no
options, so `python path/to/script.py --help` is guaranteed to be side-effect
free.
