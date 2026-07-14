# ConceptBasis

ConceptBasis asks whether an orthogonality loss can make an image–text
embedding more compositional without sacrificing ordinary retrieval. Alongside
standard contrastive training, it penalizes overlap between concept directions
so attributes can be combined more cleanly.

The resulting directions approximate a conceptual basis, with human-readable
axes for constructing additive queries.

**[Live playground](https://markjouh.github.io/conceptbasis/)**

The partial embedding effects of concepts are estimated using a regression on
synthetic concept labels; group-mean-derived directions remain as a baseline.

Training alternates minibatch contrastive updates with one full-training-set
orthogonality update per epoch.

## Evidence from compositional retrieval

Can a partial attribute description retrieve the correct held-out object? At
14 attributes, Recall@5 improves from 46.1% with contrastive training alone,
to 68.6% with group-mean orthogonality, and 84.4% with reverse-ridge
orthogonality.

Meanwhile, the reverse-ridge model retains the performance of the contrastive
baseline on standard image–text retrieval: 90.0% versus 89.0% development
Recall@5, averaged over five matched runs.

![Compositional and ordinary Recall at five across the three incremental objectives](docs/assets/composability-retrieval.svg)

## Pipeline

| Stage | Entry point |
|---|---|
| Freeze class-level train/dev/test partitions | `scripts/data/make_class_splits.py` |
| Mine image attributes and captions | `scripts/data/` |
| Label train images against the frozen dictionary | [`scripts/data/label_dictionary_concepts.py`](scripts/data/label_dictionary_concepts.py) ([model-selection record](research/experiments/2026-07-11-local-vlm-dictionary-labeling.md)) |
| Construct the 256-concept dictionary | `scripts/dictionary/build_dictionary.py` |
| Build and verify concept directions | `scripts/dictionary/` |
| Train the selected reverse-ridge adapter | `python -m conceptbasis.train --objective reverse-ridge` |
| Reproduce the incremental baselines | `python -m conceptbasis.train --objective {contrastive,group-mean,reverse-ridge}` |
| Evaluate basis behavior and compositional retrieval | `scripts/evaluation/` |
| Generate playgrounds and galleries | `scripts/visualization/` |

The dictionary is constructed only from train-class CC0 images and is then held
fixed for unseen development and test classes. The selected reverse-ridge
configuration is recorded in the experiment note above; `reproduce.sh`
reproduces all three incremental objectives under the same training entry point.

## Evaluation protocol

The 1,854 THINGS object classes are split once at the class level: 1,298 train,
278 development, and 278 test. Every full-set image and its corresponding CC0
representative inherit the same class split.

- Train classes may be used for tags, dictionary construction, direction
  estimation, calibration, and adapter training.
- Development classes may be used for direction-recipe selection,
  hyperparameters, case studies, and the public CC0 playground.
- Test classes require an explicit `--allow-test` flag and are reserved for the
  final confirmatory evaluation.

The manifest is tracked in `data/splits.json`; test annotations remain under an
ignored `data/heldout/` directory.

## Repository layout

- `conceptbasis/` — model, loss, and training implementation.
- `scripts/` — data, dictionary, evaluation, sweep, and visualization CLIs.
- `research/` — experiment notes, compact results, and artifact provenance.
- `docs/` — static public playground and inspection galleries.
- `data/` — tracked annotations plus ignored regenerable arrays.
- `outputs/` — ignored checkpoints and local generated artifacts.

## Setup

```bash
pip install -e '.[dev]'
./reproduce.sh
python -m pytest -q
```

Python 3.10 or newer is required. Attribute mining, captioning, and concept
verification use an OpenAI-compatible local VLM endpoint configured with
`VLM_API_URL` and `VLM_MODEL`.

THINGS images are not redistributed. The public demos use the freely licensed
THINGSplus CC0 subset; the full THINGS dataset remains subject to its original
research license.
