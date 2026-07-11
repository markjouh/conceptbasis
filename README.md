# ConceptBasis

**[Live playground](https://markjouh.github.io/conceptbasis/)**

ConceptBasis learns an image–text embedding organized around a dictionary of
human-readable concepts. Each concept is represented by a direction in the
embedding space, so images can be described by concept activations and queries
can be edited through **sliders you can search with**.

The project has two main parts:

1. A pipeline for constructing a concept dictionary and recovering semantic
   directions already present in a pretrained embedding, using lightweight
   model-generated image attributes and text prompts rather than a large
   human-labeled concept dataset.
2. An orthogonality loss that pushes those directions toward a useful basis
   while preserving the embedding's ordinary image–text retrieval behavior.

The training objective is standard symmetric image–text contrastive loss plus
a correlation-weighted penalty on pairwise concept-direction overlap. Concept
pairs that are strongly dependent in the data receive less orthogonality
pressure.

In the ideal linear picture, orthogonal concept directions are independently
decodable and can be added without one component overwriting another. Real
semantic concepts are neither independent nor perfectly orthogonal, so we do
not force that ideal. The goal is to recover some of its useful behavior while
retaining a strong general-purpose embedding.

## Evidence from additive retrieval

Adding concept directions provides a direct behavioral test of the geometry:
does each added concept preserve useful information? The model is never trained
on this retrieval task. On classes excluded from dictionary construction and
adapter training, adding the orthogonality loss raises Recall@10 from 57.7% to
77.4% over a matched contrastive-only adapter at 14 concepts. Ordinary
image–text Recall@5 remains essentially unchanged (89.1% versus 88.7%).

![Recall at 10 as concept attributes are composed](docs/assets/composability-retrieval.svg)

This development benchmark retrieves 52 eligible query classes against the
278-class development gallery, with 24 nested attribute-subset rollouts per
query and identical queries across models. It is evidence on classes unseen
during training, not the sealed final test. Full metrics and provenance are in
[`research/experiments/2026-07-10-class-disjoint-composability.md`](research/experiments/2026-07-10-class-disjoint-composability.md).

## Pipeline

| Stage | Entry point |
|---|---|
| Freeze class-level train/dev/test partitions | `scripts/data/make_class_splits.py` |
| Mine image attributes and captions | `scripts/data/` |
| Construct the 256-concept dictionary | `scripts/dictionary/build_dictionary.py` |
| Build and verify concept directions | `scripts/dictionary/` |
| Train the adapter | `python -m conceptbasis.train` |
| Evaluate basis behavior and additive retrieval | `scripts/evaluation/` |
| Generate playgrounds and galleries | `scripts/visualization/` |

The dictionary is constructed only from train-class CC0 images and is then held
fixed for unseen development and test classes. The active adapter objective
uses smooth, correlation-weighted orthogonality; its reference configuration is
recorded in `reproduce.sh`.

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
pip install -e .
./reproduce.sh
python -m unittest discover -s tests
```

Python 3.10 or newer is required. Attribute mining, captioning, and concept
verification use an OpenAI-compatible local VLM endpoint configured with
`VLM_API_URL` and `VLM_MODEL`.

THINGS images are not redistributed. The public demos use the freely licensed
THINGSplus CC0 subset; the full THINGS dataset remains subject to its original
research license.
