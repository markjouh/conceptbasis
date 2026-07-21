# ConceptBasis

ConceptBasis asks whether an orthogonality loss can make an image–text
embedding more compositional without sacrificing ordinary retrieval. Alongside
standard contrastive training, it penalizes overlap between concept directions
so attributes can be combined more cleanly.

The resulting directions approximate a conceptual basis, with human-readable
axes for constructing additive queries.

**[Live playground](https://markjouh.github.io/conceptbasis/)**

The partial embedding effects of concepts are estimated using a regression on
VLM-derived binary labels; group-mean-derived directions remain as a baseline.

The encoder is the pinned
[SigLIP2 Giant](https://huggingface.co/timm/ViT-gopt-16-SigLIP2-384), and all
VLM annotation jobs use
[Gemma 4 26B NVFP4](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4).
The maintained pipeline is CUDA-only.

Training alternates minibatch contrastive updates with one full-training-set
orthogonality update per epoch.

## Evidence from compositional retrieval

Can a partial attribute description retrieve the correct held-out object? With
exhaustive fixed-dictionary labels on all 278 held-out development classes, at
10 attributes Recall@5 improves from 19.8% with contrastive training alone, to
36.1% with group-mean orthogonality, and 74.5% with reverse-ridge
orthogonality. The chart reports longer queries as well and labels their
shrinking eligible cohorts explicitly.

Meanwhile, the reverse-ridge model retains the performance of the contrastive
baseline on standard image–text retrieval: 96.9% versus 97.0% development
Recall@5, averaged over five matched runs. Shaded bands and whiskers show one
sample standard deviation across seeds.

![Compositional and ordinary Recall at five across the three incremental objectives](docs/assets/composability-retrieval.svg)

## Pipeline

| Stage | Entry point |
| --- | --- |
| 0. Freeze class-level train/dev/test partitions | [`scripts/data/make_class_splits.py`](scripts/data/make_class_splits.py) |
| 1a. Caption images for contrastive training | [`scripts/vllm/caption_images.sh`](scripts/vllm/caption_images.sh) |
| 1b. Mine open tags for dictionary discovery | [`scripts/vllm/mine_open_tags.sh`](scripts/vllm/mine_open_tags.sh) |
| 2. Construct the 256-concept dictionary | [`scripts/dictionary/build_dictionary.py`](scripts/dictionary/build_dictionary.py) |
| 3. Exhaustively label the fixed dictionary | [`scripts/vllm/label_fixed_dictionary.sh`](scripts/vllm/label_fixed_dictionary.sh) |
| 4. Materialize frozen encoder/training inputs | [`scripts/data/build_siglip2_training_inputs.sh`](scripts/data/build_siglip2_training_inputs.sh) |
| 5. Train the three incremental objectives | `python -m conceptbasis.train --objective {contrastive,group-mean,reverse-ridge}` |
| 6. Build profiles and evaluate compositional retrieval | [`scripts/evaluation/`](scripts/evaluation/) |
| 7. Generate playgrounds, galleries, and charts | [`scripts/visualization/`](scripts/visualization/) |

Open tags and fixed labels are deliberately different artifacts: open tags are
sparse, free-form discovery input, while fixed labels are exhaustive decisions
against the frozen vocabulary. The exact entry points and artifact flow are
indexed in [`scripts/README.md`](scripts/README.md).

The dictionary is constructed from open tags on every available train-split
exemplar and is then held fixed for unseen development and test classes. The
accepted artifact is
[`data/dictionary_usage_profile_v8.json`](data/dictionary_usage_profile_v8.json),
with the construction and selected reverse-ridge configuration recorded under
`research/experiments/`. `reproduce.sh` reproduces all three incremental
objectives under the same training entry point.

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
- [`data/`](data/README.md) — tracked annotation/dictionary releases plus
  ignored regenerable arrays.
- `outputs/` — ignored checkpoints and local generated artifacts.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m pytest -q
```

Python 3.10 or newer, an NVIDIA GPU, and a working CUDA installation are
required for the maintained data, dictionary, training, and evaluation
pipeline. CPU remains sufficient for repository tests and metadata-only
utilities. Attribute mining, captioning, and fixed-dictionary labeling use the
isolated local runtime documented in `scripts/vllm/README.md`.

## Data and licensing

The pipeline expects the full and CC0 THINGS image sets at
`data/raw/object_images/` and `data/raw/object_images_CC0/`. Download the
official archives from the [THINGS OSF project](https://osf.io/jum2f/):

```bash
pip install osfclient
mkdir -p data/raw/downloads
osf -p jum2f fetch osfstorage/images_THINGS.zip data/raw/downloads/images_THINGS.zip
osf -p jum2f fetch osfstorage/images_THINGSplus-CC0.zip data/raw/downloads/images_THINGSplus-CC0.zip
osf -p jum2f fetch osfstorage/password_images.txt data/raw/downloads/password_images.txt
osf -p jum2f fetch osfstorage/LICENSE data/raw/downloads/THINGS-LICENSE.txt
unzip data/raw/downloads/images_THINGSplus-CC0.zip -d data/raw
unzip -P YOUR_PASSWORD data/raw/downloads/images_THINGS.zip -d data/raw
```

The unzip password is in `password_images.txt`. Using it confirms agreement to
the THINGS license: the full image set is limited to research and educational,
non-commercial use and may not be redistributed or modified. The THINGSplus
subset is CC0.

## Reproduction

The accepted VLM annotations are tracked, so the default reproduction starts
from them and rebuilds the frozen encoder cache, five matched training seeds,
the exhaustive development evaluation, and the chart:

```bash
./reproduce.sh
```

To smoke-test the non-VLM stages:

```bash
REPRODUCE_SMOKE=1 ./reproduce.sh
```

`reproduce.sh` uses the tracked annotations from the accepted experiment. To
generate new annotations, run the corresponding wrapper in `scripts/vllm/`.

THINGS images are not redistributed. The public demos use the freely licensed
THINGSplus CC0 subset; the full THINGS dataset remains subject to its original
research license.
