# Concept-Axis Embeddings on THINGS

**Live demo: <https://markjouh.github.io/conceptbasis/>**

A label-free pipeline that trains an image-text embedding around a
**conceptual basis**: ~256 human-interpretable visual concepts become
explicit, (conditionally) orthogonal directions, and every object is
representable as its profile of concept activations — while the vector
remains a normal retrieval embedding. Concepts you can read off, compose,
and slide along ("sliders you can search with").

Trained on [THINGS](https://things-initiative.org/) (26,107 object images),
supervised entirely by pretrained foundation models (SigLIP2 + a local VLM);
no human annotation anywhere in training.

## Pipeline (scripts in run order)

| step | script | output |
|---|---|---|
| 1. mine attributes (local VLM) | `scripts/mine_attributes.py` | `data/attributes.jsonl` |
| 2. build 256-concept dictionary | `scripts/build_dictionary.py` | `data/dictionary.json` |
| 3. caption 26k images (local VLM) | `scripts/caption_images.py` | `data/captions.jsonl` |
| 4. embeddings + soft labels | `scripts/compute_labels.py` | `data/labels.parquet`, `data/*_embeddings.npy` |
| 5. contrastive prompt pairs | `scripts/generate_contrastive_prompts.py` | `data/contrastive_prompts.json` |
| 6. validate direction constructions | `scripts/validate_directions.py` | `data/concept_directions_contrastive.npy` |
| 7. VLM-verified anchors | `scripts/verify_concepts.py` | `data/concept_judgments.jsonl`, `data/direction_sources.json` |
| 8. train adapter | `python -m conceptbasis.train` (package: `conceptbasis/`) | `outputs/checkpoints/*` |
| 9. playgrounds/viewers | `scripts/make_playground_{directions,axis_aligned,frozen}.py`, `scripts/make_*_preview.py` | `outputs/*.html` |

Checkpoint: `outputs/checkpoints/latest` (conditional orthogonality:
`--lambda_orth 5 --corr_exempt 0.15`).

## Demos (`docs/` — served via GitHub Pages, CC0 images only)

- `index.html` — landing page.
- `playground.html` — **the playground**: objects as profiles over the model's
  concept directions; sliders compose and retrieve live.
- `playground-axis-aligned.html` — experiment: the same model with the concept
  basis rotated onto coordinate axes.
- `playground-baseline.html` — no-training control (raw SigLIP2 + text
  directions).
- `dictionary.html`, `attributes.html` — the mined concept dictionary and
  per-image attributes.

All demos are static, self-contained HTML: the model's outputs over the
gallery are precomputed at build time; the browser only does dot products.
`outputs/` holds local-only artifacts (checkpoints, full-gallery variants).

## Data & licensing

- **THINGS images are NOT included and must not be redistributed**
  (research/non-commercial license). Download via the
  [THINGS OSF](https://osf.io/jum2f/); the CC0 subset (1,854 images) is
  freely licensed.
- Generated annotations in `data/` (captions, attributes, VLM judgments,
  dictionary) are our own model-generated artifacts (captioner/judge:
  Qwen3.6-35B, Apache-2.0).
- Some HTML viewers embed full-set THINGS images and are therefore
  **git-ignored** (see `.gitignore`); viewers built on the CC0 subset
  (`attribute_preview.html`, `dictionary_preview.html`) are shareable.
- Large regenerable caches (`*.npy`, `*.parquet`, checkpoints) are ignored;
  rerun steps 4/8 to rebuild.

## Setup

```bash
pip install -e .        # installs dependencies + the conceptbasis package
./reproduce.sh          # full pipeline (see prereqs in the script header)
```

Python ≥3.10. VLM steps expect an OpenAI-compatible server (e.g. LM Studio)
at `http://127.0.0.1:1234` — set `VLM_API_URL`/`VLM_MODEL`; reasoning must be
disabled (`reasoning_effort: "none"`, handled by the scripts) — thinking mode
is much slower on these calls and degrades output quality.

