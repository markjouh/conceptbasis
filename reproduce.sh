#!/usr/bin/env bash
# End-to-end reproduction of the concept-basis pipeline on THINGS.
#
# Prereqs:
#   - THINGS images downloaded (see README "Data & licensing"):
#       data/raw/object_images_CC0/   (1,854 CC0 images — dictionary mining)
#       data/raw/object_images/       (26,107 images — training)
#   - pip install -e .   (installs deps + the conceptbasis package)
#   - a local OpenAI-compatible VLM server for steps 1-3 and 6-8:
#       export VLM_API_URL=http://127.0.0.1:1234/v1/chat/completions
#       export VLM_MODEL=qwen/qwen3.6-35b-a3b
#     (reasoning must be off / reasoning_effort "none" — handled by the scripts)
#
# Steps 1-3 + 5-7 call the VLM (~5h total on an M-series Mac, resumable).
# Steps 4, 8-9 are local compute only (~30 min total).
set -euo pipefail

# 1. mine candidate attributes from the CC0 set (bottom-up concept discovery)
python scripts/mine_attributes.py --img-dir data/raw/object_images_CC0 --n-images 1854 --workers 8 --out data/attributes.jsonl

# 2. cluster attribute mentions into the 256-concept dictionary
python scripts/build_dictionary.py --merge-corr 0.5 --max-twin-corr 0.75

# 3. caption all 26k training images (contrastive supervision)
python scripts/caption_images.py --workers 8

# 4. frozen-backbone (SigLIP2) embeddings + GMM-calibrated soft labels + splits
python scripts/compute_labels.py

# 5. contrastive prompt pairs per concept (semantic axes need real negatives)
python scripts/generate_contrastive_prompts.py

# 6. per-concept validation: generic-base vs contrastive direction construction
python scripts/validate_directions.py

# 7. VLM-verified pos/neg anchors -> final directions + degenerate-axis flags
python scripts/verify_concepts.py --per-side 24 --workers 8

# 8. train the adapter (flagship config: conditional orthogonality)
python -m conceptbasis.train --lambda_orth 5 --corr_exempt 0.15 --run_name flagship
ln -sfn flagship outputs/checkpoints/latest

# 9. build the public site (docs/ = GitHub Pages, CC0 gallery) and the
#    local full-gallery playgrounds (outputs/, not redistributable)
python scripts/make_playground_directions.py --cc0 --out docs/playground.html
python scripts/make_playground_axis_aligned.py --cc0 --out docs/playground-axis-aligned.html
( cd scripts && python make_playground_frozen.py --cc0 --out docs/playground-baseline.html )
python scripts/make_attribute_preview.py
python scripts/make_dictionary_preview.py
python scripts/make_playground_directions.py
python scripts/make_playground_axis_aligned.py
( cd scripts && python make_playground_frozen.py )
