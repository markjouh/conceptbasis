#!/usr/bin/env bash
# End-to-end reproduction of the concept-basis pipeline on THINGS.
#
# Prereqs:
#   - THINGS images downloaded (see README "Data & licensing"):
#       data/raw/object_images_CC0/   (1,854 CC0 images — dictionary mining)
#       data/raw/object_images/       (26,107 images — training)
#   - pip install -e .   (installs deps + the conceptbasis package)
#   - a local OpenAI-compatible VLM server for steps 1, 3, 5, and 6:
#       export VLM_API_URL=http://127.0.0.1:1234/v1/chat/completions
#       export VLM_MODEL=qwen/qwen3.6-35b-a3b
#     (reasoning must be off / reasoning_effort "none" — handled by the scripts)
#
# VLM calls are resumable. The other steps are local compute.
set -euo pipefail

# 1. mine candidate attributes from the CC0 set (bottom-up concept discovery)
python scripts/data/mine_attributes.py --img-dir data/raw/object_images_CC0 --n-images 1854 --workers 8 --out data/attributes.jsonl

# 2. cluster attribute mentions into the 256-concept dictionary
python scripts/dictionary/build_dictionary.py --merge-corr 0.5 --max-twin-corr 0.75

# 3. caption all 26k training images (contrastive supervision)
python scripts/data/caption_images.py --workers 8

# 4. frozen-backbone embeddings + GMM-calibrated soft labels + splits
python scripts/data/compute_labels.py

# 5. contrastive prompt pairs per concept (semantic axes need real negatives)
python scripts/dictionary/generate_contrastive_prompts.py

# 6. VLM-verified positive/negative anchors
python scripts/dictionary/verify_concepts.py --per-side 24 --workers 8

# 7. final directions (best-of-three per concept, validated) + relabel
python scripts/dictionary/build_directions.py

# 8. train the adapter (flagship: contrastive + smooth weighted orthogonality)
python -m conceptbasis.train --lambda_orth 8 --corr_exempt 0.15 \
  --corr_weighting smooth --corr_weight_power 4 --corr_weight_floor 0.01 \
  --run_name flagship
ln -sfn flagship outputs/checkpoints/latest

# 9. build the public site (docs/ = GitHub Pages, CC0 gallery) and the
#    local full-gallery playgrounds (outputs/, not redistributable)
python scripts/visualization/make_playground_directions.py --cc0 --out docs/playground.html
python scripts/visualization/make_playground_frozen.py --cc0 --out docs/playground-baseline.html
python scripts/visualization/make_attribute_preview.py
python scripts/visualization/make_dictionary_preview.py
python scripts/visualization/make_playground_directions.py
python scripts/visualization/make_playground_frozen.py
