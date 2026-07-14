#!/usr/bin/env bash
# End-to-end reproduction of the concept-basis pipeline on THINGS.
#
# Prereqs:
#   - THINGS images downloaded (see README "Data & licensing"):
#       data/raw/object_images_CC0/   (1,854 CC0 images — dictionary mining)
#       data/raw/object_images/       (26,107 images — training)
#   - pip install -e .   (installs deps + the conceptbasis package)
#   - a local OpenAI-compatible VLM server for steps 2, 4, 6, and 7:
#       export VLM_API_URL=http://127.0.0.1:1234/v1/chat/completions
#       export VLM_MODEL=qwen/qwen3.6-35b-a3b
#     (reasoning must be off / reasoning_effort "none" — handled by the scripts)
#
# VLM calls are resumable. The other steps are local compute.
set -euo pipefail

# 1. freeze the class-level train/dev/test split before any annotation
python scripts/data/make_class_splits.py

# 2. tag discovery and development CC0 images separately; test remains sealed
python scripts/data/mine_attributes.py --split train --workers 8
python scripts/data/mine_attributes.py --split dev --workers 8

# 3. cluster train-class attribute mentions into the fixed dictionary
python scripts/dictionary/build_dictionary.py --merge-corr 0.5 --max-twin-corr 0.75

# 3b. label every train image against that dictionary for reverse ridge
python scripts/data/label_dictionary_concepts.py \
  --model google/gemma-4-26b-a4b-qat --workers 4

# 4. caption all 26k images (the adapter consumes train classes only)
python scripts/data/caption_images.py --workers 8

# 5. frozen embeddings + train-fitted GMM labels + class-level splits
python scripts/data/compute_labels.py

# 6. contrastive prompt pairs per concept
python scripts/dictionary/generate_contrastive_prompts.py

# 7. VLM-verified anchors sampled only from train classes
python scripts/dictionary/verify_concepts.py --per-side 24 --workers 8

# 8. frozen direction choices selected on development classes + relabel
python scripts/dictionary/build_directions.py

# 9. matched incremental objectives; evaluation uses development classes only
python -m conceptbasis.train --objective contrastive \
  --run_name classsplit_contrastive
python -m conceptbasis.train --objective group-mean --lambda_orth 8 \
  --corr_exempt 0.15 \
  --corr_weighting smooth --corr_weight_power 4 --corr_weight_floor 0.01 \
  --run_name classsplit_group_mean
python -m conceptbasis.train --objective reverse-ridge \
  --ridge_alpha 0.001 --lambda_orth 512 \
  --run_name classsplit_reverse_ridge
ln -sfn classsplit_reverse_ridge outputs/checkpoints/latest

# 10. class-disjoint development evaluation; the test partition remains sealed
python scripts/evaluation/build_groupmean_profiles.py \
  --embeddings data/image_embeddings.npy \
  --cc0-embeddings data/image_embeddings_cc0.npy \
  --labels data/labels.parquet --include-frozen \
  --checkpoint contrastive=outputs/checkpoints/classsplit_contrastive/ckpt.pt \
  --checkpoint group_mean=outputs/checkpoints/classsplit_group_mean/ckpt.pt \
  --checkpoint reverse_ridge=outputs/checkpoints/classsplit_reverse_ridge/ckpt.pt \
  --out outputs/evals/classsplit_dev_profiles.npz
python scripts/evaluation/eval_playground_subset_composability.py \
  --dictionary data/dictionary.json \
  --profiles-npz outputs/evals/classsplit_dev_profiles.npz \
  --reference-model reverse_ridge --eval-split dev \
  --subset-sizes 1,2,4,6,8,10,12,14 --rollouts 24 \
  --out outputs/evals/classsplit_dev_composability.json

# 11. public and local galleries use development classes, never test
python scripts/visualization/make_playground_directions.py --cc0 --out docs/playground.html
python scripts/visualization/make_playground_frozen.py --cc0 --out docs/playground-baseline.html
python scripts/visualization/make_attribute_preview.py
python scripts/visualization/make_dictionary_preview.py
python scripts/visualization/make_playground_directions.py
python scripts/visualization/make_playground_frozen.py
python scripts/visualization/make_readme_composability_chart.py
