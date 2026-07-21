#!/usr/bin/env bash
# Reproduce the reported SigLIP2 Giant / usage-profile-v8 experiment (Stages 0
# and 4–7 in scripts/README.md). This intentionally pins the accepted
# dictionary, captions, open tags, and fixed labels instead of rerunning VLM
# annotation.
#
# VLM annotations and the accepted dictionary are frozen inputs here. Creating
# a new annotation release is a different experiment; use the versioned
# wrappers under scripts/vllm/ and promote its artifacts deliberately.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

python_bin="${CONCEPTBASIS_PYTHON:-$repo_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "ConceptBasis Python is missing from $python_bin" >&2
  exit 1
fi
if [[ ! -d data/raw/object_images || ! -d data/raw/object_images_CC0 ]]; then
  echo "missing THINGS images; see README 'Data and licensing'" >&2
  exit 1
fi

encoder_key="siglip2-gopt-p16-384@ad3410b"
dictionary_key="usage-profile-v8-v1"
input_dir="outputs/training_inputs/$encoder_key/$dictionary_key"
dictionary="data/dictionary_usage_profile_v8.json"
captions="data/captions_vllm_gemma4_nvfp4_clip_grounded_v2.jsonl"
dev_labels="data/dictionary_labels_cc0_dev_vllm_gemma4_nvfp4_usage_profile_v8_object_grounded_v11.jsonl"
train_labels="data/dictionary_labels_train_vllm_gemma4_nvfp4_usage_profile_v8_object_grounded_v11_merged.jsonl"
smoke="${REPRODUCE_SMOKE:-0}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "missing required artifact: $1" >&2
    exit 1
  fi
}

"$python_bin" scripts/data/make_class_splits.py

require_file "$dictionary"
require_file "$captions"
require_file "$train_labels"
require_file "$dev_labels"

# Bind the selected dictionary to a checksummed SigLIP2 cache. Existing valid
# stages are reused, so this command is also the resume entry point.
scripts/data/build_siglip2_training_inputs.sh \
  --output-dir "$input_dir" --dictionary "$dictionary" \
  --batch-size 16 --preprocess-workers 8 --prefetch-batches 3

training_args=()
seeds=(0 1 2 3 4)
if [[ "$smoke" == "1" ]]; then
  training_args=(--epochs 1 --max_steps 1 --eval_every 1)
  seeds=(0)
  echo "REPRODUCE_SMOKE=1: running one step for one seed" >&2
fi

checkpoint_args=()
for seed in "${seeds[@]}"; do
  for objective in contrastive group-mean reverse-ridge; do
    family="${objective//-/_}"
    run_name="siglip2_giant_usage_profile_v8_v11_${family}_s${seed}"
    "$python_bin" -m conceptbasis.train \
      --objective "$objective" --seed "$seed" --run_name "$run_name" \
      --image_ids "$input_dir/image_ids.json" \
      --image_embeddings "$input_dir/image_embeddings.npy" \
      --caption_embeddings "$input_dir/caption_embeddings.npy" \
      --soft-labels "$input_dir/labels.parquet" \
      --captions "$captions" --dictionary "$dictionary" \
      --fixed-labels "$train_labels" "${training_args[@]}"
    checkpoint_args+=(--checkpoint "${family}_s${seed}=outputs/checkpoints/${run_name}/ckpt.pt")
  done
done

profiles="outputs/evals/siglip2_usage_profile_v8_v11_exhaustive_cc0_dev_profiles.npz"
metrics="outputs/evals/siglip2_usage_profile_v8_v11_exhaustive_cc0_dev_k20.json"
"$python_bin" scripts/evaluation/build_retrieval_profiles.py \
  --embeddings "$input_dir/image_embeddings.npy" \
  --cc0-embeddings "$input_dir/image_embeddings_cc0.npy" \
  --soft-labels "$input_dir/labels.parquet" --image-ids "$input_dir/image_ids.json" \
  --dictionary "$dictionary" --train-fixed-labels "$train_labels" \
  --eval-fixed-labels "$dev_labels" "${checkpoint_args[@]}" --out "$profiles"

"$python_bin" scripts/evaluation/evaluate_compositional_retrieval.py \
  --dictionary "$dictionary" --profiles-npz "$profiles" \
  --fixed-labels "$dev_labels" --label-field present \
  --reference-model reverse_ridge_s0 --eval-split dev \
  --include-flagged \
  --subset-sizes 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 \
  --rollouts 24 --out "$metrics"

if [[ "$smoke" != "1" ]]; then
  "$python_bin" scripts/evaluation/summarize_seeded_composability.py \
    --metrics "$metrics" \
    --history contrastive=outputs/checkpoints/siglip2_giant_usage_profile_v8_v11_contrastive_s{seed}/history.json \
    --history group_mean=outputs/checkpoints/siglip2_giant_usage_profile_v8_v11_group_mean_s{seed}/history.json \
    --history reverse_ridge=outputs/checkpoints/siglip2_giant_usage_profile_v8_v11_reverse_ridge_s{seed}/history.json \
    --out research/results/siglip2_usage_profile_v8_v11_exhaustive_cc0_dev_k20.json
  "$python_bin" scripts/visualization/make_readme_composability_chart.py
fi
