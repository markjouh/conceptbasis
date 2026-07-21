#!/usr/bin/env bash
# Stage 3 production wrapper: exhaustively label the finalized dictionary.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${CONCEPTBASIS_PYTHON:-$ROOT/.venv/bin/python}"
MODEL="${VLM_MODEL:-nvidia/Gemma-4-26B-A4B-NVFP4}"
MODEL_REVISION="${VLM_MODEL_REVISION:-a19cfe00be84568a6867111c9a68c9c44fdcffe6}"
SERVER_PROFILE="${VLM_SERVER_PROFILE:-gemma4-nvfp4-quality-v280-s64-t16384-u090}"
API_URL="${VLM_API_URL:-http://127.0.0.1:8000/v1/chat/completions}"
DICTIONARY="${VLM_DICTIONARY:-data/dictionary_usage_profile_v8.json}"
OUT="${VLM_LABEL_OUT:-data/dictionary_labels_train_vllm_gemma4_nvfp4_usage_profile_v8_object_grounded_v11.jsonl}"
RUN_ID="${VLM_LABEL_RUN_ID:-${VLM_RUN_ID:-gemma4-26b-a4b-nvfp4-quality-v280-file-usage-profile-v8-object-grounded-v11}}"
MAX_OUTPUT_TOKENS="${VLM_LABEL_MAX_OUTPUT_TOKENS:-${VLM_MAX_OUTPUT_TOKENS:-1350}}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ConceptBasis Python is missing from $PYTHON" >&2
  exit 1
fi

exec env \
  VLM_MODEL_REVISION="$MODEL_REVISION" \
  VLM_SERVER_PROFILE="$SERVER_PROFILE" \
  "$PYTHON" "$ROOT/scripts/data/label_fixed_dictionary.py" \
  --api-url "$API_URL" \
  --model "$MODEL" \
  --dictionary "$DICTIONARY" \
  --out "$OUT" \
  --run-id "$RUN_ID" \
  --workers "${VLM_WORKERS:-80}" \
  --image-transport "${VLM_IMAGE_TRANSPORT:-file}" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" \
  --review-mode named_binary \
  --no-structured-output \
  "$@"
