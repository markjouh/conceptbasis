#!/usr/bin/env bash
# Stage 1b production wrapper: mine open-set tags for dictionary discovery.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${CONCEPTBASIS_PYTHON:-$ROOT/.venv/bin/python}"
MODEL="${VLM_MODEL:-nvidia/Gemma-4-26B-A4B-NVFP4}"
MODEL_REVISION="${VLM_MODEL_REVISION:-a19cfe00be84568a6867111c9a68c9c44fdcffe6}"
SERVER_PROFILE="${VLM_SERVER_PROFILE:-gemma4-nvfp4-quality-v280-s64-t16384-u090}"
API_URL="${VLM_API_URL:-http://127.0.0.1:8000/v1/chat/completions}"
OUT="${VLM_ATTRIBUTE_OUT:-data/attributes_train_vllm_gemma4_nvfp4_open_tags_nonredundant_v8.jsonl}"
RUN_ID="${VLM_ATTRIBUTE_RUN_ID:-${VLM_RUN_ID:-gemma4-26b-a4b-nvfp4-full-train-open-tags-nonredundant-v8}}"
MAX_OUTPUT_TOKENS="${VLM_ATTRIBUTE_MAX_OUTPUT_TOKENS:-${VLM_MAX_OUTPUT_TOKENS:-192}}"
RETRY_MAX_OUTPUT_TOKENS="${VLM_ATTRIBUTE_RETRY_MAX_OUTPUT_TOKENS:-${VLM_RETRY_MAX_OUTPUT_TOKENS:-300}}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ConceptBasis Python is missing from $PYTHON" >&2
  exit 1
fi

exec env \
  VLM_MODEL_REVISION="$MODEL_REVISION" \
  VLM_SERVER_PROFILE="$SERVER_PROFILE" \
  "$PYTHON" "$ROOT/scripts/data/mine_open_tags.py" \
  --api-url "$API_URL" \
  --model "$MODEL" \
  --out "$OUT" \
  --run-id "$RUN_ID" \
  --workers "${VLM_WORKERS:-80}" \
  --image-transport "${VLM_IMAGE_TRANSPORT:-file}" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" \
  --retry-max-output-tokens "$RETRY_MAX_OUTPUT_TOKENS" \
  "$@"
