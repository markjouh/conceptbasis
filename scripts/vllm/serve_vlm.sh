#!/usr/bin/env bash
# Local runtime utility: serve the pinned Gemma VLM used by all Stage 1/3 jobs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLLM_ENV="${VLLM_ENV:-$ROOT/.venv-vllm}"
DEFAULT_MODEL="nvidia/Gemma-4-26B-A4B-NVFP4"
DEFAULT_MODEL_REVISION="a19cfe00be84568a6867111c9a68c9c44fdcffe6"
MODEL="${VLLM_MODEL:-$DEFAULT_MODEL}"
MODEL_REVISION="${VLLM_MODEL_REVISION:-}"
if [[ -z "${VLLM_MODEL_REVISION+x}" && "$MODEL" == "$DEFAULT_MODEL" ]]; then
  MODEL_REVISION="$DEFAULT_MODEL_REVISION"
fi
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
MOE_BACKEND="${VLLM_MOE_BACKEND:-cutlass}"
QUANTIZATION="${VLLM_QUANTIZATION:-}"
REASONING_PARSER="${VLLM_REASONING_PARSER:-gemma4}"
PROFILE="${VLLM_PROFILE:-quality}"
PERFORMANCE_MODE="${VLLM_PERFORMANCE_MODE:-throughput}"
OPTIMIZATION_LEVEL="${VLLM_OPTIMIZATION_LEVEL:-3}"
MM_PROCESSOR_CACHE_GB="${VLLM_MM_PROCESSOR_CACHE_GB:-0}"
ALLOWED_LOCAL_MEDIA_PATH="${VLLM_ALLOWED_LOCAL_MEDIA_PATH:-$ROOT/data/raw}"
RENDERER_NUM_WORKERS="${VLLM_RENDERER_NUM_WORKERS:-4}"
BUILD_JOBS="${VLLM_BUILD_JOBS:-4}"
COMPILE_THREADS="${VLLM_COMPILE_THREADS:-4}"
OMP_THREADS="${VLLM_OMP_THREADS:-8}"
NICE_LEVEL="${VLLM_NICE_LEVEL:-10}"
IONICE_LEVEL="${VLLM_IONICE_LEVEL:-7}"

case "$PROFILE" in
  fast)
    DEFAULT_VISION_TOKENS=140
    ;;
  quality)
    DEFAULT_VISION_TOKENS=280
    ;;
  *)
    echo "Unknown VLLM_PROFILE=$PROFILE (expected fast or quality)" >&2
    exit 2
    ;;
esac
VISION_TOKENS="${VLLM_VISION_TOKENS:-$DEFAULT_VISION_TOKENS}"
case "$VISION_TOKENS" in
  70|140|280|560|1120) ;;
  *)
    echo "Unsupported VLLM_VISION_TOKENS=$VISION_TOKENS (expected 70, 140, 280, 560, or 1120)" >&2
    exit 2
    ;;
esac

if [[ ! -x "$VLLM_ENV/bin/vllm" ]]; then
  echo "vLLM is missing from $VLLM_ENV" >&2
  echo "Create the environment described in scripts/vllm/README.md first." >&2
  exit 1
fi

CUDA_HOME="${CUDA_HOME:-$("$VLLM_ENV/bin/python" -c 'from pathlib import Path; import nvidia; print(Path(next(iter(nvidia.__path__))) / "cu13")')}"
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "CUDA compiler is missing from $CUDA_HOME/bin/nvcc" >&2
  exit 1
fi
for CUDA_LIB in libcudart libcublas libcublasLt; do
  if [[ ! -e "$CUDA_HOME/lib/$CUDA_LIB.so" && -e "$CUDA_HOME/lib/$CUDA_LIB.so.13" ]]; then
    ln -s "$CUDA_LIB.so.13" "$CUDA_HOME/lib/$CUDA_LIB.so"
  fi
done

export CUDA_HOME
export PATH="$VLLM_ENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export MAX_JOBS="$BUILD_JOBS"
export TORCHINDUCTOR_COMPILE_THREADS="$COMPILE_THREADS"
export OMP_NUM_THREADS="$OMP_THREADS"

echo "Host limits: MAX_JOBS=$MAX_JOBS, TORCHINDUCTOR_COMPILE_THREADS=$TORCHINDUCTOR_COMPILE_THREADS, OMP_NUM_THREADS=$OMP_NUM_THREADS, nice=$NICE_LEVEL, ionice=$IONICE_LEVEL" >&2
echo "Kernels: MoE=$MOE_BACKEND, FlashInfer sampler=$VLLM_USE_FLASHINFER_SAMPLER" >&2
echo "Model adapters: quantization=${QUANTIZATION:-auto}, reasoning_parser=${REASONING_PARSER:-none}" >&2
echo "Profile: $PROFILE, vision_tokens=$VISION_TOKENS, max_seqs=$MAX_NUM_SEQS, max_batched_tokens=$MAX_NUM_BATCHED_TOKENS, max_model_len=$MAX_MODEL_LEN, renderer_workers=$RENDERER_NUM_WORKERS" >&2
echo "Model: $MODEL revision=${MODEL_REVISION:-floating}" >&2

REVISION_ARGS=()
if [[ -n "$MODEL_REVISION" ]]; then
  REVISION_ARGS=(--revision "$MODEL_REVISION" --tokenizer-revision "$MODEL_REVISION")
fi
QUANTIZATION_ARGS=()
if [[ -n "$QUANTIZATION" ]]; then
  QUANTIZATION_ARGS=(--quantization "$QUANTIZATION")
fi
REASONING_ARGS=()
if [[ -n "$REASONING_PARSER" ]]; then
  REASONING_ARGS=(--reasoning-parser "$REASONING_PARSER")
fi

# These names are launcher inputs, not vLLM runtime environment variables.
# Avoid passing explicitly supplied values through to vLLM's env-var validator.
unset VLLM_GPU_MEMORY_UTILIZATION VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS
unset VLLM_MAX_NUM_BATCHED_TOKENS VLLM_MOE_BACKEND VLLM_PROFILE
unset VLLM_QUANTIZATION VLLM_REASONING_PARSER
unset VLLM_VISION_TOKENS VLLM_PERFORMANCE_MODE VLLM_OPTIMIZATION_LEVEL
unset VLLM_MM_PROCESSOR_CACHE_GB VLLM_BUILD_JOBS VLLM_COMPILE_THREADS
unset VLLM_OMP_THREADS VLLM_NICE_LEVEL VLLM_IONICE_LEVEL
unset VLLM_ALLOWED_LOCAL_MEDIA_PATH VLLM_RENDERER_NUM_WORKERS
unset VLLM_MODEL_REVISION

exec nice -n "$NICE_LEVEL" ionice -c 2 -n "$IONICE_LEVEL" \
  "$VLLM_ENV/bin/vllm" serve "$MODEL" \
  "${REVISION_ARGS[@]}" \
  "${QUANTIZATION_ARGS[@]}" \
  --served-model-name "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --moe-backend "$MOE_BACKEND" \
  --performance-mode "$PERFORMANCE_MODE" \
  --optimization-level "$OPTIMIZATION_LEVEL" \
  --mm-processor-cache-gb "$MM_PROCESSOR_CACHE_GB" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  --renderer-num-workers "$RENDERER_NUM_WORKERS" \
  --mm-processor-kwargs "{\"max_soft_tokens\":$VISION_TOKENS}" \
  --generation-config vllm \
  --disable-uvicorn-access-log \
  --limit-mm-per-prompt '{"image":1,"video":0,"audio":0}' \
  "${REASONING_ARGS[@]}" \
  --trust-remote-code \
  "$@"
