# vLLM batch-labeling runtime

This is the isolated RTX 5090 runtime for the three ConceptBasis vision-language
workflows: image captioning, open-tag mining, and exhaustive fixed-dictionary
labeling. The main project environment remains separate because the
project currently uses Python 3.14, while this runtime uses vLLM 0.25.1 on
Python 3.12 with CUDA 13 PyTorch wheels.

The single production VLM is NVIDIA's
[`Gemma-4-26B-A4B-NVFP4`](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4).
The launcher pins the tested Hub commit
`a19cfe00be84568a6867111c9a68c9c44fdcffe6` rather than floating on repository
HEAD. Set `VLLM_MODEL_REVISION` only when intentionally testing another release,
and use new output/run IDs for it.

## Host preparation and first launch

Stop or unload any LM Studio model before starting vLLM so it can reserve the
GPU. The server binds only to `127.0.0.1:8000`.

The first launch is materially different from a warm restart: it downloads the
checkpoint and may compile CUDA, TorchInductor, and FlashInfer kernels. Several
compiler processes can consume tens of GiB of **system RAM** even though the
model itself is on the GPU. This machine's validated host configuration is now
96 GiB RAM. The launchers cap Ninja and TorchInductor at four workers, OpenMP at
eight threads, and lower CPU and I/O priority; keep those caps for the first
build. Warm launches reuse the caches and are much lighter.

The launcher exposes `VLLM_BUILD_JOBS`, `VLLM_COMPILE_THREADS`,
`VLLM_OMP_THREADS`, `VLLM_NICE_LEVEL`, and `VLLM_IONICE_LEVEL`. Lower the first
three if the desktop becomes sluggish during a new kernel build. The caps limit
host compilation only; they do not lower steady-state GPU inference quality.

Readiness checks:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
```

## Gemma: recommended runtime

Start the production server with:

```bash
scripts/vllm/serve_vlm.sh
```

The default is the quality profile: 280 Gemma visual tokens per image, 64 live
sequences, 16,384 scheduled tokens per iteration, a 4,096-token context, 90%
GPU-memory reservation, CUTLASS MoE, and 80 queued client requests. The server
uses throughput mode and optimization level 3. The 140-token `fast` profile is
still available as an explicit image-detail tradeoff, but it is not the
quality-preserving default:

```bash
VLLM_PROFILE=fast scripts/vllm/serve_vlm.sh
```

The clients default to `--image-transport file`. vLLM therefore reads the
original local image rather than a client-side resized JPEG; the Gemma
processor still maps it to the server's fixed 280-token vision budget. The
clients' 768-pixel/JPEG-quality-88 path applies only when
`VLM_IMAGE_TRANSPORT=base64` is selected for a non-local endpoint.

### Three production workflows

Run one wrapper at a time while the Gemma server is healthy:

| Workflow | Command | Default output | Output budget | End-to-end rate |
| --- | --- | --- | ---: | ---: |
| Grounded image captioning | `scripts/vllm/caption_images.sh` | `data/captions_vllm_gemma4_nvfp4_clip_grounded_v2.jsonl` | 64 tokens; retry at 90 | **14.357 img/s** |
| Open-tag mining | `scripts/vllm/mine_open_tags.sh` | `data/attributes_train_vllm_gemma4_nvfp4_open_tags_nonredundant_v8.jsonl` | 192 tokens; retry at 300 | Not rebenchmarked after prompt expansion |
| Exhaustive fixed labels | `scripts/vllm/label_fixed_dictionary.sh` | `data/dictionary_labels_train_vllm_gemma4_nvfp4_usage_profile_v8_object_grounded_v11.jsonl` | 1,350 tokens, validated `concept: YES/NO` checklist | Selected v8/v11 release |

These are steady-state RTX 5090 rates for the exact-file, 280-token quality
configuration. They include local image preparation, HTTP, inference,
validation/schema parsing, retries, and JSONL writes. The measured runs had zero
failed rows. Caption validation requires one grounded 8--30 word sentence; open tagging
requires 5--50 unique positive one- or two-word attributes and rejects explicit
negations, ambiguous properties, and vacuous properties. Fixed labeling
exhaustively reviews all dictionary leaders in canonical order and validates one exact
`concept: YES/NO` line per leader. Its compact grounding policy labels only the
named object, excludes backgrounds and other objects, and prevents properties
from being inherited from contents, sources, or associated wholes. The prior
named-checklist release measured 16.84 seconds per image and 1,165 completion
tokens in a 32-image audit; v11 has not yet been rebenchmarked.

All three outputs are append-only and resumable. Fixed labeling defaults to
the accepted positive-only dictionary. Their `.meta.json` sidecars
record the exact model revision, server profile, prompt, run ID, transport, and
inference controls; an incompatible resume is rejected rather than silently
mixing releases. Fixed labeling additionally writes failed attempts to
`.errors.jsonl`.

The previous promoted open-tag artifact sampled one image from each of 1,298
training classes and averaged **14.98 tags/image** because its prompt requested
10–15 tags. The v8 release labels all 18,211 training images and averages
**8.24 tags/image** (median 8): 45.0% fewer per image, but 14.0× the images and
7.72× the total tag observations. The lower per-image count is intentional—the
new prompt removes negations, near-duplicates, ambiguous labels, and vacuous
properties rather than filling a quota.

### Smoke runs without contaminating production output

Always give smoke tests their own output paths. This example leaves every
sample and metadata sidecar under a fresh `/tmp` directory:

```bash
CB_SMOKE_DIR="$(mktemp -d /tmp/conceptbasis-gemma-smoke.XXXXXX)"

VLM_CAPTION_OUT="$CB_SMOKE_DIR/captions.jsonl" \
  VLM_CAPTION_RUN_ID=gemma-quality-caption-smoke-v1 \
  scripts/vllm/caption_images.sh --n-images 12

VLM_ATTRIBUTE_OUT="$CB_SMOKE_DIR/open-tags.jsonl" \
  VLM_ATTRIBUTE_RUN_ID=gemma-quality-open-tags-smoke-v1 \
  scripts/vllm/mine_open_tags.sh --n-images 12

VLM_LABEL_OUT="$CB_SMOKE_DIR/fixed-labels.jsonl" \
  VLM_LABEL_RUN_ID=gemma-quality-fixed-labels-smoke-v1 \
  scripts/vllm/label_fixed_dictionary.sh --n-images 12

find "$CB_SMOKE_DIR" -maxdepth 1 -type f -print
```

Run the wrappers without `--n-images` only after inspecting the smoke JSONL.

## Overrides

Gemma server controls include `VLLM_GPU_MEMORY_UTILIZATION`,
`VLLM_MAX_MODEL_LEN`, `VLLM_MAX_NUM_SEQS`,
`VLLM_MAX_NUM_BATCHED_TOKENS`, `VLLM_VISION_TOKENS`, and
`VLLM_RENDERER_NUM_WORKERS`. Client concurrency is `VLM_WORKERS`; task-specific
output and token variables are visible in each wrapper. When changing image
budgets, prompts, output constraints, or quantization, use a new output and run
ID and rerun the quality checks instead of treating the result as a pure speed
tuning.

## Recreate the environment

```bash
uv venv .venv-vllm --python 3.12 --seed --managed-python
uv pip install --python .venv-vllm/bin/python 'vllm==0.25.1' \
  --torch-backend=cu130 --constraint scripts/vllm/constraints.txt
```

The constraints keep CUDA compiler and header packages on CUDA 13.0. Without
them, the resolver can independently select incompatible CUDA 13.x component
releases and break FlashInfer JIT linking.
