# Encoder benchmarks and candidates

The selected encoder is the pinned SigLIP2 Giant release. The historical
`PE-Core-bigG-14-448`/`meta` cache remains available as the `project` preset for
prior-baseline compatibility; SigLIP2 arrays stay in a checksummed namespace
and never overwrite those legacy `data/` arrays implicitly.

Official model references:

- [`timm/PE-Core-bigG-14-448`](https://huggingface.co/timm/PE-Core-bigG-14-448)
- [`timm/ViT-gopt-16-SigLIP2-384`](https://huggingface.co/timm/ViT-gopt-16-SigLIP2-384)
- [Google's SigLIP2 model documentation](https://huggingface.co/docs/transformers/model_doc/siglip2)
- [SigLIP2 paper](https://arxiv.org/abs/2502.14786)

## Releases under test

| Role | OpenCLIP model/tag | Exact tested revision | Native image input | Embedding width |
| --- | --- | --- | ---: | ---: |
| Historical baseline | `PE-Core-bigG-14-448` / `meta` | `17aa0c25addfa14198fa2ff73d845a22d433432e` | 448 | 1,280 |
| Selected | `ViT-gopt-16-SigLIP2-384` / `webli` | `ad3410bee2c3373be5ed01e7c4e7fcd2bf95a183` | 384 | 1,536 |

Both were tested with FP16 inference and normalized FP32 output arrays. The
SigLIP2 revision is part of the preset and cache identity, so weights and
tokenizer resolve to the same commit. The candidate uses a 64-token text
context; every one of the 26,107 current captions fits it.

## Persistent candidate namespace

The full SigLIP2 Giant candidate is stored at:

```text
outputs/encoder_candidates/siglip2-gopt-p16-384@ad3410b/
```

Its `encoder_manifest.json` records the model revision, preprocessing, input
identity, array shapes, and SHA-256 hashes. The current cache contains 26,107
full image embeddings, 1,854 CC0 embeddings, 256 initial directions, labels,
and the exact image-ID order. Candidate code refuses an output under `data/`,
and it refuses an existing candidate cache when its identity or checksum does
not match.

Generate or resume that namespace with:

```bash
scripts/data/build_siglip2_training_inputs.sh \
  --batch-size 16 \
  --preprocess-workers 8 \
  --prefetch-batches 3
```

That full 26,107-image run sustained **108.627 img/s**. It does not change the
project's `BACKBONE` constant or production arrays.

## Exact paired benchmark

`benchmark_embeddings.py` writes an `.npy`, an ordered-ID sidecar, and a
manifest. It rejects destinations under production `data/`. The following
commands recreate the 4,096-pair comparison used for the quality decision:

```bash
CB_PE_DIR="$(mktemp -d /tmp/conceptbasis-pe-paired.XXXXXX)"
CB_SIGLIP_DIR="$(mktemp -d /tmp/conceptbasis-siglip2-paired.XXXXXX)"
CB_COMPARE_DIR="$(mktemp -d /tmp/conceptbasis-encoder-compare.XXXXXX)"
CB_PE_REV=17aa0c25addfa14198fa2ff73d845a22d433432e
CB_SIGLIP_REV=ad3410bee2c3373be5ed01e7c4e7fcd2bf95a183

.venv/bin/python scripts/benchmarks/benchmark_embeddings.py \
  --kind image --model PE-Core-bigG-14-448 --pretrained meta \
  --revision "$CB_PE_REV" --n-items 4096 --seed 20260718 \
  --batch-size 8 --preprocess-workers 4 --prefetch-batches 2 \
  --input-fp16 --output "$CB_PE_DIR/image_embeddings.npy"

.venv/bin/python scripts/benchmarks/benchmark_embeddings.py \
  --kind caption --model PE-Core-bigG-14-448 --pretrained meta \
  --revision "$CB_PE_REV" --n-items 4096 --seed 20260718 \
  --batch-size 512 --output "$CB_PE_DIR/caption_embeddings.npy"

.venv/bin/python scripts/benchmarks/benchmark_embeddings.py \
  --kind image --model ViT-gopt-16-SigLIP2-384 --pretrained webli \
  --revision "$CB_SIGLIP_REV" --n-items 4096 --seed 20260718 \
  --batch-size 16 --preprocess-workers 8 --prefetch-batches 3 \
  --input-fp16 --output "$CB_SIGLIP_DIR/image_embeddings.npy"

.venv/bin/python scripts/benchmarks/benchmark_embeddings.py \
  --kind caption --model ViT-gopt-16-SigLIP2-384 --pretrained webli \
  --revision "$CB_SIGLIP_REV" --n-items 4096 --seed 20260718 \
  --batch-size 512 --output "$CB_SIGLIP_DIR/caption_embeddings.npy"

.venv/bin/python scripts/benchmarks/compare_encoder_embeddings.py \
  --baseline-image "$CB_PE_DIR/image_embeddings.npy" \
  --baseline-caption "$CB_PE_DIR/caption_embeddings.npy" \
  --candidate-image "$CB_SIGLIP_DIR/image_embeddings.npy" \
  --candidate-caption "$CB_SIGLIP_DIR/caption_embeddings.npy" \
  --baseline-revision "$CB_PE_REV" \
  --candidate-model ViT-gopt-16-SigLIP2-384 \
  --candidate-pretrained webli --candidate-revision "$CB_SIGLIP_REV" \
  --retrieval-limit 4096 --retrieval-seed 20260718 --block-size 256 \
  --norm-atol 0.0005 --max-recall-drop-pp 1.0 --fail-on-gate \
  --output "$CB_COMPARE_DIR/comparison.json"
```

For the full 26,107-caption throughput measurement, use the same pinned
candidate with `--n-items 0`:

```bash
.venv/bin/python scripts/benchmarks/benchmark_embeddings.py \
  --kind caption --model ViT-gopt-16-SigLIP2-384 --pretrained webli \
  --revision ad3410bee2c3373be5ed01e7c4e7fcd2bf95a183 \
  --n-items 0 --seed 20260718 --batch-size 512 \
  --output /tmp/siglip2-giant-full-caption-embeddings.npy
```

Wrap any command with `gpu_telemetry.py` when utilization, peak VRAM, clocks,
temperature, and power are needed in a separate JSON report:

```bash
.venv-vllm/bin/python scripts/benchmarks/gpu_telemetry.py \
  --output /tmp/encoder-gpu.json -- \
  .venv/bin/python scripts/benchmarks/benchmark_embeddings.py --help
```

Replace the final `--help` with the benchmark arguments above; it is shown this
way so the documentation check itself has no expensive side effect.

## Results and quality gates

| Measurement | PE Core bigG | SigLIP2 Giant |
| --- | ---: | ---: |
| Paired image encoding, 4,096 rows | **39.985 img/s** (batch 8) | **109.067 img/s** (batch 16) |
| Caption encoding, full 26,107 rows | not rerun in the full sweep | **2,159.155 text/s** (batch 512) |
| Caption encoding, paired 4,096 rows | **1,878.838 text/s** (batch 512) | 2,174.837 text/s (batch 512) |
| Peak allocated VRAM, paired image run | 4.971 GiB | 3.899 GiB |

The quality report uses exact row-ID sidecars and exact paired ranks over all
4,096 rows. Its enforced gates are:

- all values finite, no zero-norm rows, and maximum unit-norm error at most
  `5e-4` for all four arrays;
- no more than a 1.0 percentage-point recall drop at R@1, R@5, or R@10 in
  either image-to-caption or caption-to-image retrieval;
- a separate 512-image batch-invariance check between batch sizes 8 and 16.

All gates passed. The candidate's observed maximum norm error was
`1.1921e-7`. Its minimum cosine similarity across the batch-invariance rerun
was **0.9999995**. Paired retrieval improved rather than merely meeting the
non-inferiority threshold:

| Direction | PE R@1 | SigLIP2 Giant R@1 | Change |
| --- | ---: | ---: | ---: |
| Image to caption | 86.47% | 87.82% | +1.34 pp |
| Caption to image | 80.76% | 85.57% | +4.81 pp |

The smallest recall change across all six R@1/R@5/R@10 checks was +0.049 pp.
These results cleared the frozen-encoder numerical and paired-retrieval gates.
The later positive-dictionary adapter experiment supplied the task-specific
AUROC and trained-adapter evidence used to select SigLIP2 Giant; see
`research/experiments/2026-07-19-siglip2-positive-dictionary-adapters.md`.
