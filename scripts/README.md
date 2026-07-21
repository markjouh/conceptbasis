# Pipeline entry points

Every maintained executable names the artifact it consumes or produces and
starts with a stage header. Four terms are used consistently:

- **open tags** are sparse, free-form VLM phrases used only to discover the
  dictionary;
- **fixed labels** are exhaustive YES/NO decisions against a finalized
  dictionary;
- **training inputs** are frozen image embeddings, prompt directions, and
  calibrated soft-label arrays bound to one encoder/dictionary release;
- **retrieval profiles** are held-out image-by-concept score matrices used by
  the compositional benchmark and playgrounds.

## Ordered workflow

| Stage | Entry point | Main artifact |
| --- | --- | --- |
| 0. Freeze class splits | `data/make_class_splits.py` | `data/splits.json` |
| 1a. Caption images | `vllm/caption_images.sh` → `data/caption_images.py` | caption JSONL + metadata |
| 1b. Mine open tags | `vllm/mine_open_tags.sh` → `data/mine_open_tags.py` | open-tag JSONL + metadata |
| 2. Build dictionary | `dictionary/build_dictionary.py` | dictionary JSON + provenance |
| 3. Label fixed dictionary | `vllm/label_fixed_dictionary.sh` → `data/label_fixed_dictionary.py` | fixed-label JSONL + metadata |
| 4. Build training inputs | `data/build_siglip2_training_inputs.sh` → `data/build_training_inputs.py` | checksummed encoder/input directory |
| 5. Train adapters | `python -m conceptbasis.train --objective …` | checkpoint + history |
| 6a. Build retrieval profiles | `evaluation/build_retrieval_profiles.py` | matched profile NPZ |
| 6b. Evaluate composition | `evaluation/evaluate_compositional_retrieval.py` | per-run metrics JSON |
| 6c. Summarize seeds | `evaluation/summarize_seeded_composability.py` | tracked result JSON |
| 7. Render outputs | `visualization/make_*.py` | public pages, galleries, and chart |

`reproduce.sh` is the exact accepted-run orchestration. It begins with the
tracked annotations, dictionary, and fixed labels, then runs Stages 0 and 4–7.
Stages 1–3 create a new annotation/dictionary release and are intentionally not
folded into benchmark reproduction.

Dictionary construction requires frozen SigLIP2 image embeddings. It reads
only the dictionary-independent `image_embeddings.npy` and `image_ids.json`
from a compatible Stage 4 cache; candidate phrase directions and usage
profiles are rebuilt by the dictionary script itself. This lets a new
dictionary reuse the expensive accepted encoder pass without reusing any
dictionary labels.

## Supporting tools

- `data/partition_open_tags.py` converts an imported tag JSONL to the class
  split; `data/merge_fixed_label_shards.py` validates and joins completed
  fixed-label shards.
- `dictionary/propose_merge_edges.py` and
  `dictionary/adjudicate_merge_edges.py` are optional merge-method experiments,
  not part of the accepted usage-profile recipe.
- `sweeps/sweep_reverse_ridge.py` launches the matched Stage 5/6 sweep.
- `benchmarks/` contains isolated performance and encoder-comparison tools that
  never write production caches.
- `vllm/serve_vlm.sh` starts the isolated RTX 5090 inference server used by
  all Stage 1 and Stage 3 wrappers.

The maintained neural pipeline is CUDA-only. Tests may exercise pure tensor
helpers on CPU, but production CLIs fail early without CUDA instead of silently
changing numerical backends.

Reusable models, losses, split logic, encoder provenance, and VLM transport
belong in `conceptbasis/`; dataset paths, artifact names, sweep grids, and
workflow policy remain in these entry points. Every executable Python script
uses `argparse`, so `python path/to/script.py --help` is side-effect free.
