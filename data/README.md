# Data artifact map

Tracked JSON/JSONL files are immutable research inputs or compact release
candidates. Large regenerable arrays and held-out test annotations are ignored.
Artifact filenames include the model/method/revision when changing them would
change semantics; those names are intentionally not shortened after a result
has been recorded.

## Accepted reported stack

| Role | Artifact |
| --- | --- |
| Class partition | `splits.json` |
| Full and CC0 row order | `image_ids.json`, `cc0_image_ids.json` |
| Training captions used by reported checkpoints | `captions_vllm_gemma4_nvfp4_clip_grounded_v2.jsonl` + metadata |
| Accepted 256-concept dictionary | `dictionary_usage_profile_v8.json` + provenance |
| Exhaustive train fixed labels | `dictionary_labels_train_vllm_gemma4_nvfp4_usage_profile_v8_object_grounded_v11_merged.jsonl` + metadata |
| Exhaustive development fixed labels | `dictionary_labels_cc0_dev_vllm_gemma4_nvfp4_usage_profile_v8_object_grounded_v11.jsonl` + metadata |

`reproduce.sh` pins exactly this stack. “Fixed labels” are the closed-set
`present`/`uncertain` decisions used to estimate reverse-ridge directions.

## Selected annotation inputs

| Role | Artifact |
| --- | --- |
| Full-train open tags | `attributes_train_vllm_gemma4_nvfp4_open_tags_nonredundant_v8.jsonl` + metadata |
| Gemma grounded captions | `captions_vllm_gemma4_nvfp4_clip_grounded_v2.jsonl` + metadata |

Open tags are sparse free-form discovery phrases, not exhaustive labels. The
accepted dictionary is derived from those open tags; captions are an
independent contrastive-training input.

## Historical compatibility files

`dictionary.json`, `dictionary_provenance.json`, and the short
`attributes_{train,dev}.jsonl` files remain because early tracked summaries
refer to them by path and hash. Compact results and dictionary artifacts retain
the prior edge-average comparison; its multi-megabyte tag and v9 label payloads
are retired from the active repository. Maintained defaults use
usage-profile-v8 and object-grounded-v11.

THINGS images live below ignored `raw/`; see the root README for download and
license instructions. Final test annotations belong below ignored `heldout/`.
