# Local VLM selection for dictionary labeling

**Decision (2026-07-11):** use `google/gemma-4-26b-a4b-qat` in LM Studio,
with reasoning disabled, temperature 0, images resized to a 512-pixel maximum
side, and four concurrent requests. Keep separate `present` and `uncertain`
labels. Do not use Qwen 35B for the production run.

## Why Gemma

Matched local pilots showed that Qwen 35B had somewhat higher recall, but was
more prone to background leakage, repeated output, and failure to obey the
exact-name response contract. Gemma 26B was more conservative and materially
more stable. This is preferable for fitting concept probes: abstentions remain
auditable in `uncertain`, while false-positive labels directly contaminate the
regression targets.

Reasoning is disabled because both local models could spend the entire output
budget reasoning over a monolithic 256-concept request without returning the
JSON answer. The compact non-reasoning prompt was both faster and more stable.

## Validated runtime

The final LM Studio configuration sustained 100/100 valid responses on a
100-image run at 125.3 seconds total (1.253 effective seconds per image). A
fresh 12-image durability smoke test completed 12/12 at 1.137 effective seconds
per image; an immediate rerun correctly resumed with zero remaining work.

At the sustained 100-image rate, the 18,211-image train split is approximately
6.3 hours. This is a local run with no per-request API charge.

## Production commands

```bash
lms unload -a
lms load google/gemma-4-26b-a4b-qat --gpu max \
  --context-length 8192 --parallel 4 --ttl 28800 \
  --identifier google/gemma-4-26b-a4b-qat -y

python scripts/data/label_fixed_dictionary.py
```

The runner was append-only and resumable. It recorded successful annotations in
`data/dictionary_labels_train_gemma26.jsonl`, errors separately, and the exact
prompt, dictionary hash, split hash, selected-image hash, model, and inference
settings in a metadata sidecar. It refuses incompatible resumes and refuses to
adopt an existing output whose sidecar is missing. Unknown noncanonical names
are excluded from labels but retained in `unknown_names` for audit.

## Completion and recovery

The temperature-0 production pass produced 18,119 successful rows and 92
unique invalid-JSON failures. A deterministic retry recovered 3. The remaining
89 were recovered with temperature 0.15, top-p 0.9, and repetition penalty
1.08 using the archived one-off recovery script. A matched pilot
on eight stubborn rows succeeded 8/8 before the recovery pass was launched.
Each recovered row records its sampling settings; the original error log is
retained locally under
`archive/2026-07-11-regression-experiments/data/` as historical provenance.

The final audit found exactly 18,211 expected train IDs, 18,211 unique successful
rows, no missing or extra IDs, no unresolved recovery errors, and no invalid
canonical-label rows. Dictionary, split-manifest, and selected-image hashes all
match the frozen metadata.

The multi-megabyte JSONL payload is now a local historical artifact rather than
a tracked input to the maintained pipeline; its metadata sidecar and the
compact result record remain in Git.
