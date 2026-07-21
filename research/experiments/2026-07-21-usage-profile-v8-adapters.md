# Usage-profile-v8 adapter evaluation

This run promotes the simpler usage-profile dictionary built from the final
Gemma open-tag release. Phrase profiles are class-balanced tag-usage vectors;
correlated phrases are clustered and the 256 retained clusters define a
positive-only dictionary. No LLM merge reviewer or canonicalizer is used.

All train images and the 278 CC0 development representatives were relabeled
against the fixed dictionary with the object-grounded-v11 checklist prompt.
It judges only the named object, excludes background and source/whole transfer,
and emits one binary decision per concept. Development images have a mean of
22.77 positive concepts (minimum 10, median 22, maximum 49).

The matched experiment uses the pinned SigLIP2 Giant encoder, Gemma grounded
captions, 30 training epochs, and five seeds for each incremental objective.
At ten attributes, where all 278 development images are eligible, mean
compositional Recall@5 is:

| Objective | Recall@5 |
| --- | ---: |
| Contrastive | .198 |
| Group-mean orthogonality | .361 |
| Reverse-ridge orthogonality | **.745** |

Ordinary development Recall@5 is .970, .968, and .969 respectively. At 20
attributes reverse ridge reaches .945, but only 202 images have at least 20
positive labels; the tracked summary records cohort size at every query length.

- Dictionary SHA-256: `dca532bfadaa803b351042fcef0f325efa6a86722bff7f54b493a58d8700c5f1`
- Train fixed-label SHA-256: `7deaa3ea63226900c5e5372935e5978d888d426b269d1e47e291d0abccdafd9a`
- Development fixed-label SHA-256: `91268700b46d18f2b433c3edf13647054a838e6a4b6a901e7a582d1c75ca6f4a`
- Compact result: `research/results/siglip2_usage_profile_v8_v11_exhaustive_cc0_dev_k20.json`
