# SigLIP2 Giant adapters for the positive-only dictionary

## Decision

Use the exhaustive-label reverse-ridge adapter with ridge alpha `0.01` and
orthogonality weight `1024` as the working adapter for the new SigLIP2 Giant /
positive-only dictionary stack. It preserves ordinary image-caption retrieval,
gives the highest balanced development soft-label AUROC, and improves
multi-concept composition over the original reverse-ridge checkpoint.

This is a development decision, not a sealed-test result. The test split was
not read.

## Frozen inputs

- Encoder: `ViT-gopt-16-SigLIP2-384` / `webli`, Hugging Face revision
  `ad3410bee2c3373be5ed01e7c4e7fcd2bf95a183`.
- Dictionary: accepted 256-concept positive-only `edge-average-085-v1`, SHA-256
  `8df7ad33155bc88366091d0629dccfbd52ad6e566cfcedc6b5c4fbec0bf288f1`.
- Hard labels: complete canonical-leader Gemma train pass, SHA-256
  `85ddfc1f4c10d8dc14dbddd8f767eaaedfb25fec3fa1bccbe3c85e88b54c3c50`.
- Soft labels: train-fitted two-component GMM calibration of SigLIP2 image
  projections onto member-averaged text directions. The output contains
  exactly 256 `s_<leader>` columns in dictionary order.
- Split sizes: 18,211 train, 3,914 development, 3,982 sealed test.

The dictionary-bound frozen cache is isolated at
`outputs/training_inputs/siglip2-gopt-p16-384@ad3410b/edge-average-085-v1/`.
Its full-image encoder pass sustained 107.786 img/s at native 384 px with
batch size 16. The cache manifest binds the encoder release, image order,
split manifest, dictionary, and artifact hashes.

The trainer was changed to make the text encoder release explicit. Candidate
caption caches now carry an encoder/image-order/caption hash manifest and are
rejected if that identity changes. This prevents a SigLIP2 image cache from
being silently paired with legacy PE text embeddings.

## Matched training

The three incremental objectives used identical cached inputs, adapter shape,
optimizer, schedule, seed, and number of contrastive updates:

- adapter output 320, hidden width 1,024;
- 30 epochs, batch 1,024, 540 minibatch steps;
- AdamW, learning rate 0.001, weight decay 0.0001, cosine schedule;
- seed 0;
- contrastive only: orthogonality weight 0;
- group mean: smooth correlation-weighted orthogonality, weight 8;
- reverse ridge: alpha 0.001, orthogonality weight 512, one exact full-train
  ridge update per epoch.

Training the adapters took 3.9, 4.6, and 4.5 seconds respectively after the
frozen caches existed.

## Development results

Ordinary paired retrieval and soft-label separation:

| Objective | R@1 | R@5 | R@10 | mean AUROC | direction RMS overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Contrastive | .627 | .915 | .982 | .828 | .296 |
| Group mean | .607 | .905 | .982 | .827 | .202 |
| Reverse ridge | .624 | .915 | .981 | **.854** | **.0624** |

Class-disjoint CC0 compositional retrieval uses the new Gemma open tags and
the same fixed random rollouts for every model. The table reports R@5:

| Objective / directions | k=1 | k=4 | k=8 | k=10 | k=12 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen SigLIP2, group mean | .081 | .105 | .099 | .088 | .114 |
| Contrastive, group mean | .092 | .188 | .228 | .258 | .278 |
| Group-mean orthogonality | .107 | .261 | .362 | .425 | .474 |
| Reverse ridge | **.113** | **.333** | **.588** | **.716** | **.755** |

The eligible development cohort is 278 images through `k=6`, 277 at `k=8`,
234 at `k=10`, and 66 at `k=12`. Only four images have 14 mapped concepts, so
the `k=14` point is retained in the machine-readable artifact but should not
be interpreted as a stable estimate. Development images have 10.67 mapped
concepts on average.

## Original sparse-support caveat

Eight canonical leaders have no positive train labels: `four-legged`, `wire`,
`veined`, `modular`, `insulated`, `gaseous`, `seasonal`, and `minimalist`.
Reverse ridge correctly excludes them from its supported-pair orthogonality
loss and emits zero directions for them. Four of these names occur in the
development open tags (seven total tag occurrences), so the reported reverse
result includes rather than hides this limitation. Group-mean directions
remain available for all 256 concepts from the soft labels.

The clean next step is a targeted support pass or an explicitly documented
group-mean fallback for unsupported reverse directions; it is not appropriate
to silently treat zero evidence as a learned partial effect.

## Artifacts

- Original seed-0 reference checkpoint:
  `outputs/checkpoints/siglip2_giant_edgeavg085_reverse_ridge_s0/ckpt.pt`
- Matched checkpoints:
  `outputs/checkpoints/siglip2_giant_edgeavg085_{contrastive,group_mean}_s0/`
- Profile cache: `outputs/evals/siglip2_giant_edgeavg085_s0_profiles.npz`
- Full development evaluation:
  `outputs/evals/siglip2_giant_edgeavg085_s0_composability.json`
- Compact tracked result:
  `research/results/siglip2_positive_dictionary_adapter_dev.json`

Large arrays and checkpoints remain ignored and regenerable.

## Provisional rerun with exhaustive train labels

After the exhaustive relabel, the three seed-0 objectives were rerun with the
same frozen caches and training protocol. The merged train artifact contains
all 18,211 train images, averages 33.61 positive concepts per image (previously
12.10), and gives every dictionary concept positive support. It intentionally
preserves the completed numbered shard and uses the finalized named-binary
format only for the continuation, so this is an interim mixed-prompt training
artifact rather than the final paper release artifact.

Only reverse ridge consumes these hard labels. As an internal reproducibility
check, the contrastive and group-mean runs reproduced their previous metrics
exactly. Reverse ridge changed as expected:

| Reverse-ridge metric | Previous labels | Exhaustive labels | Delta |
| --- | ---: | ---: | ---: |
| Development R@1 | .624 | .634 | +.010 |
| Development R@5 | .915 | .918 | +.004 |
| Development R@10 | .981 | .983 | +.002 |
| Development mean AUROC | .854 | .857 | +.002 |
| Direction RMS overlap | .0624 | .0550 | -.0074 |
| Explained fraction | .225 | .339 | +.114 |

The same compositional-development protocol gives a mixed result:

| Reverse-ridge directions | k=1 | k=2 | k=4 | k=6 | k=8 | k=10 | k=12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous labels | .113 | .203 | .333 | .467 | .588 | .716 | .755 |
| Exhaustive labels | .110 | .180 | .321 | .446 | .576 | .709 | **.813** |

The denser labels therefore improve support and the learned geometry, and they
produce a large high-order gain at `k=12`, but they slightly reduce R@5 from
`k=2` through `k=10`. At this stage the previous checkpoint remained the
conservative working selection pending a regularization sweep. That
provisional judgment is superseded by the sweep below. No sealed-test data was
read.

- Candidate checkpoint:
  `outputs/checkpoints/siglip2_giant_edgeavg085_interim_v9_reverse_ridge_s0/ckpt.pt`
- Full matched evaluation:
  `outputs/evals/siglip2_giant_edgeavg085_old_vs_interim_v9_composability.json`
- Compact tracked result:
  `research/results/siglip2_positive_dictionary_adapter_interim_v9_dev.json`

## Exhaustive-label reverse-ridge sweep

The relabel changed positive prevalence enough that the original
`alpha=0.001`, `lambda=512` setting was no longer assumed optimal. A 24-run
broad seed-0 grid crossed ridge alpha `{0.0001, 0.001, 0.01, 0.1}` with
orthogonality weight `{32, 64, 128, 256, 512, 1024}`. A 12-run focused grid
then crossed alpha `{0.02, 0.03, 0.05}` with weight
`{512, 768, 1024, 1536}`. All other inputs and optimization settings were
fixed, and the sealed test split was not read.

The compositional maximum (`alpha=0.1`, weight `512`) noticeably degraded
ordinary retrieval. The balanced Pareto choice is `alpha=0.01`, weight `1024`:

| Seed-0 metric | Original checkpoint | Exhaustive, untuned | Exhaustive, tuned |
| --- | ---: | ---: | ---: |
| Development R@1 | .624 | **.634** | .624 |
| Development R@5 | .915 | .918 | **.919** |
| Development R@10 | .981 | **.983** | **.983** |
| Development mean AUROC | .854 | .857 | **.858** |
| Direction RMS overlap | .0624 | **.0550** | .0568 |

The same fixed compositional rollouts report R@5:

| Directions | k=1 | k=2 | k=4 | k=6 | k=8 | k=10 | k=12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original checkpoint | .113 | **.203** | .333 | .467 | .588 | .716 | .755 |
| Exhaustive, untuned | .110 | .180 | .321 | .446 | .576 | .709 | .813 |
| Exhaustive, tuned | **.115** | .196 | **.356** | **.493** | **.616** | **.750** | **.877** |

Paired rollouts show that the tuned checkpoint improves the source-image rank
more often than it worsens it at every evaluated subset size from `k=1`
through `k=12`; the small `k=2` R@5 decrease is therefore a threshold effect,
not a general rank regression.

Four additional seeds confirmed the selected setting. Across five seeds,
ordinary development R@5 is `.920 ± .002`, AUROC is `.858 ± .001`, and
composition R@5 is `.358 ± .001` at `k=4`, `.615 ± .005` at `k=8`,
`.730 ± .015` at `k=10`, and `.830 ± .040` at `k=12`. The higher variance at
`k=12` is consistent with its smaller 66-image cohort.

- Selected seed-0 checkpoint:
  `outputs/checkpoints/siglip2_giant_edgeavg085_interim_v9_rr_a0p01_l1024_s0/ckpt.pt`
- Selected-comparison evaluation:
  `outputs/evals/siglip2_giant_edgeavg085_rr_selected_comparison_composability.json`
- Five-seed evaluation:
  `outputs/evals/siglip2_giant_edgeavg085_interim_v9_rr_a0p01_l1024_seeds_composability.json`
- Compact tracked sweep result:
  `research/results/siglip2_positive_dictionary_adapter_interim_v9_sweep_dev.json`

## Exhaustive CC0 development evaluation through k=20

The original open-tag evaluation supplied at most 14 mapped concepts per CC0
image. To evaluate longer queries without changing the dictionary, all 278
development CC0 images were labeled against the fixed 256-concept checklist
using the finalized named-binary Gemma prompt. The merged artifact has 33.65
positive concepts per image on average (minimum 18, median 33, maximum 57).
Every image is therefore eligible through `k=18`, and 276 of 278 are eligible
at `k=20`.

The chart uses five matched seeds for every objective and 24 fixed-seed
rollouts per eligible image. Mean compositional Recall@5 is:

| Objective | k=1 | k=4 | k=8 | k=12 | k=16 | k=20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Contrastive | .063 | .115 | .149 | .163 | .171 | .180 |
| Group mean | .074 | .159 | .226 | .267 | .286 | .303 |
| Tuned reverse ridge | **.096** | **.296** | **.524** | **.689** | **.796** | **.873** |

Ordinary development Recall@5 is `.916 ± .001` for contrastive, `.909 ± .003`
for group mean, and `.920 ± .002` for tuned reverse ridge. This evaluation uses
only development labels and does not read the sealed test split.

- Exhaustive CC0 labels SHA-256:
  `57e9df20c61cfc205bf5ea5b3a84101b13d9afaa337bfe290727d769d1e54e49`
- Full evaluation:
  `outputs/evals/siglip2_giant_edgeavg085_exhaustive_cc0_dev_k20_seed_composability.json`
- Compact tracked summary:
  `research/results/siglip2_positive_dictionary_exhaustive_cc0_dev_k20.json`
- Regenerated chart:
  `docs/assets/composability-retrieval.svg`
