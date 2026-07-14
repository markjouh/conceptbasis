# Matched ordinary-retrieval confirmation

## Question

Does either orthogonality objective reduce ordinary image-caption retrieval?
The earlier checkpoints were not a clean comparison: contrastive and
group-mean training ran for 60 epochs, reverse ridge ran for 30, and evaluation
used only the first 2,000 development pairs.

## Protocol

- production `conceptbasis.train` for every run;
- objectives: contrastive, group mean (`lambda_orth=8`), and reverse ridge
  (`alpha=0.001`, `lambda_orth=512`);
- reverse-schedule control with `lambda_orth=0`;
- seeds 0 through 4;
- 30 epochs, batch size 1,024, output dimension 320;
- identical optimizer, learning-rate schedule, initialization seed, and data
  order within each seed;
- all 3,914 development image-caption pairs;
- CUDA training and full-development evaluation;
- test classes remained sealed.

The reverse-schedule control performs the extra full-data ridge solve and
optimizer step but multiplies its orthogonality gradient by zero. It isolates
the learned reverse-orthogonality signal from the different update schedule.

## Results

Mean percent recall across five seeds (sample standard deviation in
parentheses):

| Objective | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| Contrastive | 57.675 (.567) | 88.983 (.249) | 96.919 (.141) |
| Group mean | 56.122 (.251) | 88.523 (.296) | 96.561 (.098) |
| Reverse schedule, zero orthogonality | 57.716 (.580) | 88.988 (.252) | 96.924 (.141) |
| Reverse ridge | **58.217 (.179)** | **89.980 (.110)** | **97.276 (.111)** |

Per-seed R@5:

| Seed | Contrastive | Group mean | Zero-orth schedule | Reverse ridge |
|---:|---:|---:|---:|---:|
| 0 | .891415 | .886817 | .891160 | .900358 |
| 1 | .888094 | .884773 | .888094 | .900102 |
| 2 | .893459 | .889116 | .893715 | .901124 |
| 3 | .888350 | .884262 | .888605 | .899080 |
| 4 | .887839 | .881196 | .887839 | .898314 |

Paired percentage-point differences, with exploratory 95% t intervals over
the five seed differences:

| Comparison | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| Group mean − contrastive | -1.553 [-2.339, -0.768] | -0.460 [-0.614, -0.306] | -0.358 [-0.562, -0.153] |
| Reverse ridge − zero-orth schedule | +0.501 [-0.315, +1.317] | **+0.991 [+0.778, +1.205]** | +0.353 [+0.138, +0.567] |

## Interpretation

Group-mean orthogonality causes a small but consistent ordinary-retrieval cost:
about 0.46 R@5 percentage points. Reverse ridge does not show a retrieval cost;
it improves R@5 by about one point in every seed. The zero-orthogonality
schedule differs from contrastive by only 0.005 R@5 points on average, so the
reverse result is not explained by its extra optimizer step.

This remains development evidence. Reverse-ridge hyperparameters were selected
using a compositional benchmark on these development classes, and five seeds do
not substitute for a sealed-test evaluation. The defensible claim is that no
ordinary-retrieval degradation was observed for the selected reverse-ridge
objective. Treat the apparent improvement as provisional until the recipe is
frozen and evaluated once on test classes.

## Local artifacts

Ignored checkpoints are named
`outputs/checkpoints/retrieval_confirm_s{0..4}_{objective}`. Each contains its
configuration, complete training history, final directions, and model weights.
