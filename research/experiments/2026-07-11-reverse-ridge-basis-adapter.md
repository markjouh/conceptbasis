# Pure reverse-ridge concept basis

## Decision

Proceed with the 30-epoch pure reverse model (`p0_l512`) as the main
development configuration. Its adapter objective contains only image-text
contrastive training and orthogonality of reverse-ridge partial effects. It has
no forward-probe, forward-orthogonality, or embedding-reconstruction loss.

The selected configuration is intentionally simpler than the numerically best
row of the broad sweep. `p1_l512` improves the sweep-average R@5 by only
`0.00027`; `p0_l512` is slightly better at fourteen concepts and removes an
entire auxiliary loss.

## Direction definition

For centered concept labels `Y` and centered adapter embeddings `Z`, solve once
per epoch over all 18,211 training images:

```text
B = (YᵀY / n + αI)⁻¹ YᵀZ / n
```

All 256 labels enter simultaneously. Row `B[k]` is the linear embedding
displacement associated with concept `k` while holding the other labeled
concepts fixed. This is the reverse orientation of a concept probe and differs
from a marginal positive-minus-negative group mean.

Training alternates minibatch contrastive optimizer steps with one exact
full-training-set reverse-ridge orthogonality step per epoch. Consequently,
`lambda_reverse_orth` is tied to the recorded batch size and update schedule;
it should not be interpreted as a batch-size-independent coefficient of one
monolithic loss.

Uncertain cells are mean-imputed per concept, making their centered target
zero. Rows with more than 25% uncertain cells are excluded from the ridge solve
but remain in contrastive training.

## Three-stage development comparison

The selected run uses `α=0.001`, reverse orthogonality weight 512, output
dimension 320, batch size 1,024, seed 0, and 30 epochs. The README comparison
uses every development class eligible at each attribute count, while retaining
all 278 classes in the retrieval gallery.

| Model and direction definition | Composition R@5 at 1 / 4 / 8 / 14 attributes |
|---|---:|
| Contrastive only, group-mean directions | .080 / .209 / .312 / .461 |
| Group-mean orthogonality | .113 / .298 / .447 / .686 |
| Pure reverse ridge `p0_l512` | **.121 / .360 / .587 / .844** |

The eligible class counts are 278 / 277 / 274 / 52 at those four points.
Rollout counts increase from 24 to 129 as the cohort shrinks, keeping the
number of sampled queries near 6,700 per point. All models receive identical
queries. Adapter profile construction uses CUDA when available; the small
278-item rank calculation remains a vectorized NumPy operation.

The selected pure model's reverse-direction RMS overlap is 0.06326. Its
14-attribute R@5 is 38.3 percentage points above contrastive-only directions
and 15.8 points above group-mean orthogonality.

## Scope and limitations

These results are development evidence, not a final confirmatory result. The
reverse-ridge hyperparameters were originally selected on the fixed 52-class,
24-rollout development benchmark, so the rebalanced three-model graph is not
an independent confirmation. The appropriate next check is a small multi-seed
rerun, followed by one sealed-test evaluation after the recipe is frozen.

Reverse effects are the principled directions for isolated embedding edits,
but the current benchmark evaluates retrieval from sums of standardized
attribute profiles. A future intervention benchmark should additionally test
whether adding one reverse direction changes its target attribute while
holding other predicted attributes and object identity fixed.

## Reproducibility

- Loss and estimator: `conceptbasis/losses.py`
- Trainer: `python -m conceptbasis.train --objective reverse-ridge`
- Pure sweep: `scripts/sweeps/sweep_reverse_ridge.py`
- Profile builder: `scripts/evaluation/build_retrieval_profiles.py`
- Unit test: `tests/test_reverse_ridge_loss.py`
- Tracked compact results: `research/results/reverse_ridge_dev_results.json`
- Tracked three-model comparison: `research/results/three_model_dev_composability.json`
- Matched ordinary retrieval: `research/experiments/2026-07-13-matched-retrieval-confirmation.md`
- Selected local checkpoint:
  `outputs/checkpoints_reverse_ridge/reverse_sweep_v1_p0_l512/`

Large checkpoints and profile arrays remain ignored and regenerable. Input
dictionary, label-table, and split-manifest hashes are recorded in both the
checkpoint configuration and the tracked compact result.
