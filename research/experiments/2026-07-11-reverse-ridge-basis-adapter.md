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

## Selected development result

The selected run uses `α=0.001`, reverse orthogonality weight 512, output
dimension 320, batch size 1,024, seed 0, and 30 epochs.

| Model | Reverse RMS overlap | Ordinary dev R@5* | Composition R@5 at 1 / 4 / 8 / 14 concepts |
|---|---:|---:|---:|
| Contrastive control, post-hoc reverse directions | 0.10485 | 0.891 | .107 / .284 / .489 / .730 |
| Pure reverse `p0_l512` | **0.06326** | **.8965** | **.139 / .393 / .627 / .845** |
| Reverse + probe `p1_l512` | 0.06291 | .8970 | .144 / .395 / .629 / .841 |
| Reverse + probe `p8_l128` | 0.07247 | .8990 | .133 / .377 / .609 / **.863** |

`*` The recorded sweep checkpoint used the legacy first-2,000 dev retrieval
diagnostic. The refactored trainer evaluates the complete development set.

The pure model reduces reverse-direction RMS overlap by about 40% relative to
the matched contrastive control. Its ordinary image-text retrieval is
effectively unchanged, while the compositional advantage grows with the number
of attributes. The fourteen-concept R@5 gain is 11.5 percentage points.

## Scope and limitations

These results are development evidence, not a final confirmatory result. The
sweep used one training seed and selected hyperparameters on 52 development
source images with 24 correlated rollouts per image. The appropriate next
check is a small multi-seed rerun of the pure configuration, followed by one
sealed-test evaluation after the recipe is frozen.

Reverse effects are the principled directions for isolated embedding edits,
but the current benchmark evaluates retrieval from sums of standardized
attribute profiles. A future intervention benchmark should additionally test
whether adding one reverse direction changes its target attribute while
holding other predicted attributes and object identity fixed.

## Reproducibility

- Loss and estimator: `conceptbasis/losses.py`
- Trainer: `python -m conceptbasis.train --objective reverse-ridge`
- Pure sweep: `scripts/sweeps/sweep_reverse_ridge.py`
- Profile builder: `scripts/evaluation/build_groupmean_profiles.py`
- Unit test: `tests/test_reverse_ridge_loss.py`
- Tracked compact results: `research/results/reverse_ridge_dev_results.json`
- Selected local checkpoint:
  `outputs/checkpoints_reverse_ridge/reverse_sweep_v1_p0_l512/`

Large checkpoints and profile arrays remain ignored and regenerable. Input
dictionary, label-table, and split-manifest hashes are recorded in both the
checkpoint configuration and the tracked compact result.
