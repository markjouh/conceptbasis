# Dataset protocol

## Split unit

The unit of assignment is the THINGS object class. All full-set images in a
class and the class's one-image CC0 representative share one split. The
deterministic manifest is `data/splits.json` (seed 20260710):

| Split | Classes | Full images | Role |
|---|---:|---:|---|
| Train | 1,298 | 18,211 | Dictionary, directions, calibration, adapter training |
| Development | 278 | 3,914 | Model selection, sweeps, public preview |
| Test | 278 | 3,982 | Final confirmatory metrics only |

## Leakage rules

- Dictionary phrases and image-score profiles use train-class CC0 images only.
- GMM label calibrators fit on train-class scores and are applied unchanged to
  development and test.
- Image-anchored directions use train-class judgments only.
- Direction construction and hyperparameters may be selected on development.
- Public HTML galleries contain development CC0 images, not test images.
- Any script that renders, tags, profiles, or evaluates test requires
  `--allow-test`.
- Test tags are stored locally in `data/heldout/` and are not versioned during
  development.

## Historical caveat

The original exploratory study used the full CC0 set for dictionary discovery,
manual inspection, and evaluation. The class-level split was introduced after
that exploration, so the new test partition is prospectively isolated by the
pipeline but was not historically unseen by the researchers. A publication
claim should state this limitation or confirm the final result on a genuinely
new external cohort.
