# Class-disjoint composability development evaluation

This rerun fixes object-class leakage across dictionary construction, adapter
training, and evaluation. The deterministic split contains 1,298 train, 278
development, and 278 test THINGS classes. The test partition was not opened.

The dictionary was constructed from train-class CC0 tags only. Image anchors
and label calibration used train classes only; direction-construction recipes
were selected on development; both adapters were trained on the same 18,211
train images with seed 0. The only objective difference was the smooth
correlation-weighted orthogonality term (`lambda_orth=8`, `tau=0.15`, power 4,
floor 0.01).

Additive retrieval used the 278-class development CC0 gallery. To compare all
values of k on the same cohort, queries were restricted to the 52 development
classes with at least 14 mapped, non-flagged dictionary attributes. Each point
averages 24 identical nested subset rollouts per query class.

| Model | Image–text R@5 | Additive R@10, k=14 | Median rank, k=14 |
|---|---:|---:|---:|
| Frozen backbone | — | 54.9% | 8 |
| Contrastive adapter | 89.1% | 57.7% | 6 |
| Contrastive + orthogonality | 88.7% | 77.4% | 2 |

The orthogonality model improves additive R@10 by 19.7 percentage points (34%
relative) over the matched contrastive-only adapter, while development
image–text R@5 changes by -0.4 percentage points. These are development
results, not final confirmatory estimates. The original exploratory work had
already exposed the researchers to the full CC0 set; see `research/DATA_PROTOCOL.md`
for the historical caveat and external-cohort recommendation.

Machine-readable metrics: `research/results/classsplit_dev_composability.json`
and `research/results/classsplit_dev_training.json`.
