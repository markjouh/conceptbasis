# Direct-composition training baseline

## Question

If compositional retrieval is the main downstream result, why not train that
operation directly instead of imposing orthogonality as a proxy?

## Prototype objective

This exploratory run adds one direct-composition update after the ordinary
image--caption contrastive minibatches in each epoch:

1. Fit the reverse-ridge concept directions on all training embeddings.
2. Sample up to 1,024 training images and, for each image, sample a random
   subset of 1--14 active dictionary labels.
3. Form the query by summing the selected directions after dividing each by
   its training projection standard deviation.
4. Use one-way cross-entropy to retrieve the exact source image from the
   sampled batch.

The run used seed 0, 30 epochs, output dimension 320, ridge alpha `0.001`, and
composition weight `1`. The contrastive-only and reverse-ridge orthogonality
comparators were matched seed-0, 30-epoch runs. Every composability evaluation
used reverse-ridge directions, so only adapter training differs.

The prototype sampled concepts from
`dictionary_labels_train_gemma26.jsonl`. Evaluation used the separately
recorded attribute lists mapped into the same dictionary. Thus it directly
optimizes the benchmark operation, but does not train on the evaluation query
lists themselves.

## Results

Development classes are disjoint from training classes. R@5:

| Concepts | Contrastive | Reverse-ridge orthogonality | Direct composition |
|---:|---:|---:|---:|
| 1 | .102 | .121 | **.130** |
| 2 | .168 | .208 | **.224** |
| 4 | .283 | .360 | **.393** |
| 6 | .391 | .485 | **.528** |
| 8 | .486 | .587 | **.634** |
| 10 | .577 | .685 | **.713** |
| 12 | .653 | .760 | **.784** |
| 14 | .780 | .844 | **.864** |

On the training split, direct composition also wins R@5 from `k=2` onward:

| Concepts | Contrastive | Reverse-ridge orthogonality | Direct composition |
|---:|---:|---:|---:|
| 1 | .036 | **.044** | .039 |
| 2 | .070 | .093 | **.095** |
| 4 | .144 | .199 | **.211** |
| 6 | .227 | .303 | **.331** |
| 8 | .311 | .406 | **.447** |
| 10 | .405 | .508 | **.554** |
| 12 | .489 | .598 | **.646** |
| 14 | .562 | .695 | **.705** |

Absolute train and development values are not directly comparable: the train
gallery contains 1,298 images and the development gallery contains 278.

The direct objective preserved ordinary development image--caption retrieval:
R@5 was `.893`, versus `.891` for contrastive-only and `.901` for
reverse-ridge orthogonality.

## Important qualifications

- This is one seed and one untuned composition weight.
- Exact-source cross-entropy creates false negatives when several images
  satisfy the same attribute subset. A multi-positive or set-based loss would
  be cleaner.
- The direct model wins development R@5 at every subset size, but at `k=14`
  reverse-ridge orthogonality has higher R@1 (`.602` versus `.466`). Direct
  composition instead has higher R@10 (`.957` versus `.889`).
- Direct training only modestly reduces direction overlap: off-diagonal cosine
  RMS is `.107`, versus `.115` for contrastive-only and `.063` for the
  orthogonality model. It appears to learn task-specific composition behavior
  without recovering the same geometric structure.

## Interpretation

Direct compositional supervision is a strong and necessary baseline. The
current evidence does not support claiming that orthogonality is the best way
to optimize compositional R@5. A defensible distinction is instead that direct
training is task-specific, whereas orthogonality explicitly improves concept
geometry and may matter for transfer, editing, or unseen composition regimes.
Those benefits need their own tests rather than being inferred from this
benchmark.

Compact metrics remain in
`research/results/direct_composition_baseline.json`; checkpoints and the large
legacy label input remain local historical artifacts. The exploratory runner
was retired when adapter training was consolidated under
`python -m conceptbasis.train`.
