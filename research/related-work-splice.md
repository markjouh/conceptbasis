# Related work: SpLiCE and ConceptBasis

## Paper positioning

SpLiCE belongs in related work, not as the paper's antagonist or the work that
ConceptBasis directly extends. The shared surface feature is a concept
dictionary, but the projects ask different questions:

- SpLiCE performs a post-hoc sparse reconstruction of frozen embeddings;
- ConceptBasis trains the representation itself for measurable, predictable
  attribute composition;
- SpLiCE uses a broad, highly overcomplete lexical dictionary, whereas
  ConceptBasis uses a curated dictionary of abstract visual attributes;
- SpLiCE aims to explain an existing embedding, whereas ConceptBasis aims to
  change its geometry.

A concise comparison is: *SpLiCE uses a text dictionary for post-hoc sparse
reconstruction of frozen embeddings. We instead train an embedding space whose
curated attribute coefficients are meaningful and whose directions compose
predictably.* Briefly note the coefficient-identifiability limitation below,
then move to the positive motivation rather than framing the paper as a
rebuttal.

## SpLiCE criticism

SpLiCE embeds a large vocabulary of words and short phrases with CLIP, then uses nonnegative LASSO to find a sparse reconstruction of an image embedding. The main method does not train CLIP or separate the dictionary directions.

The reconstruction objective identifies the product `Cw`, not the semantic meaning of each coefficient in `w`. In an overcomplete, correlated dictionary:

- one concept can be replaced by a nearby atom;
- several atoms can combine to reproduce another concept direction;
- an atom can be selected because one of its components helps reconstruct something else;
- small changes to the input or dictionary may change the selected atoms without changing the reconstruction much.

Therefore, a coefficient is a reconstruction weight, not necessarily a measure of how much of the named concept is present. A zero coefficient does not imply absence, and zeroing one coordinate does not necessarily remove that information.

This is not a criticism that SpLiCE should have changed CLIP's geometry: it is deliberately post-hoc. The limitation is that sparse reconstruction alone cannot guarantee identifiable concept coordinates, especially when the coefficients are used for prevalence estimates or interventions.

## ConceptBasis flow

1. **Group means are confounded.** If apples in the data are commonly red and round, the group-mean direction for `red` can contain components of `round` and `apple`.

2. **Reverse ridge estimates partial effects.** Joint regression estimates the embedding change associated with `red` while holding the other labeled concepts fixed.

3. **Partial-effect directions can still overlap.** Even after controlling for observed correlations, unrelated concept effects may have nonzero inner products, causing cross-talk during composition or editing. Representational superposition is one possible explanation, but it needs citations and experimental evidence; incomplete labels and genuinely shared visual structure are alternatives.

4. **Orthogonality reduces residual cross-talk.** The orthogonality loss pressures the learned partial-effect directions to interfere less, while contrastive training preserves ordinary image-text retrieval.

The current model outputs a dense embedding with conditional concept-effect directions inside it. It does not yet output calibrated per-image concept coefficients, so the paper should claim more meaningful directions or coordinates unless a coefficient readout is explicitly defined and evaluated.
