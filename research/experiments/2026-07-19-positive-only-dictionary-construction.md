# Positive-only visual dictionary construction

## Decision

Use a 256-concept, positive-only dictionary built from the complete train-split
open-tag pass. Propose phrase merges liberally with frozen SigLIP 2 Giant text
cosine, filter every proposed edge in parallel with Gemma 4 26B NVFP4, and
apply weighted average-linkage agglomerative clustering to the approved graph.

The accepted candidate is `edge-average-085-v1`. It is a practical visual
concept dictionary rather than a symbolic ontology: perceptually neighboring
labels such as `brass`/`bronze`, shade variants, and material noun/adjective
forms may share a concept. Unrelated concepts that collide in an embedding or
co-occur in the images should not.

## Why the basis is positive-only

Every downstream fixed-dictionary judgment denotes whether an affirmatively
named trait is present, and the intended coefficient readout is nonnegative.
A low coefficient for `smooth` means little evidence for smoothness; it does
not assert the stronger proposition `rough`. If roughness is useful, `rough`
is therefore its own positive concept.

Encoding antonyms as negative poles introduced machinery that most concepts
do not need, made concepts compete for globally disjoint signed axes, and did
not match the later true/false labeling target. The production dictionary
consequently has no negative poles. Eight explicit lexical negations among the
eligible phrases were excluded: `non-edible`, `non-electronic`, `non-living`,
`non-metallic`, `non-reflective`, `non-transparent`, `unpolished`, and
`unstructured`.

## Discovery data

Gemma 4 26B NVFP4 produced open positive tags for all available images from
the 1,298 training classes. The source split contains 18,211 images. There are
18,210 successful tag rows and one retained refusal
(`chest1/chest1_14s.jpg`). No development or test tags enter construction.

Candidate phrases must occur in at least three images and at least three
training classes. This gives 793 candidates before the positive-only filter
and 785 after it, from 1,593 raw phrases. Support and prevalence are computed
per class before averaging, so classes with more photographed exemplars do not
dominate the dictionary.

## Merge algorithm

### 1. Liberal geometric proposals

Encode each phrase as `an object that is {phrase}` with the pinned
`timm/ViT-gopt-16-SigLIP2-384` release
`ad3410bee2c3373be5ed01e7c4e7fcd2bf95a183`. Propose every pair whose normalized
text-embedding cosine is at least `0.85`. This produces 20,881 candidate edges
over the 785 phrases.

The threshold is intentionally a high-recall gate, not the merge decision.
This preserves the computational advantage of cosine retrieval and avoids
asking the language model to reason over all 307,720 possible pairs.

### 2. Parallel semantic review

Gemma `nvidia/Gemma-4-26B-A4B-NVFP4` at revision
`a19cfe00be84568a6867111c9a68c9c44fdcffe6` reviews compact batches of 64 edges
with temperature zero, no reasoning trace, and an exact JSON boolean contract.
The `positive-family` instruction accepts:

- synonyms and spelling or inflection variants;
- material noun/adjective variants;
- light, dark, or intensity variants of one property;
- extremely close colors or materials whose separation adds little practical
value.

The accepted dictionary and build provenance are tracked as
`data/dictionary_positive_only_edge_average_085.json` and
`data/dictionary_positive_only_edge_average_085.provenance.json`. The larger
proposal, adjudication, cluster, and preview artifacts remain local under
`outputs/`.

It rejects co-occurrence, independent dimensions, opposites, and merely
thematic or taxonomic relations. The instruction includes positive examples
such as `brown`/`light brown`, `yellow`/`yellowish`, `blue`/`light blue`, and
`brass`/`bronze`, and collision counterexamples such as
`rigid`/`heavy`, `smooth`/`soft`, `natural`/`perishable`, and
`metallic`/`shiny`.

Gemma approved 1,607 of the 20,881 proposals (7.70%). The observed run took
105.2 seconds, or 198.5 edge decisions per second. The proposal hash, model
revision, full instruction, instruction hash, and every individual judgment
are retained in the local adjudication artifact.

The v1 run sidecar records batch size but, regrettably, not worker count or
retry count. Those settings affect wall time rather than the temperature-zero
edge contract. The elapsed time and throughput above come from the interactive
run log rather than the adjudication JSON. Future run metadata should preserve
all concurrency settings and elapsed time directly.

### 3. Weighted average linkage

Let `a_ij` be Gemma's approval and `c_ij` the SigLIP text cosine. The
agglomerative similarity matrix is

```text
s_ii = 1
s_ij = c_ij  if the edge was proposed and a_ij = true
s_ij = 0     otherwise
```

Clustering uses average linkage on `d_ij = 1 - s_ij` and stops at distance
`0.75`, equivalently a minimum average approved-edge similarity of `0.25` at
each merge. Two approved singleton phrases can merge immediately, while a
larger union needs enough approved cross-cluster edges to overcome the zeroes
from rejected or unproposed pairs. This is more permissive than complete
linkage without reducing the graph to unconstrained transitive closure.

The procedure yields 287 clusters. They are ranked by equal-class-weighted
prevalence, class support, and image support; the top 256 are retained. The
final dictionary contains 752 unique positive phrases, no repeated phrase
assignments, no negative-pole fields, and a maximum cluster size of 10.

## Qualitative outcome

The intended shade merges are present:

- `brown`: beige, blonde, brown, brownish, dark brown, light brown, tan;
- `yellow`: pale yellow, yellow, yellowish;
- `blue`: blue, blueish, cyan, dark blue, light blue, teal, turquoise.

The liberal policy also creates broad but visually defensible families, for
example the ten-term red/pink/purple family. A few boundary cases remain useful
to report rather than conceal: `dense` joins the large/thick family, and the
red family sacrifices fine hue distinctions. These reflect the stated goal of
a compact practical basis, not strict lexical equivalence. The HTML preview
was inspected with image galleries for all 256 concepts and accepted as the
working dictionary candidate.

## Alternatives tried and rejected

- **Usage-vector clustering alone.** Visually unrelated phrases can share an
  image-class usage pattern. The observed `rigid`/`lightweight` merge is the
  canonical collision. Class balancing does not make this a semantic test.
- **Strict text-semantic merging.** High thresholds and strict equivalence
  preserve precision but leave obviously redundant shade variants such as
  `brown`/`light brown` and `blue`/`light blue` separate.
- **Union-find on accepted edges.** A single false-positive bridge can join two
  otherwise unrelated connected components. It has no cluster-level density
  check.
- **Binary-weight agglomeration.** Giving every approved edge weight one
  creates large tie sets and discards the useful ordering supplied by SigLIP
  cosine. Retaining cosine weights produced stable intended color merges.
- **Iterative cluster-level LLM merging.** This is sequential, parallelizes
  poorly, repeatedly spends context on the same members, and produced an
  over-broad 31-term cluster in the pilot.
- **Signed antonym axes.** They complicate term ownership and downstream
  coefficients while applying meaningfully to only a minority of concepts.

## Reproduction

The important invocations are:

```bash
python scripts/dictionary/propose_merge_edges.py \
  --open-tags data/attributes_train_full_vllm_gemma4_nvfp4.jsonl \
  --out outputs/dictionary_candidates/siglip2-gopt-p16-384@ad3410b/edge-average-085-v1/proposals.json \
  --encoder siglip2-giant --min-mentions 3 --min-class-support 3 \
  --text-cosine 0.85 --precision fp16 --batch-size 512

python scripts/dictionary/adjudicate_merge_edges.py \
  --proposals outputs/dictionary_candidates/siglip2-gopt-p16-384@ad3410b/edge-average-085-v1/proposals.json \
  --out outputs/dictionary_candidates/siglip2-gopt-p16-384@ad3410b/edge-average-085-v1/adjudication.json \
  --model nvidia/Gemma-4-26B-A4B-NVFP4 \
  --model-revision a19cfe00be84568a6867111c9a68c9c44fdcffe6 \
  --policy positive-family --batch-size 64

python scripts/dictionary/build_dictionary.py \
  --open-tags data/attributes_train_full_vllm_gemma4_nvfp4.jsonl \
  --img-dir data/raw/object_images \
  --out outputs/dictionary_candidates/siglip2-gopt-p16-384@ad3410b/edge-average-085-v1/dictionary.json \
  --provenance-out outputs/dictionary_candidates/siglip2-gopt-p16-384@ad3410b/edge-average-085-v1/provenance.json \
  --cluster-candidates-out outputs/dictionary_candidates/siglip2-gopt-p16-384@ad3410b/edge-average-085-v1/all_clusters.json \
  --encoder siglip2-giant --k 256 --min-mentions 3 --min-class-support 3 \
  --merge-method adjudicated \
  --merge-adjudication outputs/dictionary_candidates/siglip2-gopt-p16-384@ad3410b/edge-average-085-v1/adjudication.json \
  --adjudicated-linkage average --adjudicated-min-similarity 0.25 \
  --selection-method frequency --remove-components 1 --split train \
  --precision fp16 --batch-size 512
```

Large proposal, adjudication, dictionary, provenance, cluster, and preview
artifacts remain under the ignored `outputs/` tree. Their compact hashes and
metrics are tracked in
`research/results/positive_only_dictionary_edge_average_085.json`.

## Fixed-dictionary labeling

The accepted dictionary was subsequently labeled on all 18,211 train images
and all 3,914 development images. The sealed test split was not touched.

Only the 256 canonical cluster leaders were supplied to Gemma. The 752 merged
member phrases remain construction evidence and are not alternative labeling
targets. This distinction matters: an initial member-aware pilot caused the
canonical `appliance` label to fire for a blazer because its cluster also
contained `household item`. The abandoned 4,225-row partial is retained under
`outputs/labeling_experiments/` rather than mixed with production data.

The production prompt uses each leader's ordinary literal meaning, temperature
zero, reasoning disabled, strict JSON-schema decoding, original local image
files, and the 280-visual-token quality profile. It records `present` and
`uncertain` separately; omitted concepts are NO. Manual review found generally
good precision but conservative recall. Examples included missing `spotted`
for a cheetah and `mammal` for a dolphin, while occasional associations such
as `percussive` for a cello remained false positives.

Both splits completed without a failed row or retry. Train took 1,346.4
seconds at 13.525 images/s; development took 278.9 seconds at 14.032 images/s.
Every image ID is unique and exactly matches its split, all returned labels are
canonical, no row overlaps `present` and `uncertain`, and every row has at
least one present concept. Across train and development, 250 of 256 leaders
appear at least once as present and 255 appear as present or uncertain. The
sole never-used leader is `four-legged`, consistent with the observed
conservative recall.

Compact metrics and hashes are tracked in
`research/results/positive_only_dictionary_labels_leaders_v2.json`.

### Interim exhaustive relabel continuation

The later exhaustive relabel is intentionally an interim three-shard train set.
Development and the first 5,055 train images used the verbose
`N. concept: YES/NO` checklist; the next 13,095 train images use the faster
all-256 `concept: YES/NO` checklist, with no cosine pruning. The final 61 exact-
name failures use the same checklist under regex-constrained decoding. Each
continuation records and skips its immutable predecessor shards. Their union is
all 18,211 train images with no missing, extra, or duplicate IDs. These interim
annotations should be regenerated end to end under one frozen dictionary and
prompt before the final paper release.
