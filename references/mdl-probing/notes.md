# MDL Probing (Voita & Titov 2020)


**Citation:** Elena Voita and Ivan Titov. "Information-Theoretic Probing with Minimum
Description Length." EMNLP 2020, pp. 183–196.
arXiv:2003.12298; ACL Anthology: 2020.emnlp-main.14.

- arXiv abstract: https://arxiv.org/abs/2003.12298
- ACL Anthology: https://aclanthology.org/2020.emnlp-main.14
- PDF in this folder: `voita-titov-2020-mdl-probing.pdf`

**Key idea.** Probe accuracy is a poor comparative metric: it does not substantially
favour pretrained over randomly-initialized representations, and accuracy on genuine
linguistic labels can match accuracy on random synthetic tasks — unless you artificially
constrain probe data or size. MDL probing replaces accuracy with the **description length
(codelength) of the labels given the representations**, which prices in the "amount of
effort" (model size and/or data) needed to achieve a given quality
([arXiv abstract](https://arxiv.org/abs/2003.12298)).

**Methodology (detail, from the PDF).** Transmitting labels y_{1:n} given representations
x_{1:n}: the trivial uniform code costs `L_unif = n·log K` (K classes), i.e. codelength
H(y) (paper §2; grep-verified in extracted text). Any better code yields compression vs.
this baseline, and the compression *gap* measures accessible information. Two practical
estimators, both droppable into standard probing pipelines, give agreeing results:

- **Variational coding** — explicitly prices the model (probe parameters) into the
  codelength; the MDL objective coincides with the loss of variational learning.
- **Online coding** (prequential) — transmits data in blocks, training the probe on past
  blocks to encode the next; no explicit model cost is needed.

**Methodological lesson.** When comparing *across* representations/embeddings, report an
information quantity (bits, or compression vs. the H(y) baseline), not raw accuracy —
accuracy saturates and can't separate "representation encodes X" from "probe learned X".
For LeJEPA embeddings, MDL probing with online coding is directly applicable to any
per-window property and gives a dataset-size-independent, comparable scale.
