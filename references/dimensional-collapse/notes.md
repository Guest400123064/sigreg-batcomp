# Dimensional Collapse in Contrastive SSL (Jing et al. 2022)


**Citation:** Li Jing, Pascal Vincent, Yann LeCun, and Yuandong Tian.
"Understanding Dimensional Collapse in Contrastive Self-supervised Learning."
ICLR 2022. arXiv:2110.09348.

- arXiv abstract: https://arxiv.org/abs/2110.09348

**Key idea.** Joint-embedding methods must fight collapse; beyond complete collapse
(constant output) there is **dimensional collapse**, where embeddings span only a
lower-dimensional subspace. The paper shows dimensional collapse occurs in *contrastive*
learning too (not just non-contrastive methods), analyzes the dynamics causing it, and
derives DirectCLR, which optimizes the representation space directly without a trainable
projector and beats SimCLR-with-projector on ImageNet
([arXiv abstract](https://arxiv.org/abs/2110.09348)).

**Methodological lesson.** An embedding can look fine on probes while silently using only a
small subspace: probe accuracy is blind to *how many dimensions* carry the information.
Diagnosis: inspect the singular-value spectrum of the embedding covariance — a steep drop
to zero indicates dimensional collapse. For LeJEPA-style time-series pretraining, spectral
analysis complements RankMe (§7) and explains *why* an embedding contains little
information, and motivates which subspace probes actually act on.
