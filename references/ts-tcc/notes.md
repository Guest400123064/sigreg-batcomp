# TS-TCC — Time-Series Representation Learning via Temporal and Contextual Contrasting

- **Citation:** Emadeldeen Eldele, Mohamed Ragab, Zhenghua Chen, Min Wu,
  Chee Keong Kwoh, Xiaoli Li, Cuntai Guan. IJCAI 2021, pp. 2352–2359.
  arXiv:2106.14112.
- **URLs:** <https://arxiv.org/abs/2106.14112> ·
  <https://doi.org/10.24963/ijcai.2021/324> ·
  [dblp](https://dblp.org/rec/conf/ijcai/Eldele0C000G21)
- **Key idea:** Two correlated views of the same series are produced via *weak* and
  *strong* augmentations. (1) A *temporal contrasting* module sets up a tough
  cross-view prediction task (predict the future of one augmentation view from the
  other, in latent space, with an autoregressive module) to learn robust temporal
  representations. (2) A *contextual contrasting* module then maximizes agreement
  between the contexts of the two views of the same sample while pushing apart
  contexts of different samples. (Verified from the arXiv abstract.)
- **What the embeddings capture (per the paper):** Robustness to the chosen
  augmentations (invariance to jittering/scaling/permutation-style transforms) plus
  temporal predictive structure. Reported: a linear classifier on frozen TS-TCC
  features performs comparably to fully supervised training on three real-world
  datasets, with strong few-label and transfer-learning results — evidence that
  class-discriminative information is linearly accessible in the embeddings.
