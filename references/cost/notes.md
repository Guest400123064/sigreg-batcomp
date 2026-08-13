# CoST — Contrastive Learning of Disentangled Seasonal-Trend Representations

- **Citation:** Gerald Woo, Chenghao Liu, Doyen Sahoo, Akshat Kumar, Steven C. H. Hoi.
  ICLR 2022. arXiv:2202.01575; OpenReview `PilZY3omXV2`.
- **URLs:** <https://arxiv.org/abs/2202.01575> ·
  [OpenReview](https://openreview.net/forum?id=PilZY3omXV2) ·
  [dblp](https://dblp.org/rec/conf/iclr/WooLSKH22)
- **Key idea:** Forecasting-oriented representation learning with a *causal
  motivation*: first learn disentangled feature representations, then fit a simple
  regression on top. CoST uses a time-domain contrastive loss to learn discriminative
  *trend* representations and a frequency-domain contrastive loss to learn
  discriminative *seasonal* representations. (Verified from the arXiv abstract.)
- **What the embeddings capture (per the paper):** Explicitly disentangled
  seasonal vs. trend factors — the paper reports +21.3% MSE improvement on
  multivariate forecasting benchmarks, robustness to backbone encoder and downstream
  regressor choice. CoST is the clearest example of a method that *decides a priori*
  which factors of variation (season/trend) the embedding should separate — a useful
  contrast for probing what LeJEPA embeddings separate on their own.
