# TNC — Temporal Neighborhood Coding

- **Citation:** Sana Tonekaboni, Danny Eytan, Anna Goldenberg. ICLR 2021.
  arXiv:2106.00750; OpenReview `8qDwejCuCN`.
- **URLs:** <https://arxiv.org/abs/2106.00750> ·
  [OpenReview](https://openreview.net/forum?id=8qDwejCuCN) ·
  [dblp](https://dblp.org/rec/conf/iclr/TonekaboniEG21)
- **Key idea:** Exploit the *local smoothness* of a signal's generative process to
  define temporal neighborhoods with (approximately) stationary properties. A
  *debiased contrastive objective* learns to make the distribution of signals from
  within a neighborhood distinguishable from the distribution of non-neighboring
  signals in encoding space. (Verified from the arXiv abstract.) The "debiased" part
  corrects for the fact that randomly sampled "negatives" may in fact belong to the
  same underlying state — a sampling-bias problem specific to time series, where true
  class labels are unavailable.
- **What the embeddings capture (per the paper):** Motivated by medicine: the goal is
  to encode a patient's underlying *latent state* so that windows in the same state
  cluster together. The paper demonstrates superior clustering and classification
  performance vs. other unsupervised approaches on multiple datasets. So TNC
  embeddings are explicitly designed to capture *state identity over local windows*,
  not fine-grained within-window dynamics.
