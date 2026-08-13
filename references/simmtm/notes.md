# SimMTM — A Simple Pre-Training Framework for Masked Time-Series Modeling

- **Citation:** Jiaxiang Dong, Haixu Wu, Haoran Zhang, Li Zhang, Jianmin Wang,
  Mingsheng Long. NeurIPS 2023. arXiv:2302.00861.
- **URLs:** <https://arxiv.org/abs/2302.00861> ·
  [dblp (NeurIPS 2023 listing)](https://dblp.org/search/publ/api?q=SimMTM+masked+time-series+modeling&format=json)
- **Key idea:** Argues that random point-wise masking *destroys the temporal
  variations that carry the semantics of time series*, making naive reconstruction too
  hard to guide representation learning. SimMTM relates masked modeling to manifold
  learning: masked points are recovered by the *weighted aggregation of multiple
  neighbor series outside the manifold*, re-assembling complementary (individually
  ruined) temporal variations; an additional constraint uncovers the local structure
  of the series manifold. (Verified from the arXiv abstract.)
- **What the embeddings capture (per the paper):** SOTA fine-tuning performance vs.
  other TS pretraining methods on forecasting and classification, both in-domain and
  cross-domain. Conceptually important: it claims embeddings should encode the
  *manifold/neighborhood structure* of series, not raw point values — a useful
  hypothesis for probing (do embeddings encode neighbors-in-data-space?).
