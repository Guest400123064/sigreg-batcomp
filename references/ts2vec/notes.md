# TS2Vec

- **Citation:** Zhihan Yue, Yujing Wang, Juanyong Duan, Tianmeng Yang, Congrui Huang,
  Yunhai Tong, Bixiong Xu. AAAI 2022, pp. 8980–8987. arXiv:2106.10466.
- **URLs:** <https://arxiv.org/abs/2106.10466> ·
  <https://doi.org/10.1609/aaai.v36i8.20881> ·
  [dblp](https://dblp.org/rec/conf/aaai/YueWDYHTX22)
- **Key idea:** Contrastive learning performed *hierarchically over augmented context
  views* (random cropping), producing a robust contextual representation for **each
  timestamp**. Both instance-wise and temporal contrastive losses are applied at
  multiple semantic scales; representations of arbitrary sub-sequences are obtained by
  simple (max-)pooling aggregation over timestamp embeddings. (Hierarchical/multi-scale
  structure verified from the abstract; the instance-wise + temporal loss decomposition
  is from the paper, see PDF.)
- **What the embeddings capture (per the paper):** Multi-scale contextual information:
  SOTA unsupervised results on 125 UCR + 29 UEA classification datasets; a linear
  regression on frozen representations beat prior SOTA in forecasting; SOTA
  unsupervised anomaly detection via a simple score on the learned representations.
  TS2Vec is the de-facto frozen-encoder linear-probe baseline — its strong linear-readout
  results imply the embeddings linearly encode class identity, future values, and
  anomaly-relevant structure.
