# T-Loss — Unsupervised Scalable Representation Learning for Multivariate Time Series

- **Citation:** Jean-Yves Franceschi, Aymeric Dieuleveut, Martin Jaggi. NeurIPS 2019,
  pp. 4652–4663. arXiv:1901.10738.
- **URLs:** <https://arxiv.org/abs/1901.10738> ·
  [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2019/hash/53c6de78244e9f528eb3e1cda69699bb-Abstract.html) ·
  [dblp](https://dblp.org/rec/conf/nips/FranceschiDJ19)
- **Key idea:** Combine an encoder based on causal dilated convolutions with a novel
  triplet loss that uses *time-based negative sampling*. A reference subseries and a
  random subseries of the same series form a positive pair; negatives are sampled from
  other series (with sampling weight tied to time). The method handles variable-length,
  multivariate series and is scalable in series length. (Verified from the arXiv
  abstract.)
- **What the embeddings capture (per the paper):** The abstract claims the learned
  representations are general-purpose, transferable, and practical — demonstrated via
  "thorough experiments" on quality, transferability, and scalability. The abstract
  itself does not detail which signal properties (trend, frequency, phase) are
  retained; see the PDF for the full experimental analysis.
