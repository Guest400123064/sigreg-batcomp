# TST — A Transformer-based Framework for Multivariate Time Series Representation Learning

- **Citation:** George Zerveas, Srideepika Jayaraman, Dhaval Patel, Anuradha
  Bhamidipaty, Carsten Eickhoff. KDD 2021, pp. 2114–2124. arXiv:2010.02803.
- **URLs:** <https://arxiv.org/abs/2010.02803> ·
  <https://doi.org/10.1145/3447548.3467401> ·
  [dblp](https://dblp.org/rec/conf/kdd/ZerveasJPBE21)
- **Key idea:** First transformer-based unsupervised framework for multivariate time
  series. Pretraining is a masked-reconstruction (denoising) task: a fraction of
  input timestamps is masked and the model regresses the masked values from the
  unmasked context. Pretrained models serve downstream regression, classification,
  forecasting, and missing-value imputation. (Masked-input regression objective is
  from the paper; the abstract verifies the framework and downstream-task claims.)
- **What the embeddings capture (per the paper):** The paper reports that unsupervised
  pretraining beats supervised SOTA on multivariate regression/classification
  benchmarks even with very few labels, and that pretraining helps even when only
  reusing the *same* labeled data through the unsupervised objective — i.e., the
  reconstruction task itself extracts transferable structure.
