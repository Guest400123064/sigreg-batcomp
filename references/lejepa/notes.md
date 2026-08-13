# LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics

## Citation

- Authors: Randall Balestriero (Brown University, Meta-FAIR), Yann LeCun (NYU, Meta-FAIR) — equal contribution
- Title: LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics
- Venue: arXiv preprint (cs.LG), 2025. No peer-reviewed venue could be verified from the sources consulted.
- arXiv ID: 2511.08544 (v1 2025-11-11; v3 2025-11-14)
- URLs:
  - Abstract: https://arxiv.org/abs/2511.08544
  - PDF: https://arxiv.org/pdf/2511.08544 (saved as `paper.pdf` in this folder)
  - Official code: https://github.com/rbalestr-lab/lejepa (also `pip install lejepa`)
- Companion paper by overlapping authors: "Gaussian embeddings: How JEPAs secretly learn your data density" (arXiv:2510.05949), cited in the paper's references.

## Summary

LeJEPA (Latent-Euclidean JEPA) is a self-supervised joint-embedding predictive
training objective built from two axioms the authors derive from theory:

1. Solve the standard JEPA prediction task: embeddings of one view of a sample
   must be predictable from embeddings of other views.
2. The embedding distribution must be (near-)isotropic Gaussian — which they
   prove is the optimal embedding distribution for a "foundation model" that
   will face unknown downstream tasks.

The total loss is a convex combination of a prediction term and a new
distribution-matching regularizer, SIGReg, with a single trade-off
hyperparameter `lambda` (recommended default 0.05). Because SIGReg provably
prevents representational collapse, LeJEPA removes all the standard
anti-collapse heuristics: no stop-gradient, no teacher–student / EMA networks,
no predictor network, no negative samples, no whitening layers, no
hyperparameter schedulers, and no register tokens. The core implementation is
~50 lines of PyTorch with O(N) time and memory in the minibatch size.
([abstract](https://arxiv.org/abs/2511.08544); paper Secs. 1, 5)

### Why isotropic Gaussian is optimal (paper Sec. 3)

The paper asks: given a frozen encoder whose embeddings will be probed for
*arbitrary, unknown* downstream targets `y`, which embedding distribution
minimizes worst-case risk?

- Linear probing (ridge/OLS): comparing two embeddings with the same column
  span and same total variance, one anisotropic and one isotropic —
  - Lemma 1: whenever covariance eigenvalues differ, there always exists a
    downstream task for which the anisotropic embedding gives a higher-bias
    ridge estimator (for any regularization strength > 0).
  - Lemma 2: with no regularization, total variance of the OLS estimator is
    strictly lower for the isotropic embedding.
  Together these force isotropy of the covariance.
- Nonlinear probing (radius-based k-NN and kernel regression): Theorem 1 shows
  the integrated squared bias depends on a functional `J(p)` of the embedding
  density, and among all distributions under a scalar covariance constraint
  (fixed trace or Frobenius norm of the covariance), the isotropic Gaussian is
  the *unique* minimizer of the bias.
- Hence the design principle: embeddings `f_theta(x)` should follow an
  isotropic Gaussian to minimize worst-case risk over post-training tasks.

### SIGReg: Sketched Isotropic Gaussian Regularization (paper Sec. 4)

SIGReg enforces the isotropic-Gaussian constraint by turning distribution
matching into hypothesis testing:

- Null hypothesis `H0: P_theta = Q` (embedding distribution equals the target).
  Instead of estimating a high-dimensional divergence (quadratic complexity in
  samples), SIGReg decomposes the test into univariate tests along random unit
  directions `a`: `H0(a): a^T P_theta = a^T Q`.
- Lemma 3 (Hyperspherical Cramér–Wold): two random vectors are equal in
  distribution iff all their 1-D projections are equal in distribution —
  so matching enough univariate projections matches the full distribution.
- Theorem 2: aggregating directional test statistics over a direction set A
  (via max, union–intersection principle) is a valid, consistent test of the
  multivariate null (correct level, power → 1). In the loss, the max is
  replaced by an average over directions to avoid sparse gradients:
  `SIGReg_T(A, {f_theta(x_n)}) = (1/|A|) sum_a T({a^T f_theta(x_n)})`.
- Choice of univariate test `T` (Sec. 4.2): the paper surveys moment-based
  (Jarque–Bera), CDF-based (Cramér–von Mises, Anderson–Darling, Watson), and
  characteristic-function (CF) tests, and argues for the Epps–Pulley test:
  - Theorem 3: matching any finite set of K moments does not identify the
    distribution — moment matching leaves collapse shortcuts; and high-order
    moments have exploding gradient norms (O(k)) and MC variance, so
    moment-based tests cannot be both identifiable and stable.
  - CDF-based tests need sorting/order statistics — non-differentiable and
    hostile to multi-GPU SGD.
  - Epps–Pulley compares the empirical characteristic function (ECF)
    `phi_hat(t) = (1/n) sum_j e^{i t X_j}` to the target CF in weighted L2,
    `EP = N ∫ |phi_hat(t) - phi(t)|^2 w(t) dt`, with Gaussian weight
    `w(t) = e^{-t^2/sigma^2}`. The ECF is a smooth average of complex
    exponentials: differentiable, bounded, and computed with a simple
    `all_reduce` average across GPUs.
  - Theorem 4: Epps–Pulley has uniformly bounded gradients (<= 4 sigma^2 / N
    per sample) and bounded curvature regardless of the input distribution —
    this is what makes heuristics-free training stable.
- Beating the curse of dimensionality (Sec. 4.3):
  - Theorem 5: if the embedding density has Sobolev smoothness alpha and the
    test is satisfied on |A| directions, the expected discrepancy over a random
    unseen direction decays as `|A|^{-2 alpha / (K-1)}`; smooth embeddings mean
    |A| = O(K) directions suffice.
  - Resampling fresh random directions every SGD step compounds coverage:
    even |A| as low as 16 per step outperforms a fixed set of thousands.
- Minibatch bias: Theorem 6 shows the minibatch Epps–Pulley estimate (loss and
  gradient) is biased by an explicit O(1/N) term; the authors report it is
  negligible even at batch size 16.
- Practical settings (Algorithm 1): 17 trapezoidal quadrature knots over
  t in [-5, 5] (integrand symmetry doubles knots for free), ~256–1024 slices
  per step, directions sampled from a seeded generator synchronized across GPUs.
- Relation to prior work (Sec. 5.2): exact evaluation of the Epps–Pulley
  integral per slice recovers a kernel MMD (quadratic complexity); using a
  degenerate mean/variance-matching test `T` makes SIGReg recover VICReg in
  the many-slices limit — which the authors argue against (Theorem 3 shortcuts).

### LeJEPA objective and training recipe (paper Sec. 5; repo README)

- Views: DINO-style multi-crop — `V_g = 2` global views (224x224, crop scale
  0.3–1.0) and `V_l = 6–8` local views (98x98 / 96x96, crop scale 0.05–0.3),
  with standard color jitter, grayscale, blur, solarize. For non-ViT
  architectures, global views = all views.
- Prediction loss: mean squared error between each view's embedding and the
  mean of the global-view embeddings `mu_n = (1/V_g) sum z_{n,v}` — i.e. an
  L2 "predict the global centroid" loss (Eq. 5–7). No predictor network.
- Total loss: `L = lambda * mean_v SIGReg({z_{n,v}}) + (1 - lambda) * L_pred`
  with recommended `lambda = 0.05`.
- Recipe: AdamW, lr ~5e-4 (searched in {5e-3, 5e-4}), weight decay 5e-2 for
  ViT / 5e-4 for ResNet (searched in {1e-1, 1e-2, 1e-5}), bf16 mixed precision,
  linear warmup + cosine decay to lr/1000, batch size >= 128 works on
  ImageNet-1K. Optional SWA on the global-view encoder gives a small boost for
  ViTs but is not needed against collapse.
  ([repo README](https://github.com/rbalestr-lab/lejepa))
- Notably, the framework does not assume image augmentations: the dataset is
  treated abstractly as an (N, V, D) tensor where the V views may be ordered
  by time (frames) rather than augmentation — the stated hook for temporal /
  sequential data (paper Sec. 2.1).

### Empirical results (paper Sec. 6; repo README)

- Headline: ImageNet-1K pretraining, frozen backbone, linear probe — 79%
  top-1 with ViT-H/14 (abstract). Online linear probe 77.1% (ViT-L, 0.3B) and
  78.5% (ConvNeXtV2-H, 0.6B) (Sec. 6.4).
- Stability: ~50 timm architectures from 8 families (<20M params) pretrained
  on ImageNet-10 all reach 91.5–95% top-1 frozen linear probe with the same
  recipe; performance is flat across lambda, view counts, slice counts,
  quadrature settings, register tokens, and projector sizes (Fig. 8, Table 1).
- Training loss as a label-free model-selection signal: Spearman correlation
  ~85% between the LeJEPA training loss and downstream linear accuracy; after
  rescaling the loss by `lambda^alpha` with alpha ≈ 0.4, correlation reaches
  up to ~99% across models and datasets (Figs. 10–11). Claimed as "the first
  practical loss for model selection without supervised probing".
- In-domain pretraining: on Galaxy10 (11k samples) and even flowers102
  (1000 samples), small LeJEPA models pretrained in-domain beat DINOv2/DINOv3
  transfer (both frozen linear probe and full fine-tuning) across 1-shot to
  full-data regimes (Fig. 12, Table 3).
- Few-shot transfer: LeJEPA ViT-L (100 IN-1K epochs) matches or beats I-JEPA
  ViT-H (300 epochs) on average over 8 datasets (DTD, aircraft, cars, CIFAR,
  flowers102, food, pets) at 1-shot, 10-shot, and full-data linear probe —
  roughly 3x less pretraining compute (Table 2).
- Scaling: stable heuristic-free training demonstrated up to a 1.8B-parameter
  ViT-g (Fig. 1).
- Emergent semantics: PCA of last-layer features separates foreground from
  background; thresholding [CLS] self-attention yields unsupervised object
  segmentation with temporal consistency on video (Figs. 13–14).
- Ablations (Table 4, per Sec. 6.1 text): removing the predictor and the
  teacher–student pair causes collapse in prior JEPA recipes but not in
  LeJEPA; register tokens are unnecessary.

## Analysis / probing methodology in the paper

This paper is itself partly about probing — its theory is built around
downstream probes, which is directly relevant to our probing project:

- Linear probing formalized as ridge regression on the embedding matrix
  `Z in R^{N x K}`; the bias (Lemma 1) and variance (Lemma 2) of the probe
  estimator are derived as functions of the embedding covariance eigenspectrum.
  Takeaway for probing: anisotropic embeddings inflate probe bias *and*
  variance for some tasks, so probe results on non-isotropic embeddings are
  task-dependent in a way isotropic-Gaussian embeddings are not.
- Nonlinear probes analyzed: radius-based k-NN and Nadaraya–Watson kernel
  regression, with integrated squared bias over query points as the risk
  measure (Theorem 1).
- Evaluation protocol used empirically: frozen backbone + linear probe
  (features = concat of [CLS] tokens from the last two layers, LayerNorm on
  the concatenation, AdamW, weight decay 1e-6 — per repo README), few-shot
  linear probes (1/10/all shots), and full fine-tuning; embedding-quality
  visualizations via PCA of patch features and [CLS] attention-map
  thresholding.
- Label-free model selection: training loss (rescaled by `lambda^0.4`) used
  as a proxy for downstream accuracy, validated with Spearman correlation
  across hyperparameter sweeps.

## Relevance

LeJEPA is the direct methodological basis of our project (LeJEPA-style JEPA
pretraining on time series), and this paper defines both the objective we
build on and several constraints on how we can interpret our probes:

- The (N, V, D) formalism explicitly allows V to be time-ordered views, which
  is exactly the time-series instantiation: windows/segments of a series as
  views, L2 prediction toward the global-view centroid, plus SIGReg on each
  view's embedding batch. Our time-series LeJEPA variant inherits the same
  single-hyperparameter (lambda ≈ 0.05) recipe.
- Strong prior on embedding geometry: SIGReg forces embeddings toward an
  isotropic Gaussian. Any information-content analysis of LeJEPA embeddings
  must account for the fact that the marginal distribution is heavily
  regularized — e.g. covariance spectrum, intrinsic dimensionality, and
  Gaussianity tests are diagnostics of SIGReg convergence more than of learned
  content. Linear-probe results are the "native" evaluation the theory
  optimizes for (isotropic Gaussian minimizes worst-case linear-probe risk).
- Theorem 1's framing — embeddings should minimize worst-case risk over
  *unknown* downstream tasks — gives a principled expectation for probing:
  LeJEPA embeddings should retain broadly linearly-decodable information;
  information that is only nonlinearly decodable would be surprising and
  informative about what the prediction task discarded.
- The training-loss/downstream-accuracy correlation (Spearman up to ~99% after
  lambda^0.4 rescaling) offers a label-free way to select time-series
  pretraining checkpoints before probing.
- The claimed in-domain pretraining advantage (small domain datasets beating
  frontier-model transfer) motivates pretraining directly on time-series data
  rather than transferring vision or text foundation models.
- Open questions for our project that this paper does not answer: it is
  vision-only empirically; nothing is measured about *which* semantic
  variables (e.g. frequency, trend, phase, amplitude for signals) survive the
  isotropic-Gaussian bottleneck — that is precisely what our probing study
  must measure.

## Source files in this folder

- `paper.pdf` — arXiv v3 PDF (50 pages, verified as valid PDF 1.7)
- `paper.txt` — text extracted from the PDF (via pypdf) used for these notes
- `notes.md` — this file

All technical claims above were verified against the arXiv abstract page, the
paper PDF text, and the official GitHub README (URLs cited inline). The paper
states no peer-reviewed venue; if one is later published, update the citation.
