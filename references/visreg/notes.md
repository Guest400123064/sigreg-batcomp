# VISReg: Variance-Invariance-Sketching Regularization for JEPA training

## Citation

- **Authors:** Haiyu Wu, Randall Balestriero, Morgan Levine
- **Title:** VISReg: Variance-Invariance-Sketching Regularization for JEPA training
- **Venue:** arXiv preprint, June 2026 (the arXiv HTML header tags "ICML"; not yet a confirmed published venue — flag as unverified)
- **arXiv:** [2606.02572](https://arxiv.org/abs/2606.02572) [cs.CV], v1 submitted 2026-06-01; DOI 10.48550/arXiv.2606.02572
- **Full text:** [arXiv HTML](https://arxiv.org/html/2606.02572v1)
- **Code:** [github.com/HaiyuWu/visreg](https://github.com/HaiyuWu/visreg) (CC BY-NC 4.0; pretrained ViT-B/16 and ViT-L/14 ImageNet-1K weights on HuggingFace `BooBooWu/visreg`)
- **Project page:** https://haiyuwu.github.io/visreg
- Author affiliations (from press coverage, not the paper itself — flagged): Wu now at Altos Labs (PhD Notre Dame), Balestriero at Brown (first author of LeJEPA/SIGReg), Levine at Altos Labs.

## Summary

VISReg is a heuristic-free self-supervised JEPA regularizer that merges the VICReg
loss decomposition with LeJEPA/SIGReg's distributional-sketching principle.

Lineage: **VICReg → SIGReg (LeJEPA) → VISReg** (a phrase Yann LeCun used when
endorsing the paper on social media, per press coverage).

Motivation — complementary flaws of the two parents:

- **VICReg** regularizes only *second-order* statistics (variance + covariance).
  A distribution can match mean/covariance yet be far from Gaussian, so covariance
  is a weak proxy for the isotropy that stable training needs. Also costs O(ND²).
- **SIGReg** sketches the embedding distribution toward an isotropic Gaussian via
  the Epps–Pulley characteristic-function test (grounded in the Cramér–Wold
  theorem), giving full distributional control at O(D). But (1) it does not
  decouple scale from shape, and (2) **its gradient vanishes as the embedding
  collapses** — the Epps–Pulley test statistic's gradient diminishes with feature
  norm, disappearing exactly when corrective signal is most needed (paper Fig. 2:
  simulated gradient magnitude vs. collapse stage; Barlow Twins and VISReg retain
  strong gradients under collapse, SIGReg does not).

VISReg fix: keep VICReg's variance term for *scale*, replace the covariance term
with a Sliced-Wasserstein (SWD) sketching objective for *shape*, applied to
stop-gradient-normalized embeddings. Decoupling scale and shape gives robust
anti-collapse gradients, full distributional rigor, and per-term reweighting
flexibility, at linear complexity.

## Methodology

### Loss (all equations from [arXiv HTML](https://arxiv.org/html/2606.02572v1))

Given centered embeddings `Ẑ ∈ R^{N×D}` (N = batch, D = projection dim):

- **Scale (variance), Eq. 1:** `L_scale = (1/D) Σ_j (1 − σ_j(Ẑ))²` — VICReg-style
  hinge-free variance term; its gradient approaches a constant during collapse.
- **Normalization with stop-gradient, Eq. 2:** `Z̃ = Ẑ / (sg(σ) + ε)` — isolates
  shape from magnitude; the stop-gradient here is principled objective
  decomposition, not a collapse heuristic.
- **Shape (Sliced Wasserstein), Eq. 5:** project onto K random unit directions
  `w_k`, sort each 1D projection, and match Gaussian quantiles:

  `L_shape = (1/K) Σ_k ‖sort(Z̃ w_k) − q_N‖²₂`

  where `q_N` are the fixed standard-Gaussian quantiles (icdf of `i/(N+1)`).
  Justified by Cramér–Wold (Lemma 3.1: matching all 1D projections ⟺ matching
  distributions) and the 1D Wasserstein closed form (Lemma 3.2: W_p^p = L_p
  distance between quantile functions). Unlike SIGReg's frequency-domain
  Epps–Pulley test, this is an optimal-transport-domain sketch.
- **Center, Eq. 6:** `L_center = ‖μ‖²₂` on the batch mean (empirical robustness).
- **Regularization total, Eq. 7:** weighted sum of the three terms; default
  λ's = 1, but **increasing the shape weight improves low-quality-dataset
  performance** — a benefit of decoupling impossible under monolithic sketching.
- **Invariance/prediction, Eq. 8:** follows LeJEPA — MSE between each view's
  embedding and the mean of global-view embeddings (multi-crop: 2–4 global +
  6 local views). Full loss: `(1−λ)L_pred + λL_reg` (Eq. 9).
- Algorithm 1 is ~20 lines of PyTorch: center, scale, randn projections,
  sort, Gaussian icdf target, done.

### Complexity and scaling

- Cost: O(NDK) projection + O(KN log N) sorting ⇒ effectively **O(NDK), linear in
  all factors**, vs. VICReg's O(ND²); SWD avoids SIGReg's 17-knot Epps–Pulley
  sampling overhead (13.7% memory of SIGReg at batch 50K in simulation).
- Empirical subtlety: K must exceed D by a factor C>1 for best accuracy, which
  nominally makes it O(NCD²) — but since slices are drawn independently per GPU,
  **distributing slices across M GPUs keeps per-GPU K constant** (verified: 8 GPUs
  at K=128/GPU match single-GPU K=1024 accuracy within ~0.2–0.3%).
- VISReg is the most robust of the Cramér–Wold family (SIGReg, plain SWD, VISReg)
  to very few slices (works even at K = D/8); unlike SIGReg it is naturally
  batch-size invariant (no λ rescaling with batch size).

### Results

Setup: ViT-B/16 and ViT-L/14, pretrained on ImageNet-1K (400 epochs, LeJEPA
augmentations), linear-probe/transfer/segmentation protocols follow DINOv2.

- **In-domain linear probe** (Table 5): 75.7% (ViT-B/16), 77.0% (ViT-L/14) on
  ImageNet-1K — best among heuristic-free methods (beats MAE and LeJEPA), but a
  gap remains to heuristic methods (DINO 78.2%, iBOT ViT-L 81.0%).
- **OOD linear probe** (Table 6): best average across 6 OOD datasets (DTD,
  Galaxy10, AID, ChestXRay, RetinaMNIST, OrganAMNIST) — ViT-L/14 70.63 avg vs.
  iBOT's 69.64, despite no heuristics. ViT-B/16 beats other methods' ViT-L/H.
- **Data efficiency:** ViT-L/14 pretrained on **ImageNet-22K matches DINOv2
  (LVD-142M, 10× more data)** on OOD average (72.94 vs. 72.93).
- **Transfer (fine-tuning, Table 7):** beats both supervised DeiT and DINO on
  CIFAR-10/100, Flowers, ImageNet-1K, Galaxy10 — even though DINO has >3% higher
  in-domain linear probe.
- **Dense prediction (Table 8):** ADE20K linear segmentation mIoU 30.16 — on par
  with DINO (29.40), behind MoCoV3 (31.69) and iBOT.
- **Generation guidance (Table 9):** as an iREPA-style feature target for SiT-B/2,
  improves all metrics (gFID 40.36 vs. DINO 41.15).
- **Low-quality data:** from-scratch on ImageNet-LT (long-tailed) VISReg tops all
  shot regimes; on Galaxy10 (low-rank) 80.76 vs. DINO 73.49. DINO largely fails
  to learn meaningful features in these regimes.

## Relevance

VISReg is the most direct successor to LeJEPA/SIGReg and a strong candidate
regularizer for LeJEPA-style time-series pretraining; it matters for probing in
several concrete ways:

- **Different imposed geometry than SIGReg.** SIGReg forces the *joint* embedding
  distribution (including scale) toward an isotropic Gaussian in one monolithic
  term. VISReg forces unit variance per dimension separately, then matches only
  the *shape* of the normalized distribution to a Gaussian along K random 1D
  projections. Two consequences for probes: (1) per-dimension variances are pinned
  near 1 by construction, so variance-based diagnostics (e.g. rank/collapse
  metrics, "how much variance does dimension i carry") measure regularizer
  compliance, not learned information content; (2) marginals along *random*
  projections are Gaussian, but the distribution is constrained only up to K
  slices plus a centering term — fine-scale or multi-modal structure along
  un-sampled directions, and any structure carried in the stop-gradient scale
  pathway, is free to encode information. Probing should therefore look at
  geometry (e.g. sliced-Wasserstein residuals, kurtosis along task-relevant
  directions), not just covariance spectra.
- **Cleaner baseline for "what does the embedding contain" studies.** Because
  VISReg is heuristic-free (no EMA teacher, no stop-gradient asymmetry in the
  prediction path), any information found in its embeddings is attributable to the
  data + objective, not to architectural stabilization tricks — same argument
  LeJEPA makes vs. DINO, now with better OOD/transfer results behind it.
- **Vanishing-gradient analysis is directly transferable to time series.** The
  Fig. 2 collapse-gradient experiment (gradient norm vs. feature norm) is a cheap
  diagnostic worth replicating when choosing SIGReg vs. VISReg-style losses for
  time-series JEPA, especially since time-series datasets are often low-rank /
  long-tailed — exactly the regime where VISReg's reweightable shape term
  outperforms.
- **Hyperparameter guidance ports over.** The findings that K ≥ C·D slices are
  needed, that slices can be distributed across GPUs, that smaller projection
  dimensions help dense/local prediction but bottleneck in-domain linear probe,
  and that shape-loss weight should rise on low-quality data, all apply when
  sizing a probe-ready time-series encoder.
- **Caveat for probing comparisons:** VISReg's reported OOD/transfer advantage is
  on vision benchmarks with ViT backbones; nothing in the paper addresses
  time-series modalities. Claims about representation generality should be treated
  as vision-domain evidence only.

### Unverified / flagged

- ICML venue tag appears in the arXiv HTML header but the arXiv listing is a
  preprint; acceptance not independently confirmed.
- Author affiliations and the LeCun endorsement come from press articles
  ([aitntnews](https://www.aitntnews.com/newDetail.html?newId=27222),
  [36kr](https://m.36kr.com/p/3915058026894727)), not the paper.
- Exact per-dataset in-domain numbers were taken from the GitHub README results
  tables (rendered more completely than the arXiv HTML); they match the paper's
  qualitative claims.
