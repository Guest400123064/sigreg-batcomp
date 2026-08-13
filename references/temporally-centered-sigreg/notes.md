# Temporally Centered SIGReg Improves Multi-Task LeWorldModel Learning: From Analysis to Method

## Citation

- **Authors:** Chang Liu, Fei Suo, Yanzhou Jin, Yusuke Iwasawa, Yutaka Matsuo, Yaonan Zhu (corresponding, yaonan.zhu@weblab.t.u-tokyo.ac.jp)
- **Affiliation:** Graduate School of Engineering, The University of Tokyo
- **Venue:** arXiv preprint (cs.LG, cs.RO); no peer-reviewed venue stated as of v2
- **Year:** 2026 (v1: 2026-07-29; v2: 2026-07-31)
- **arXiv ID:** 2607.26924v2, DOI https://doi.org/10.48550/arXiv.2607.26924
- **URLs:**
  - Abstract: https://arxiv.org/abs/2607.26924
  - Full text (HTML): https://arxiv.org/html/2607.26924v2
  - Project page: referenced in the paper but URL not extracted from the HTML text (unverified)

## Summary

LeWorldModel (LeWM, Maes et al. 2026) trains an encoder plus action-conditioned
latent predictor from pixels, using SIGReg (Sketched Isotropic Gaussian
Regularizer, from LeJEPA — Balestriero & LeCun 2025) to push the latent
**marginal** toward an isotropic Gaussian via the Epps–Pulley (EP) normality
statistic on random 1-D projections. This works single-task but degrades badly
under multi-task joint training.

### Failure-mode analysis (Sec. 3)

- On LIBERO-Long, Raw LeWM drops from 40.0% average behavior-cloning success
  under independent single-task training to 29.9% under 10-task joint training
  (8 of 10 tasks degrade), despite the latent remaining globally non-collapsed.
- Decomposition: each latent is split into a local temporal mean (low-frequency,
  task/context-dependent cluster structure) and a short-timescale residual:
  `z_t = z̄_t + r_t`, where `z̄_t` is the mean over a temporal window `W_t`.
- Linearity of projections means a SIGReg-projected scalar decomposes as
  `x = c + ε` (projected center + projected residual). They model `c` as a
  balanced homoscedastic K-component Gaussian mixture, `ε ~ N(0, σ_r²)`,
  giving `p(x) = (1/K) Σ N(x; μ_k, σ²)`.
- **Monte Carlo diagnostic** (M=500 randomized center configurations per K;
  K ∈ {2,5,10,20}; normalized center spread ρ = std_k[μ_k]/σ swept 0–4):
  - Excess EP loss increases with ρ → EP penalizes well-separated components.
  - Radial gradient signal is positive outside a small-ρ weak-gradient region →
    gradient descent **contracts** the component-center spread.
  - Fraction of configurations showing contractive pressure grows with ρ → the
    effect is systematic, not geometry-specific.
- Consequences: as ρ ≲ 1, center spread becomes comparable to within-component
  σ → cluster overlap → **representation aliasing** across tasks and states.
  A task-conditioned policy `π(a_t | z_t, i)` can use the task label to resolve
  task ambiguity but **cannot** recover within-task state distinctions already
  aliased in `z_t`; similar latents then map to incompatible actions.
- Measured on LIBERO-Long frozen encoders: Raw LeWM ρ = 0.74 (low-separation
  regime) vs TC-LeWM ρ = 2.54.

### Method: TC-LeWM / TC-SIGReg (Sec. 4)

- Keep LeWM encoder `z_t = f_θ(x_t)` and predictor `ẑ_{t+1} = g_ψ(z_t, a_t)`
  unchanged; change only the SIGReg target:
  - Raw: `L = L_pred + λ L_SIGReg(Z)` on the full latent marginal.
  - TC:  `L = L_pred + λ L_SIGReg(R)` on temporally centered residuals
    `r_t = z_t − z̄_t` (window W=8 frames ≈ 1.4 s on LIBERO; ablation in App. A.5).
- Why it works (two properties):
  - **No direct pressure on cluster separation:** under `x = c + ε`, scaling ρ
    changes `c` but not the residual distribution, so the residual EP objective
    is invariant to ρ (blue dash-dot curves in Fig. 2 stay ≈ 0).
  - **Anti-collapse retained:** temporal collapse implies `r_t ≡ 0`, i.e. the
    point mass δ₀, which EP strongly penalizes. Unstructured residual noise
    could satisfy the regularizer but is hard to predict from `(z_t, a_t)`, so
    prediction + residual regularization jointly favor non-degenerate,
    predictable temporal variation.

### Results (Sec. 5)

- Suite-wise 10-task training: average success 53.2% → 73.6% across the four
  LIBERO suites; largest gains on LIBERO-Object (+41.4 pts) and LIBERO-Long
  (+21.9 pts). Slightly exceeds from-scratch Diffusion Policy (72.4%) and
  approaches fine-tuned Octo (75.1%) / OpenVLA (76.5%) — authors note these are
  contextual references only (different pipelines).
- Negative transfer eliminated: on LIBERO-Long, joint vs single-task changes by
  +2.2 pts for TC-LeWM vs −10.1 pts for Raw LeWM; TC-LeWM also beats Raw LeWM
  by +9.6 pts even single-task (authors interpret this as the full-marginal
  Gaussian prior being overly restrictive even without cross-task interference).
- Unified 40-task training: TC-LeWM 73.5% (−0.1 pts vs 10-task) vs Raw LeWM
  44.4% (−8.8 pts); TC–Raw gap widens from 20.4 to 29.1 pts.

## Methodology relevant to representation analysis

- **Latent geometry metrics:**
  - Normalized cluster spread ρ = std_k[μ_k]/σ (episode-level cluster centers vs
    within-cluster std) — their key separability metric.
  - Residual-energy fraction (App. A.3): share of latent energy in the
    short-timescale residual. TC-LeWM: 2%; Raw LeWM: 24%; demonstrated robot
    joint positions at the same timescale: 4–5% — i.e. Raw LeWM's residual
    "jitter" is far above the physical signal's.
  - Latent-norm distributions: Raw LeWM concentrates norms on a thin shell
    (isotropic-Gaussian signature); TC-LeWM shows broader, multimodal norms.
- **Visualization:** global PCA of 1024-d concatenated two-view CLS latents
  colored by task (top-2 PCs explain 93% variance for TC-LeWM); per-task PCA
  colored by normalized episode progress (top-2 PCs: 90% avg for TC vs 31% for
  Raw). TC-LeWM shows contiguous task clusters on a curved manifold and
  progress-aligned intra-task trajectories; Raw LeWM shows task overlap and
  fragmented trajectories. Nearby TC-LeWM clusters correspond to visually
  similar tasks (App. A.3).
- **Perturbation robustness (Sec. 5.4, App. A.4):** eight state-preserving
  visual perturbation families (translation, rotation, zoom, blur, partial
  occlusion, brightness, contrast, plus stochastic noise). Metric: relative
  latent displacement normalized by the encoder's median distance between
  distinct clean observations (s=1 = typical clean-state separation). TC-LeWM
  has smaller displacement across all eight families; largest gains under
  geometric transforms; brightness/contrast show smaller differences with
  overlapping error bars. Interpretation: in Raw LeWM's compressed geometry,
  perturbation-induced displacement occupies a larger fraction of the available
  clean-state separation, pushing the closed-loop policy toward OOD conditioning.
- **EP statistic (App. A.1):** `T_EP(h) = B ∫ w(s) |φ̂_B(s) − e^{−s²/2}|² ds`
  with Gaussian weight w(s), J=1024 random unit projections, applied per
  temporal position and per camera view (views not concatenated). SIGReg is
  applied only to CLS tokens, not patch tokens.
- **Downstream evaluation:** frozen encoder + task-conditioned rectified-flow
  behavior-cloning policy (12-layer DiT, AdaLN-Zero, 34 visual tokens: 1 CLS +
  4×4 pooled patches per view, no proprioception, 8-step action chunks, 10
  Euler steps at inference). Checkpoint selection via validation
  inverse-dynamics-model (iDM) metric. Results: mean ± std over 3 training
  seeds × 3 eval seeds × 50 rollouts/task.

## Relevance

This paper is essentially a controlled study of what SIGReg (the LeJEPA
regularizer our time-series project also uses) does to the *information content*
of embeddings, which is exactly our probing question.

- **SIGReg actively destroys cluster-level (low-frequency) information.** The
  EP objective's gradient contracts the separation between component centers
  relative to within-component variance. In our probing terms: a marginal
  isotropic-Gaussian prior does not merely fail to preserve class/task identity
  — it exerts systematic pressure *against* class separability. Probes of
  LeJEPA-style embeddings should therefore expect attenuated cluster-level
  linear separability as a direct consequence of the regularizer, not as a
  failure of the encoder.
- **Global non-collapse ≠ preserved semantics.** Raw LeWM latents pass the
  marginal Gaussianity check (statistically non-collapsed) while aliasing
  task/state distinctions. This warns against using marginal statistics as
  evidence that information is retained; only downstream/probe-based measurement
  reveals the aliasing.
- **Within-cluster vs between-cluster trade-off is the right axis.** Their ρ
  metric (center-spread / within-cluster σ) is a cheap, transferable diagnostic
  we can apply to time-series embeddings to quantify how much the regularizer
  compresses the structure a probe could recover.
- **Temporal centering is an information-preserving surrogate.** Applying
  SIGReg to `r_t = z_t − z̄_t` keeps anti-collapse pressure (δ₀ point mass is
  penalized) while leaving slow task/context structure untouched. For time
  series this suggests: regularize the high-frequency/residual component, let
  slow structure (regime, trend, subject identity) organize freely. The
  prediction loss and residual regularizer are complementary — predictability
  filters out pure-noise residuals.
- **Perturbation sensitivity as an information-content probe.** Their
  normalized-latent-displacement metric ties representation compression to
  robustness: compressed geometry means small input perturbations move the
  embedding by a large fraction of inter-state distance. This is a concrete
  methodology we can reuse to test whether LeJEPA-pretrained time-series
  embeddings encode stable state distinctions.
- **Caveats:** analysis uses a balanced homoscedastic Gaussian-mixture model of
  projected latents (a simplification of real embedding geometry); the
  contractive-pressure result is Monte Carlo / empirical, not a closed-form
  theorem; results are demonstrated on LIBERO vision-based manipulation, so
  transfer to time-series domains is plausible but unverified.
