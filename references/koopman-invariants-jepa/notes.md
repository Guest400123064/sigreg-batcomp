# Koopman Invariants as Drivers of Emergent Time-Series Clustering in Joint-Embedding Predictive Architectures

## Citation

- **Authors:** Pablo Ruiz-Morales, Dries Vanoost, Davy Pissoort, Mathias Verbeke (KU Leuven)
- **Venue:** AAAI 2026 (Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40)
- **Year:** 2025 (arXiv, Nov 2025) / 2026 (AAAI)
- **arXiv:** [2511.09783](https://arxiv.org/abs/2511.09783) — [HTML full text (v2)](https://arxiv.org/html/2511.09783v2)
- **DOI:** 10.1609/aaai.v40i30.39708 — [AAAI publisher page](https://ojs.aaai.org/index.php/AAAI/article/view/39708/43669)
- Funded by EU Horizon Europe grant no. 101168880 (dn-isense.eu project)
- No official code repository was found linked from the paper (checked arXiv page; not verified further)

## Summary

The paper asks: why do JEPA models trained on time series spontaneously cluster latent
embeddings by underlying (unlabeled) dynamical regime, when e.g. an autoencoder with the
same encoder architecture does not? The answer proposed is a first-principles theoretical
one: JEPA's latent-prediction loss implicitly drives the encoder to learn the
**invariant subspace of the system's Koopman operator**, i.e. regime indicator functions,
which are eigenfunctions with eigenvalue 1.

### Setup / loss

- Input windows `x_t = (s_t, ..., s_{t+n-1})` from a stationary process with invariant measure μ.
- Standard JEPA: online encoder `f_θ`, predictor `g_ϕ`, EMA target encoder (α = 0.996).
- Loss: `E_x[ ||g_ϕ(f_θ(x_t)) − f_EMA(x_{t+Δ})||² ]`, idealized by assuming `f_EMA ≈ f_θ`.
- **Assumption 3.1 (finite mixture of ergodic regimes):** μ = Σ αᵢ μᵢ over r ergodic
  components with essentially disjoint, dynamically immiscible supports 𝒳ᵢ.
- **Assumption 3.3:** predictor is linear, `g(z) = Mz`.

### Theory

- The Δ-step Koopman operator `(𝒦ψ)(x) = E[ψ(x_{t+Δ}) | x_t = x]` is a linear contraction
  on L²(𝒳, μ). Eigenfunctions with eigenvalue 1 are invariants of the dynamics.
- **Lemma 3.2:** regime indicator functions χᵢ = 1_{𝒳ᵢ} satisfy 𝒦χᵢ = χᵢ and are
  pathwise invariant (χᵢ(x_{t+Δ}) = χᵢ(x_t)); span{χ₁..χᵣ} =: 𝒱 is exactly the
  eigenvalue-1 eigenspace, of dimension r.
- The loss decomposes (Eq. 5) into **Term 1** (mean prediction error,
  `E||M ψ⃗ − 𝒦ψ⃗||²`) and **Term 2** (inherent stochasticity,
  `E||𝒦ψ⃗(x) − ψ⃗(x_{t+Δ})||²`). Regime-indicator representations drive *both* to zero.
- **Theorem 3.4 (regime indicator theorem):** with k ≥ r and a linear predictor, the
  idealized JEPA loss is globally minimized iff encoder components fⱼ satisfy
  `(𝒦fⱼ)(x) = fⱼ(x_{t+Δ})` a.e. and M acts as (𝒦f) on f(𝒳). This is achieved when
  fⱼ ∈ 𝒱 and **M acts as the identity on the learned regime subspace**.
- Clustering corollary: if f(x) = A·χ⃗(x) with rank-r A, all windows from regime 𝒳ᵢ map
  to the same latent point A·eᵢ → r distinct latent clusters aligned with regimes.
- **Disentanglement caveat (key):** the JEPA loss is invariant to any invertible linear
  change of basis of the latent space. Regime information lives in an r-dimensional
  *subspace*, but only an inductive bias toward an identity-like predictor (e.g.
  identity initialization of M) selects the interpretable canonical basis rather than an
  entangled one.

## Methodology (experimental validation)

- **Synthetic dataset, r = 18 regimes** (Assumption 3.1 holds by construction), 10,000
  sequences per regime, master length 1024 steps → 180,000 context–target pairs;
  70/20/10 train/val/test split; per-sequence standardization; no additive observation
  noise on deterministic signals.
- Regime types: sinusoids (varied frequency/amplitude/harmonics, randomized phase),
  square waves, sawtooth, AR models with varying coefficients, MA, ARMA, linear trends
  (± slope), sparse pulse sequences, sinusoid + trend, high-process-noise sinusoid.
- **Windowing:** context = first n_c = 768 steps; target = window shifted by horizon
  Δ = 256 (substantial overlap). Non-overlapping schemes (Δ ≥ n_c) reportedly also work.
- **Model (PyTorch):** encoder = 4-layer 1-D CNN (ReLU) + linear projection head,
  latent dim k = 32 (≥ r). Predictor variants:
  - linear `Mz`, no bias, **initialized as identity** (main analysis),
  - linear with random init (control),
  - non-linear MLP (2 hidden layers, ReLU, width 2k) for "typical JEPA" comparison.
- **Baselines / clustering measurement:**
  - t-SNE of test embeddings, colored by ground-truth regime.
  - **K-Means (K=18) cluster purity** as the quantitative metric:
    JEPA (MLP predictor) 65.48% vs. identical-architecture autoencoder 38.81%.
- **Predictor-matrix probes (the paper's main diagnostic toolkit):** for the
  identity-initialized linear M:
  - relative Frobenius distance to identity `||M−I||_F/||M||_F` = **2.34%**;
  - skew-symmetry `||M−Mᵀ||_F/||M||_F` = **2.06%**;
  - **eigenvalue spectrum of M**: dominated by r eigenvalues near 1.0 (predictor
    preserves the r-dim regime subspace, attenuates other directions);
  - action on K-Means cluster centroids cᵢ: mean relative error
    `||Mcᵢ − cᵢ||/||cᵢ||` = **0.80%**.
- **Control result:** randomly initialized M reaches the same low loss but is a dense,
  non-identity (entangled) transform, while clustering still emerges visually —
  confirming the loss-level basis ambiguity of the optimal subspace.

## Key results

- JEPA's predictive objective (not encoder capacity) produces regime-aligned latent
  clusters; an identical reconstruction-based AE does not.
- The learned linear predictor empirically behaves as a near-identity operator on the
  regime subspace, matching Theorem 3.4: eigenvalues ≈ 1 on r directions, near-zero
  action error on regime centroids.
- The interpretable solution is one optimum among a class of equivalent
  linearly-transformed optima; the predictor's initialization/inductive bias selects it.
- **Limitations (authors' own):** synthetic data with perfectly immiscible regimes only;
  linear predictor and `f_EMA ≈ f_θ` idealizations; real-world data with gradual regime
  transitions or hierarchical structure untested.

## Relevance

This is arguably the most directly relevant theory paper for this project — it explains
*what* a JEPA-style time-series encoder should be expected to contain and why:

- **Predicts a specific probe target.** Theorem 3.4 says LeJEPA-style embeddings should
  linearly encode **regime identity** (indicator functions / the eigenvalue-1 Koopman
  subspace). Probes for regime labels, cluster purity, and subspace structure are not
  ad hoc here — they test the quantity the theory says the encoder *must* learn to
  minimize its loss. A linear probe recovering regime membership is the predicted
  signature, not just an observed correlation.
- **Invariance, not forecasting detail, may dominate.** The optimal solution throws away
  within-regime variation (eigenvalues beyond the top r are attenuated). This gives a
  concrete hypothesis for probing: within-regime signal details (phase, amplitude,
  short-term dynamics) may be *absent* or weakly encoded, while regime membership is
  sharply encoded. Probe batteries should therefore test both regime-level and
  fine-grained dynamical properties to map this predicted information asymmetry.
- **Basis ambiguity matters for probe design.** Because the loss is invariant under
  invertible linear transforms of the latent space, regime information is guaranteed at
  the *subspace* level, not per-dimension. Linear probes (which find any basis) should
  succeed; neuron-level or per-dimension interpretability analyses may misleadingly
  conclude "no clean encoding." The paper's own predictor-diagnostics (Frobenius-to-identity,
  eigenvalue spectrum of the predictor, action on cluster centroids) are directly
  reusable as analysis tools for our LeJEPA models.
- **Role of the predictor constraint.** Constraining the predictor to be near-identity
  (or analyzing how far a learned predictor is from identity) is identified as *the*
  inductive bias driving invariant learning — a design lever for LeJEPA (predictor
  capacity/init/regularization) and a measurable quantity during probing.
- **Scope caveats for our setting:** LeJEPA adds a SigReg-type regularizer and may use
  non-linear predictors and real (non-immiscible) dynamics — all outside the theorem's
  assumptions (linear predictor, exact regime immiscibility, `f_EMA ≈ f_θ`). The theory
  is best used as a hypothesis generator: check whether clustering/invariant-subspace
  signatures survive under gradual transitions, stochastic regimes, and regularized
  objectives.

### Unverified / open items

- AAAI DOI (10.1609/aaai.v40i30.39708) and publisher page existence taken from the
  assignment prompt + search results; not independently fetched page-by-page.
- No public code repository identified; experimental details (Appendices B–C with full
  generative parameters and architecture) are in the PDF but only partly visible in the
  fetched HTML truncation.
