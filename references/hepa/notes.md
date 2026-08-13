# HEPA: A Self-Supervised Horizon-Conditioned Event Predictive Architecture for Time Series

## Citation

- **Authors:** Jonas Petersen (ETH Zurich / Forgis), Gian-Alessandro Lombardi (Forgis), Riccardo Maggioni (Forgis), Camilla Mazzoleni (Forgis), Federico Martelli (ETH Zurich / Forgis), Philipp Petersen (University of Vienna). Correspondence: jep79@cantab.ac.uk
- **Title:** HEPA: A Self-Supervised Horizon-Conditioned Event Predictive Architecture for Time Series
- **Year/Venue:** 2026, arXiv preprint (v4, 3 Jun 2026). GitHub README search snippet mentions "Spotlight at FMSD @ ICML 2026" (workshop); the fetched README itself shows an anonymized citation ("under double-blind review"), while the arXiv v4 HTML lists the named authors above.
- **arXiv:** 2605.11130 — https://arxiv.org/abs/2605.11130 (HTML: https://arxiv.org/html/2605.11130v4)
- **Code:** https://github.com/Forgis-Labs/HEPA (PyTorch >= 2.0, CC BY-NC-SA 4.0)
- **Local PDF:** `paper.pdf` (arXiv v4, 21 pages)

## Problem setting

Unified formulation of event prediction in multivariate time series: given observations
up to time t, estimate P(event within Δt) for every prediction horizon Δt. This one
surface p(t, Δt) covers remaining-useful-life (RUL) prognostics, anomaly prediction,
arrhythmia detection, water contamination, cyberattack detection, volatility regimes,
etc. — tasks that are normally benchmarked separately with incomparable metrics
(RMSE for RUL, point-adjusted F1 for anomaly detection). Motivation: events are rare
and costly to label, so the method must work from unlabeled streams plus a handful of
event labels. ([abstract](https://arxiv.org/abs/2605.11130), [HTML §1](https://arxiv.org/html/2605.11130v4))

## Architecture

Three components, one fixed 2.16M-parameter configuration used unchanged across all
14 benchmarks (only the input projection changes with sensor count)
([HTML §3.1](https://arxiv.org/html/2605.11130v4)):

- **Context encoder f_θ:** causal Transformer, d=256, 2 layers, 4 heads. Input tokenised
  into non-overlapping patches of size P=16 (following PatchTST), per-context instance
  normalisation, sinusoidal positional encodings. Maps x_{≤t} to a summary embedding
  h_t ∈ R^256.
- **Horizon-conditioned predictor g_φ:** 2-layer MLP taking (h_t, Δt) and outputting a
  predicted embedding of the future interval: ĥ_{(t,t+Δt]} = g_φ(h_t, Δt). During
  pretraining Δt is sampled log-uniformly over [1, Δt_max], forcing the encoder to
  internalise dynamics at multiple timescales — this is the "horizon conditioning".
- **Target encoder:** the *same* f_θ (weight-shared, not an EMA copy), applied
  *bidirectionally* with attention pooling to the future window x_{(t,t+Δt]}, producing
  the target representation h*_{(t,t+Δt]}.

## Pretraining objective (Stage 1)

Self-supervised JEPA on unlabeled data:

    L = (1-α) ‖ĥ - h*‖_1 + α · L_SIG,   α = 0.1

- L1 (not L2) prediction loss — chosen because L1 distributes gradient magnitude
  equally across samples, avoiding domination by outlier predictions.
- **L_SIG is SIGReg** (Sketched Isotropic Gaussian Regularisation) from LeJEPA
  (Balestriero & LeCun, arXiv:2511.08544): constrains predictor outputs toward an
  isotropic Gaussian, which Balestriero & LeCun prove is the optimal embedding
  distribution for minimising downstream prediction risk. This replaces the EMA
  momentum schedule of I-JEPA and prevents collapse without heuristics.
- Because target and online encoders share weights, no stop-gradient is needed;
  collapse is prevented jointly by SIGReg and by input asymmetry (predictor never sees
  the future window). The paper explicitly positions HEPA "closer to LeJEPA / SIGReg
  variants than to the original I-JEPA recipe".
- Cost: under one minute per dataset on a single A10G; the full 14-dataset × 5-seed
  sweep takes under two hours.

## Downstream recipe (Stage 2): predictor finetuning + survival head

Departure from the standard JEPA recipe (which discards the predictor and linear-probes
the frozen encoder): HEPA **freezes the encoder and finetunes the predictor** together
with a lightweight linear event head ("pred-FT", 198K tuned params vs 2.16M end-to-end
vs 513 for a linear probe). ([HTML §3.2](https://arxiv.org/html/2605.11130v4))

- Predictor is run at K discrete unit-step horizons (K=150 for C-MAPSS/TEP, K=200 else).
- Shared linear head maps each predicted representation to a per-interval conditional
  hazard: λ_Δt(t) = σ(wᵀ ĥ_{(t,t+Δt]} + b) ∈ (0,1).
- Hazards compose into a discrete-time survival CDF: p(t,Δt) = 1 − Π_{j=1..Δt}(1−λ_j(t)).
  Monotonicity in Δt holds by construction; no distributional assumptions.
- Finetuning loss: positive-weighted BCE summed over horizons on the *cumulative*
  probability (not the standard per-step discrete-survival likelihood) — a deliberate
  choice acting as cross-horizon smoothing; it improves h-AUROC under the
  positive-weighted regime but distorts the probability scale (their appendix O).
- Rationale: more expressive than a linear probe (the predictor reshapes its
  horizon-conditioned outputs to align with the event), while the frozen encoder
  supplies the pretrained dynamics that make few labels sufficient.

## Theory: event-information retention bound

**Proposition 1 (Event-Information Retention).** Under (A1) target sufficiency
(E_{t+Δt} ⊥ X_{≤t} | H*), (A2) bounded prediction error E‖Ĥ−H*‖² ≤ ε, (A3) L-Lipschitz
event posterior, (A4) posterior bounded away from 0/1:

    I(H_t; E_{t+Δt}) ≥ I(H*; E_{t+Δt}) − C_η L² ε,   C_η = (2 η_underline (1−η_bar))^{-1}

Proof: data-processing inequality on the Markov chain E → H_t → Ĥ, plus a Jensen-gap
argument on the convex KL divergence. ([HTML §3.3, App. A](https://arxiv.org/html/2605.11130v4))

- Falsifiable prediction: within a dataset, lower pretraining loss ε ⇒ higher
  downstream h-AUROC. Validated by snapshotting encoders at epochs {1,3,8,25}+best and
  running the full pred-FT recipe: pooled Spearman ρ(ε, h-AUROC) = −0.67 (p=0.017,
  C-MAPSS-3), −0.64 (p=0.026, MBA), −0.49 (p=0.13, SMAP); C-MAPSS-1 gives ρ=−0.87,
  p<0.001 (appendix). Cross-dataset pooling does *not* show the trend (r=−0.05),
  because L, C_η and representation scale differ per dataset — the authors are explicit
  that the bound is only testable within a dataset.
- **Corollary 2 (Precursor necessity):** the bound is non-vacuous iff the future window
  contains event precursors the target encoder captures (I(H*;E) > 0) and
  ε < I(H*;E)/(C_η L²). This explains both successes (C-MAPSS: degradation unfolds over
  hundreds of cycles) and failures (short-window anomaly sets like GECCO, where
  within-dataset ρ=+0.14, p=0.67, with finetuning instability — reported as expected).
- Caveats stated by the authors: L and C_η are never estimated (only the monotone
  relationship is validated, not the quantitative bound); A1 failure makes the bound
  loose in a favourable direction.

## Probing / analysis methodology

- **PCA representation analysis (§5.3, Fig. 3b):** PCA of pretrained C-MAPSS-1 encoder
  representations for four test engines. With no labels, the encoder organises
  representations into a smooth degradation manifold: PC1 alone captures 61% of
  variance and tracks time-to-failure monotonically within each engine (median
  per-engine Spearman ρ=+0.97; 84% of engines have ρ>0.9). Engines starting from
  different healthy regions converge toward a shared failure region. This is direct
  evidence that JEPA pretraining linearises event-relevant structure.
- **Pretraining-loss ↔ downstream-performance probing (§3.3, Fig. 3a):** encoder
  snapshots at pretraining epochs {1,3,8,25}+converged-best, each followed by the
  standard pred-FT recipe and test h-AUROC (3 seeds) — a checkpoint-probing protocol
  linking representation quality to downstream event information. Notes mild
  over-pretraining: the converged-best snapshot regresses slightly vs epoch 25 on all
  three datasets tested.
- **Evaluation metric — h-AUROC (§4):** mean of per-horizon AUROC values over the
  p(t,Δt) surface. Motivated by prevalence varying ~200× across horizons on one surface
  (C-MAPSS-1: 0.5% at Δt=1 vs 96% at Δt=150), which makes pooled AUPRC nearly
  meaningless (0.957 prevalence baseline). Per-horizon binary problems each have a
  universal 0.5 baseline. Domain metrics (RUL RMSE, PA-F1, F1) are derived as lossy
  projections of the same surface for comparability with published baselines; the paper
  flags PA-F1 as inflated (citing Kim et al. 2022, Schmidl et al. 2022).

## Benchmarks and baselines

**Datasets — 14 benchmarks, 11 domains** (GitHub README + §5.1): C-MAPSS FD001–FD004
(turbofan degradation, 14 ch), SMAP (spacecraft telemetry, 25 ch), PSM (server metrics,
25 ch), MBA (cardiac ECG arrhythmia, 2 ch), GECCO (drinking-water quality, 9 ch),
BATADAL (water-distribution cyberattacks, 43 ch), TEP (chemical-plant faults, 52 ch),
ETTm1 (electricity transformer, 7 ch), Weather (21 ch), BeijingAQ (air quality, 11 ch),
VIX (financial volatility, 6 ch). Each dataset evaluated at 100% and 10% labels.

**Baselines:** architectural comparisons with matched-capacity downstream heads on
frozen encoders — PatchTST (supervised), iTransformer, MAE (masked reconstruction);
foundation model Chronos-2 (different regime: large-corpus pretraining); appendix
comparisons vs MOMENT, TFM-2.5, Moirai, and MTS-JEPA (HEPA wins 8/9 datasets where
MTS-JEPA could be reproduced). Domain SOTAs referenced: STAR (RUL RMSE 10.61 on
C-MAPSS), Anomaly Transformer, DCdetector, TranAD (anomaly detection).

## Key results

- Wins 10/14 benchmarks at full labels (including all four C-MAPSS variants and FD004,
  the hardest subset); wins 6/14 at 10% labels. Beats PatchTST, iTransformer, MAE, and
  Chronos-2 on at least 10 of 14 while tuning ~11× fewer parameters than PatchTST.
- Label efficiency: on C-MAPSS-1 retains 92% of full-label h-AUROC at 2% labels (2 of
  85 training engines); C-MAPSS-3 retains 97% at 10% labels.
- Honest losses reported: BATADAL and MBA (sensor-localised events — per-variate
  attention / channel-independent models win; iTransformer reaches 0.84 vs HEPA 0.75
  on MBA), SMAP and ETTm1 (MAE's reconstruction transfers well under gradual drift).
- vs MTS-JEPA: wins 8 of 9 comparable datasets.
- GitHub note: repo C-MAPSS h-AUROC values are higher than the paper's because the repo
  uses fixed-epoch finetuning instead of validation-loss early stopping (deterministic;
  all other datasets unchanged).

## Relevance

HEPA is the closest published prior work to this project, in both objective and method:

- **Same pretraining family.** HEPA is explicitly a LeJEPA/SIGReg-style JEPA for time
  series: it predicts future *representations* (not values), uses SIGReg (isotropic
  Gaussian regularisation, the LeJEPA collapse-prevention mechanism) instead of
  EMA/stop-gradient heuristics, and adds a horizon-conditioning input Δt to the
  predictor. Any claims about what LeJEPA-style time-series embeddings contain should
  be checked against HEPA's design choices (weight-shared target encoder, L1 latent
  loss, log-uniform horizon sampling, patch size 16, tiny 2-layer/256-d encoder).
- **Direct evidence about embedding content.** Its PCA analysis shows the frozen
  encoder's representations form a smooth, label-free degradation manifold whose first
  principal component tracks time-to-failure nearly monotonically (ρ≈0.97 per engine).
  This is exactly the kind of "what information is in the embedding" result our probing
  study aims to generalise — and HEPA only probes PC1 vs. time-to-failure on one
  dataset, leaving open systematic probing of what else (noise, channel identity,
  horizon-invariant vs. horizon-specific structure) the embeddings encode.
- **A theoretical frame we can reuse.** Proposition 1 lower-bounds the mutual
  information between the encoder embedding and a downstream event in terms of the
  pretraining prediction error ε — a formal link between JEPA loss and retained
  task-relevant information, with an honest treatment of when it is vacuous
  (Corollary 2: no precursors in the future window ⇒ no guarantee). Its
  checkpoint-probing protocol (ε vs. downstream performance across pretraining epochs)
  is a ready-made methodology for our analysis.
- **Methodological contrast points.** HEPA argues the JEPA latent space "retains what
  is predictable and discards what is not", and its failure cases (sensor-localised
  events diluted by channel-fusion tokenisation; MAE winning under gradual drift) map
  out where representation-level prediction loses information that value-level
  objectives keep — precisely the hypotheses a probing study should test. Its
  pred-FT-vs-linear-probe comparison also quantifies how much event information is
  linearly accessible in the frozen encoder versus recoverable only through the
  predictor.
- **Practical reuse.** Code is public (PyTorch, CC BY-NC-SA 4.0), training is cheap
  (<1 min pretraining per dataset on one A10G), and the 14-dataset benchmark suite with
  the h-AUROC evaluation framework is a candidate testbed for probing experiments.

## Verification notes

- All technical claims above were verified against the arXiv v4 HTML
  (https://arxiv.org/html/2605.11130v4) and abstract page
  (https://arxiv.org/abs/2605.11130); dataset channel counts and the reproducibility
  note come from the GitHub README (https://github.com/Forgis-Labs/HEPA).
- Discrepancy: the fetched GitHub README displays an anonymized citation ("Anonymous
  authors, under double-blind review"), while the arXiv v4 page names the authors and a
  search snippet of the same repo shows a named citation plus "Spotlight at FMSD @
  ICML 2026". The named author list above follows arXiv v4; the workshop-spotlight
  claim was seen only in a search snippet, not verified on the fetched page.
- Numeric results for individual datasets beyond those quoted here live in Table 1 of
  the PDF (`paper.pdf`); the HTML extraction truncated parts of the appendices, so
  appendix-level details (G/H baseline numbers, I.3 SIGReg configuration, J metric
  projections, O calibration) were not fully re-verified line by line.
