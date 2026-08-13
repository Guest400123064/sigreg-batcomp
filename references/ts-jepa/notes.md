# TS-JEPA — Joint Embeddings Go Temporal

- **Citation:** Sofiane Ennadir (KTH), Siavash Golkar (NYU), Leopoldo Sarra
  (Flatiron Institute). arXiv:2509.25449, Sep 2025.
- **URL:** <https://arxiv.org/abs/2509.25449> (full text read at
  <https://arxiv.org/html/2509.25449v1>)
- **Key idea:** First systematic study of JEPA for time series. Four components
  (verified from full text):
  1. **Tokenizer:** non-overlapping patches embedded by a 1D-CNN, plus absolute
     sin-cos positional embeddings; uniform patch masking (70% mask ratio used).
  2. **Encoder** (E_theta): standard transformer, processes only non-masked patches.
  3. **Predictor** (P_beta): transformer mapping encoded visible tokens to
     predictions of masked-token representations.
  4. **EMA-Encoder** (E_thetabar): exponential-moving-average copy of the encoder
     (momentum m=0.998) that encodes the masked patches to produce targets;
     motivated (following BYOL/I-JEPA) as collapse prevention.
- **Loss:** mean L1 distance between predicted and EMA-target embeddings of masked
  patches: L = (1/|M|) * sum_{i in M} || P_beta(E_theta(P_N))_i - E_thetabar(p_i) ||_1.
  No contrastive negatives, no variance/covariance regularizer — collapse is avoided
  purely via the EMA target.
- **Experiments (verified from full text):** Frozen-encoder protocol with small
  trainable heads. Classification: FordA/FordB, FaultDetectionA/B, ECG5000, including
  transfer (train on FordA → test FordB). Forecasting: Weather, ETT-Small, Electricity;
  short-term (next patch) and long-term (autoregressive rollout). Baselines: TS2Vec,
  MAE-style masked reconstruction in input space, autoregressive prediction, random
  frozen encoder, and fully supervised CNN/transformer — all with the *same* encoder
  architecture for fairness. Findings: TS-JEPA ≈ MAE and > TS2Vec/autoregressive on
  classification, close to fully supervised; autoregressive wins short-term
  forecasting, TS-JEPA wins long-term on 2/3 datasets; TS-JEPA is much more
  label-efficient than supervised training at 5–20% labels. Authors position it as a
  building block for time-series foundation models.
- **Relevant observations for probing:** The paper motivates latent-space prediction
  precisely by *not having to model input noise/confounders* — the central claim that a
  probe study can test (what is actually retained/discarded?). It also cites two
  earlier TS-JEPA-flavored works: Girgis et al., "Time-series JEPA for predictive
  remote control under capacity-limited networks" (arXiv:2406.04853, applied to
  encoded frames) and LaT-PFN, "a joint embedding predictive architecture for
  in-context time-series forecasting" (arXiv:2405.10093, JEPA + in-context prediction).

---

## Related: HEPA — Horizon-Conditioned Event Predictive Architecture (reference point)

- **Citation:** Jonas Petersen et al. "HEPA: A Self-Supervised Horizon-Conditioned
  Event Predictive Architecture for Time Series." arXiv:2605.11130, 2026.
- **URL:** <https://arxiv.org/abs/2605.11130> (details from search-result excerpts of
  the arXiv HTML; the abstract page itself was not fetched — treat architecture
  details here as *partially verified*).
- **Key idea (per excerpts):** A causal transformer encoder pretrained with a JEPA
  objective in which a *horizon-conditioned predictor* g_phi(h_t, Δt)
  forecasts future representations for arbitrary horizons Δt; a second stage
  freezes the encoder and composes horizon-specific hazard rates into a survival CDF
  for event prediction. HEPA reportedly differs from BYOL/I-JEPA/V-JEPA in that the
  target encoder is a weight-shared copy (rather than an EMA copy). This is the
  project baseline architecture; included here for orientation. (See also the
  dedicated `references/hepa/` folder.)

## Related: Other recent JEPA-style time-series models (beyond HEPA)

- **CHARM / Multimodal JEPA for Semantic Time-Series Embeddings** — Gerardo Pastrana
  et al., arXiv:2605.31580, May 2026. <https://arxiv.org/abs/2605.31580>. Channel-aware
  transformer encoder (equivariant to channel order) that incorporates channel-level
  *textual descriptions*, trained with JEPA plus a novel loss promoting "informative,
  temporally stable embeddings." Evaluated with *linear probes only* on anomaly
  detection, classification, and short/long-term forecasting; ablations attribute the
  gain primarily to the JEPA objective and conditioning architecture. (Verified from
  arXiv abstract.)
- **CGM-JEPA** — Hada Melino Muhammad et al., arXiv:2605.00933, May 2026.
  <https://arxiv.org/abs/2605.00933>. Domain application to continuous glucose
  monitoring: predicts masked *latent* representations instead of raw values;
  X-CGM-JEPA adds a cross-view objective against a "Glucodensity" distributional
  summary. Pretrained on ~389k unlabeled CGM readings; evaluated on small clinical
  cohorts for insulin-resistance and β-cell-dysfunction classification with cross
  validation; claims improved transfer across modality shifts (venous OGTT ↔ CGM) and
  reduced subgroup AUROC gaps. (Verified from arXiv abstract.)
- **LaT-PFN** (arXiv:2405.10093) and **Girgis et al.** (arXiv:2406.04853) — cited via
  the TS-JEPA reference list; not independently fetched.
- Related analysis worth noting: an OpenReview paper (id `SdOYmP67a2`, seen in search
  results) characterizes a **collapse phase in JEPA training** — a transient regime
  where prediction loss drops while representations remain uninformative — traced to
  tight encoder/EMA-target coupling, with experiments on ImageNet *and time-series
  data*, and proposes an encoder–EMA discrepancy metric as a diagnostic. (Seen only in
  a search-result excerpt; fetch before relying on it.)
  <https://openreview.net/pdf?id=SdOYmP67a2>
