# References Index

Literature collected for the project analyzing what information is contained in embeddings
from LeJEPA-style joint-embedding predictive pretraining on time series data.

Each folder holds one paper/topic: `notes.md` (citation, technical summary, methodology,
relevance to this project) and, where available, `paper.pdf`.

## Core frameworks

- `lejepa/` — Balestriero & LeCun, *LeJEPA: Provable and Scalable SSL Without the Heuristics* (arXiv 2511.08544). L2 latent prediction + SIGReg (isotropic-Gaussian regularizer via the Epps–Pulley test); provably optimal embedding distribution for downstream probes; no EMA/stop-gradient/predictor.
- `lewm/` — Maes et al., *LeWorldModel* (arXiv 2603.19312). Small JEPA world model trained end-to-end from pixels with SIGReg; includes a ready-made embedding-analysis battery (linear/MLP probes, decoding, violation-of-expectation).
- `v-jepa/` — V-JEPA (TMLR 2024) and V-JEPA 2, plus I-JEPA lineage. Masked feature prediction from video; attentive-probe evaluation; evidence that feature prediction encodes predictable dynamics over static detail.
- `hepa/` — Petersen et al., *HEPA* (arXiv 2605.11130). Closest prior work: LeJEPA-style JEPA for time series with a horizon-conditioned predictor; theory linking pretraining loss to retained event information.
- `what-should-embeddings-embed/` — Zhang et al. (TMLR). Direct inspiration: autoregressive embeddings represent latent generating distributions (predictive sufficient statistics / belief states); synthetic DGPs + weak linear probes + strong controls.
- `visreg/` — Wu, Balestriero et al. (arXiv 2606.02572). Variance (scale) + Sliced-Wasserstein sketching (shape) regularization for JEPA; fixes SIGReg's vanishing-gradient-under-collapse problem; "VICReg begat SIGReg begat VISReg".
- `temporally-centered-sigreg/` — Liu et al. (arXiv 2607.26924). Analysis + fix: SIGReg's marginal Gaussianization actively contracts cluster separation (aliasing, perturbation sensitivity) in multi-task LeWM; applying SIGReg to temporally centered residuals restores it. Reusable diagnostics: separability metric ρ, residual-energy fraction, perturbation robustness.

## Probing & representation analysis

- `linear-probes-alain-bengio/` — Alain & Bengio 2016. The original linear-classifier probe for intermediate layers.
- `structural-probes/` — Hewitt & Manning 2019. Probing for syntax tree structure in NLP embeddings.
- `mdl-probing/` — Voita & Titov 2020. Minimum-description-length probing: information content in bits, not just accuracy.
- `pareto-probing/` — Pimentel et al. 2020. Critique of MDL probing; Pareto frontier of probe complexity vs. accuracy.
- `control-tasks-selectivity/` — Hewitt & Liang 2019. Control tasks and selectivity: probes must beat memorization baselines.
- `amnesic-probing/` — Elazar et al. 2021. Removing information (INLP) to test causal usage, not just decodability.
- `rankme/` — Garrido et al. 2023. Label-free SSL representation quality via effective rank.
- `dimensional-collapse/` — Jing et al. 2022. Embedding-covariance spectrum as a collapse diagnostic.
- `probing-literature/` — slim leftover: cross-cutting methodological lessons spanning the probing papers above.

## Time series representation learning

- `t-loss/` — Franceschi et al. 2019. Triplet-loss unsupervised time-series embeddings.
- `cpc/` — van den Oord et al. 2018. Contrastive predictive coding; autoregressive latents that predict the future.
- `tnc/` — Tonekaboni et al. 2021. Temporal neighborhood coding: neighborhood vs. non-neighborhood discrimination.
- `ts-tcc/` — Eldele et al. 2021. Temporal + contextual contrasting for time series.
- `ts2vec/` — Yue et al. 2022. Hierarchical contrastive learning over timestamp- and instance-wise views.
- `cost/` — Woo et al. 2022. Contrastive learning with designed season/trend disentanglement.
- `tst/` — Zerveas et al. 2021. Transformer with masked reconstruction pretraining for time series.
- `simmtm/` — Dong et al. 2023. Masked modeling via manifold neighborhood aggregation.
- `ts-jepa/` — Ennadir et al. 2025. JEPA for time series (EMA target encoder, latent patch prediction); also notes on related minor TS-JEPA variants (CHARM, CGM-JEPA, LaT-PFN).
- `koopman-invariants-jepa/` — Ruiz-Morales et al. (AAAI 2026, arXiv 2511.09783). Theory: JEPA's predictive objective learns Koopman-invariant regime indicator functions, explaining emergent clustering of time series by dynamical regime; predicts regime identity is sharply encoded while within-regime detail is attenuated.
- `time-series-representation-learning/` — slim leftover: shared evaluation protocols (UCR/UEA, forecasting transfer) and relevance notes.
