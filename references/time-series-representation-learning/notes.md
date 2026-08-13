# Time Series Representation Learning / SSL — Survey Notes

Literature survey for analyzing what information is contained in embeddings learned by
LeJEPA-style joint-embedding predictive pretraining on time series.

- Compiled: 2026-08-07
- Method: arXiv abstract pages, arXiv HTML full text (TS-JEPA), and dblp records were
  fetched and read; venues/authors cross-checked against dblp.
- Claims below are limited to what was verified in those sources; anything inferred
  beyond the sources is flagged as such.

Note: this survey has been flattened — each method now has its own folder under
`references/` (`t-loss`, `cpc`, `tnc`, `ts-tcc`, `ts2vec`, `cost`, `tst`, `simmtm`,
`ts-jepa`), each holding that paper's notes and `paper.pdf`. Only the cross-cutting
sections (evaluation protocols and overall relevance) remain here.

---

## Common evaluation protocols

Verified across the papers above (UCR/UEA usage from TS2Vec abstract and TS-JEPA full
text; linear-probe protocol from TS-TCC, TS2Vec, TS-JEPA, CHARM abstracts/full text):

- **UCR archive classification** — 128 univariate datasets (125 used by TS2Vec);
  standard train/test splits, accuracy; critical-difference diagrams for multi-dataset
  comparison. Reference: Dau et al., "The UCR Time Series Archive," IEEE/CAA J.
  Automatica Sinica 2019 (cited in TS-JEPA).
- **UEA archive classification** — 30 multivariate datasets (29 used by TS2Vec).
- **Frozen-encoder linear probe** — the dominant protocol: pretrain encoder, freeze,
  train a linear classifier (or ridge regression for forecasting) on the embeddings.
  Used by TS2Vec, TS-TCC, T-Loss, TS-JEPA, CHARM. End-to-end fine-tuning is the
  alternative (SimMTM, TST).
- **Forecasting transfer** — standard long-horizon benchmarks (ETT, Weather,
  Electricity); either linear regression on frozen representations (TS2Vec, CoST) or
  fine-tuning (SimMTM); short-term next-patch vs. long-term autoregressive rollout
  (TS-JEPA).
- **Auxiliary protocols:** few-label regimes (fraction of labels for fine-tuning;
  TS-TCC, TS-JEPA, TST), transfer across related datasets (FordA→FordB in TS-JEPA;
  TS-TCC transfer experiments), unsupervised anomaly detection (TS2Vec; CHARM),
  clustering metrics for state discovery (TNC).
- **Notably absent:** systematic *probing* of what signal properties (frequency
  content, trend, phase, noise level, channel identity) are decodable from the
  embeddings. The closest precedents are CoST's designed season/trend disentanglement
  and SimMTM's manifold-neighborhood framing — the literature evaluates *task
  performance*, not *information content*.

---

## Relevance

This survey matters for the LeJEPA probing project in four ways:

1. **Baseline objectives define the hypothesis space.** Contrastive methods (TS2Vec,
   TS-TCC, TNC, T-Loss, CoST) are explicitly trained to make embeddings invariant to
   augmentations/neighborhoods and discriminative of instance/context; CPC-style and
   JEPA-style predictive methods instead keep what is *predictive of the future in
   latent space*. Probing LeJEPA embeddings should therefore be framed against these
   alternatives: does latent-space prediction retain the same information
   contrastive methods retain (class identity, season/trend), and what does it drop
   that masked reconstruction (TST, MAE-style) would keep (raw signal detail, noise)?
2. **TS-JEPA is the closest published relative.** It is the same architectural family
   (patch tokenizer, transformer encoder, predictor, EMA target, L1 latent loss, no
   negatives) and shows the family already achieves a classification/forecasting
   balance with frozen-encoder probes — but it does not analyze *what* the embeddings
   contain. Our information-content probing is a direct, unfilled gap.
3. **Protocols to reuse.** The frozen-encoder linear probe on UCR/UEA classification
   and ETT/Weather/Electricity forecasting is the shared evaluation language of this
   literature; results expressed in it are comparable to TS2Vec, TS-TCC, CoST,
   TS-JEPA, CHARM. Few-label and transfer (FordA→FordB-style) settings are the
   established robustness checks.
4. **Known failure modes to probe for.** JEPA collapse dynamics (the OpenReview
   collapse-phase study), SimMTM's warning that point-level masking destroys temporal
   semantics, and TNC's debiasing argument (sampling bias from unlabeled temporal
   neighborhoods) all predict specific ways embeddings can look good on the pretext
   loss while carrying little task-relevant information — each is a concrete,
   testable probing target.
