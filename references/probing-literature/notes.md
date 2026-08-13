# Probing / Representation-Analysis Literature Survey

Scope: methods for answering "what information is contained in learned embeddings", with
methodological lessons for designing probing experiments on LeJEPA-style joint-embedding
predictive pretraining on time series.

Note: this survey has been flattened — each paper now has its own folder under
`references/` (`linear-probes-alain-bengio`, `structural-probes`, `mdl-probing`,
`pareto-probing`, `control-tasks-selectivity`, `amnesic-probing`, `rankme`,
`dimensional-collapse`), each holding that paper's notes and, where available,
`paper.pdf`. Only the cross-cutting Relevance section remains here.

Downloaded PDFs (verified as complete papers, page counts via pypdf) were moved to
their per-paper folders: Hewitt & Liang 2019 (11 pages) → `control-tasks-selectivity`,
Voita & Titov 2020 (14 pages) → `mdl-probing`, Elazar et al. 2021 (15 pages) →
`amnesic-probing`. These three were chosen as most central to probing-experiment
design (controls/selectivity, information-theoretic measurement, causal/behavioral
interpretation).

---

## Relevance

Why this literature matters for analyzing LeJEPA-style time-series embeddings:

- **The core question decomposes.** "What information is contained in the embedding?" splits
  into: is property X *linearly accessible* (Alain & Bengio; structural probes), *how much*
  of it is there in bits (MDL/Pareto probing), and is it *causally used* by the predictive
  objective (amnesic probing). A complete analysis of LeJEPA embeddings should report all
  three; probe accuracy alone supports only the weakest claim.
- **JE-SSL-specific diagnostics come first.** LeJEPA is a joint-embedding method without
  reconstruction, so RankMe's effective rank and Jing et al.'s singular-spectrum analysis
  are label-free, domain-agnostic quality gates that directly detect the failure modes
  (complete and dimensional collapse) the JEPA family is designed to avoid — check these
  before interpreting any probe result.
- **Controls are mandatory.** Hewitt & Liang's control tasks and Elazar et al.'s
  random-direction INLP controls show that without baselines, high-capacity probes and
  naive interventions both produce inflated or confounded conclusions. Any probing suite
  for time-series embeddings needs: random-label control tasks (per-window identity),
  majority-class baselines, random-direction removal controls, and a
  randomly-initialized-encoder comparison (Voita & Titov's motivation for MDL).
- **Task difficulty and probe complexity are design axes, not afterthoughts.** Pareto
  probing shows trivial tasks (the time-series analogues of POS tagging: mean/slope/linear
  trend) cannot discriminate representations; include hard tasks (forecasting, long-range
  phase/lag structure — the analogue of full dependency parsing and of the structural
  probe's distance hypothesis) and report the accuracy–complexity frontier.
- **Encoded ≠ used.** The central negative result across this literature (Hewitt & Liang;
  Voita & Titov; Elazar et al.) is that decoding success does not imply the pretrained
  objective put the information there for use. For LeJEPA, the sharpest question — "does
  the predictor actually exploit property X when forecasting?" — requires intervention
  (INLP-style amnesic probing on time-series properties), not just classification.

Caveats: the Hewitt & Manning structural probe has no arXiv version (ACL Anthology N19-1419
used as canonical source); the Pimentel et al. ACL 2020 mutual-information paper was
verified only via citations in other papers' reference lists, not fetched directly.
