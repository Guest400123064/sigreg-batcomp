# What Should Embeddings Embed? Autoregressive Models Represent Latent Generating Distributions

## Citation

- **Authors:** Liyi Zhang (Princeton CS), Michael Y. Li (Stanford CS), R. Thomas McCoy (Yale Linguistics / Wu Tsai Institute), Theodore R. Sumers (Anthropic), Jian-Qiao Zhu (Princeton CS), Thomas L. Griffiths (Princeton Psychology & CS)
- **Venue:** Transactions on Machine Learning Research (TMLR). Listed on the TMLR accepted-papers page ([jmlr.org/tmlr/papers](https://jmlr.org/tmlr/papers/)); OpenReview search results show "Accepted by TMLR", published 15 Jul 2025. Note: openreview.net itself was bot-blocked during this research, so the OpenReview page content could not be fetched directly; acceptance status was verified via the JMLR/TMLR listing and OpenReview search snippets.
- **IDs:** arXiv:2406.03707 (v1: 6 Jun 2024, v2: 7 Jan 2026); OpenReview forum id `YyMACp98Kz`
- **URLs:**
  - Abstract: https://arxiv.org/abs/2406.03707
  - Full text (v2 HTML): https://arxiv.org/html/2406.03707v2
  - PDF: https://arxiv.org/pdf/2406.03707 (local copy: `paper.pdf`, v2, 28 pages)
  - OpenReview: https://openreview.net/forum?id=YyMACp98Kz
  - Code: https://github.com/zhang-liyi/llm-embeddings
- **Version note:** v1 (Jun 2024) had three authors (Zhang, Li, Griffiths) and framed "three settings" (i.i.d. data, latent state models, discrete hypothesis spaces). v2 (the TMLR version, used here) adds McCoy, Sumers, Zhu, restructures into two major cases (exchangeable models, latent state models) and adds the LDA / HMM-LDA / natural-corpora experiments. This notes file follows v2.

## Core theory

Setup: an autoregressive LM predicts `p(x_{n+1} | x_{1:n})`; write the embedding as `φ_n = f(x_{1:n})` and the predictive distribution as `g(φ_n)` (the final layer). Question: what must `φ_n` represent?

Key concept — **predictive sufficient statistic**: a statistic `s(x_{1:n})` such that `p(x_{n+1} | x_{1:n}) = p(x_{n+1} | s(x_{1:n}))`. The paper's central observation: *a perfect autoregressive model learns a predictive sufficient statistic*, because the next token only depends on the embedding. This lets them analytically identify the optimal embedding content in cases where predictive sufficient statistics are interpretable ([v2 HTML, §3](https://arxiv.org/html/2406.03707v2)):

- **Case 1 — Exchangeable models.** By de Finetti, exchangeable sequences are i.i.d. mixtures over a latent parameter `θ`. Proposition (§3.1): for exchangeable data, if `g ∘ f` is a perfect autoregressive predictor, then `f(x_{1:n})` is a (general) sufficient statistic for the sequence, and hence fully specifies the posterior `p(θ | x_{1:n})`. So embeddings of exchangeable data should encode the sufficient statistics / posterior over the generating parameters.
  - **Case 1.1 — Discrete hypothesis spaces.** `p(x_{n+1}|x_{1:n}) = Σ_h p(x_{n+1}|h) p(h|x_{1:n})`, so the embedding need only encode the posterior over hypotheses `p(h | x_{1:n})`.
  - **Case 1.2 — Topic models (LDA).** The topic mixture `θ` (and word distributions `β`) are the sufficient statistics, so autoregressive LMs on text should implicitly encode document topic structure.
- **Case 2 — Latent state models (HMM, Kalman filter).** `p(x_{n+1}|x_{1:n}) = ∫ p(x_{n+1}|z_{n+1}) p(z_{n+1}|x_{1:n}) dz_{n+1}`, so the belief state over the *next* latent state `p(z_{n+1}|x_{1:n})` (or the current one `p(z_n|x_{1:n})`) is a predictive sufficient statistic. Which of the two is encoded depends on which downstream integral is easier to approximate in the output layer.
- **Appendix A.1 — Masked LMs.** A parallel decomposition shows a masked LM learns the posterior `p(θ | x_U)` over the unmasked set `U` — a strictly less expressive objective than the per-position autoregressive one, predicting that MLM embeddings encode latent structure less well (confirmed empirically with BERT in harder regimes).

## Experimental methodology (this is the part most relevant to our project)

General probe setup (§3.3, A.3): train a small transformer on synthetic sequences from a *known* generative process; take the **last-layer, last-token embedding**; train a deliberately weak probe — a **linear layer (with softmax for distribution targets)** — to decode a ground-truth target `t_n` (sufficient statistic, posterior, or mixture vector). A weak probe is used so that decodability reflects information in the embedding, not probe capacity. Three-way data split: set 1 trains the transformer, set 2 trains the probe (on frozen embeddings), set 3 validates the probe. Default synthetic transformer: 3-layer decoder, hidden/embed size 128, 8 heads, dropout 0.1, lr 1e-3, batch 64; continuous inputs enter via a Linear layer instead of a token embedder. Sequences are 500 tokens; splits 10000/3000/1000.

**Important:** the paper does *not* use mutual-information estimators. "What the embedding captures" is measured by **linear-probe decodability** (top-1 accuracy, L2 loss, total-variation loss on simplex targets), complemented by a reverse-direction test, PCA visualization, and the control experiments below.

### Synthetic data-generating processes

1. **Bayesian conjugate models** (§4.1, A.2): Gaussian-Gamma (unknown mean/precision), Beta-Bernoulli, Gamma-Exponential. Each sequence is i.i.d. given its own latent parameters drawn from the prior. Probe targets: the sufficient statistics and the moments of the posterior `p(θ|x_{1:n})`.
2. **Discrete hypothesis spaces** (§4.1): each sequence is 2-D points sampled uniformly from an unknown axis-aligned rectangle on an 8×8 integer grid — 784 possible hypotheses. Probe target: the full 784-dim posterior simplex `p(h|x_{1:n})`. Two difficulty variants (equal/unequal widths) and sample sizes 20/50; splits 20000/19000/1000.
3. **Hidden Markov model** (§4.2): C=4 states, V=64 vocab, transition/emission rows from Dirichlet priors (γ=0.5; δ ∈ {0.5, 1} controls how state-distinct emissions are). Ground-truth belief states computed exactly with the **forward-backward algorithm**; Viterbi for hard decodes.
4. **LDA topic models** (§4.3): V=1000, K=5 topics, 10⁴ documents × 100 words, Dirichlet α ∈ {0.5, 0.8, 1} (larger α = harder). Probe target: the document topic mixture θ_i. Four models compared: small autoregressive transformer (AT; 4 layers, 128-dim, 655,336 params), BERT-tiny (608,747 params), LDA itself (upper bound), and an end-to-end trained word embedder + probe (WE; an upper bound on what a probe could extract from word identity alone).
5. **HMM-LDA** (§4.4.1, A.2.4): words are split between an LDA "semantic" class (fraction `p`) and HMM "syntactic" classes — a partially exchangeable sequence. Sequences of 400 tokens; splits 10000/1000/1000.
6. **Natural corpora** (§4.4.2): 20Newsgroups (11,314 train / 7,532 val docs) and WikiText-103 (28,475 / 60). Targets are topic mixtures (K ∈ {20, 100}) inferred by LDA trained on each corpus. Probes are trained on frozen embeddings of pretrained GPT-2 (124M/355M/774M), Llama-2 7B and Llama-2-chat 7B, BERT 110M/336M; a randomly initialized "Null GPT-2" controls for probe-only capacity.

### Control / invariance experiments (methodology highlights)

- **Out-of-distribution generalization** (§4.1.2): probe on data generated with *different prior hyperparameters* (e.g. Gaussian-Gamma OOD hyperparameters {2,1,5,1} vs in-distribution {5,1,1,1}). A Bayes-optimal agent just updates its posterior, so decodability should transfer — and it does.
- **Memorization control** (§4.1.2, Fig. 4): probe the *token values themselves* from the 10th-token embedding. The 10th token is recovered perfectly, tokens 1–9 are not (only the weak correlation expected from a single token's information about the generating distribution) — embeddings store sufficient statistics, not memorized context.
- **Parsimony / reverse direction** (§4.1.2): train an MLP to predict the embedding *from* the sufficient statistics. Sufficient statistics explain well over 50% of embedding variance in most cases — the statistic↔embedding relationship runs both ways, i.e. embeddings are close to minimal.
- **Alternative-target controls** (§4.2, Table 2): in the HMM, probing the hypothesized predictive sufficient statistic `p(z_{n+1}|x_{1:n})` from the matched embedding beats (a) mismatched embedding/target pairs, (b) the belief-state target used in prior work `p(z_{n+1}|x_{1:n+1})`, and (c) the argmax Viterbi hard decode ẑ (59.8% vs 90.8% accuracy) — the *full distribution* is encoded better than any point estimate.
- **Probe-capacity control** (§4.3.2, Fig. 5a/b): train 5 ATs and 5 BERTs on 5 datasets from 5 *distinct* topic models, then cross-probe all model×dataset pairs. AT probes succeed only on the matching topic model — proving the topic information lives in the LM's representation, not constructed by the probe from generic word embeddings. BERT shows the same pattern but weaker.
- **Null-model control** (§4.4.2): Null (random-init) GPT-2 reaches only 27.3% (K=20) / 13.8% (K=100) on 20NG vs ~62.9% / 43.2% for trained LLMs.
- **Prediction-quality correlation** (§4.4.2, Fig. 5c, A.4): probes trained at 100 document positions; probe accuracy decreases as the LM's per-token perplexity increases. A linear mixed-effects model on 701,243 20NG tokens (`perplexity ~ token_position + topic_accuracy + (1|document)`) confirms the effect is significant after controlling for position — supporting that encoding the latent posterior *is* what drives next-token prediction.

## Key results

- Conjugate models: linear probes recover sufficient statistics and posterior moments, in- and out-of-distribution (Fig. 2, 8, 9).
- Discrete hypothesis spaces: true rectangle recovered with 87.3% (n=20) → 99.5% (n=50) accuracy out of 784 hypotheses; PCA of embeddings clusters by the generating rectangle's position and spread (Fig. 3).
- HMM: `p(z_{n+1}|x_{1:n})` decoded at ~90.8%/90.4% accuracy (δ=0.5/1) — better than all alternative targets (Table 2).
- LDA synthetic: AT decodes topic mixtures at 82.8/75.5/70.5% top-topic accuracy for α=0.5/0.8/1; BERT collapses in harder regimes (83.6/51.5/46.6%); LDA upper bound 87.0/82.6/79.6% (Table 3). Autoregressive objective beats masked objective exactly where the task is hard.
- HMM-LDA: topic mixtures are recoverable once the exchangeable semantic component covers p ≳ 0.4 of tokens (Fig. 7) — the theory extends to partially exchangeable, realistic data.
- Natural corpora: LDA topic mixtures decode well above chance (5%/1%) from pretrained LLM embeddings — best 88.5%/74.2% (WikiText-103) and 62.9%/43.2% (20NG) with averaged last-layer embeddings; last-token embeddings still strong (73.7/58.9 and 52.2/34.7). Autoregressive LLMs significantly outperform BERT models; topic decodability grows through Llama-2's layers.

## Main conclusions — what AR embeddings do and do not embed

**Do embed:** latent generating distributions — posteriors over parameters (exchangeable data), belief states over latent states (HMMs), posteriors over discrete hypotheses, and topic mixtures — including the *uncertainty* (full distributions, not just point estimates). These representations generalize to OOD hyperparameters like a Bayesian posterior update would, are parsimonious (largely reconstructible from the sufficient statistics), and their presence tracks the model's predictive quality.

**Do not embed (or embed less):** the identities of earlier context tokens (no token memorization); point-estimate/argmax latents (decoded worse than full posteriors); topic structure when the exchangeable component is too weak (p < ~0.4); and non-predictive-sufficient quantities in general (harder to decode). Masked-LM (BERT) embeddings encode the same latent structure but less robustly.

## Relevance

This paper is the direct methodological template for our LeJEPA time-series probing project:

- **Identical research question, different objective.** It answers "what should embeddings embed?" for *autoregressive likelihood* training, via the predictive-sufficient-statistic / Bayesian-posterior framing. LeJEPA-style joint-embedding predictive pretraining replaces explicit next-step likelihood with prediction in representation space plus regularizers (e.g. VICReg-style variance/invariance terms, SIGReg). The open question our project asks — whether JEPA embeddings likewise come to encode predictive sufficient statistics (posteriors over latent generating factors of the time series), or instead encode only what the regularizer forces them to keep — is exactly the LeJEPA analogue of this paper's question. The paper itself flags time series as an application area in its Discussion (§5).
- **Portable experimental recipe.** (a) Generate synthetic data from processes with *known, analytically computable* latents (conjugate exchangeable models, HMMs with forward-backward ground truth, mixture models); (b) train the encoder; (c) probe frozen embeddings with deliberately weak **linear** probes against the exact posterior/sufficient statistic; (d) run the control battery: OOD-hyperparameter generalization, memorization probing, parsimony/reverse prediction, alternative-target comparisons, cross-dataset probing, and a random-init null encoder. Every one of these transfers directly to time-series DGPs (switching regimes, AR/state-space models, mixture dynamics).
- **The HMM experiment is already a (discrete) time-series result.** Probing `p(z_{n+1}|x_{1:n})` from sequence embeddings *is* belief-state probing of a latent-regime time series — the discrete analogue of probing LeJEPA embeddings for latent regime/state posteriors of a switching dynamical system.
- **Concrete hypotheses and metrics to reuse.** Full distributions should decode better than point estimates; decodability should track prediction quality (their perplexity correlation analysis); embeddings should be parsimonious (reverse-predictable from the latents) and non-memorizing. Their metrics (top-1 accuracy, L2, total variation on simplex targets) suit distribution-valued probing targets; note they do *not* use mutual-information estimation — if we want MI-style quantification we would need to add it ourselves.
- **Contrast class.** Their BERT-vs-AT comparison shows the training objective materially changes what gets embedded (masked objective degrades in hard regimes). This legitimizes expecting LeJEPA embeddings to differ systematically from autoregressive ones — and gives a baseline (this paper's AR results) to compare against.
