# Representativeness, Not Just Effective Sample Size: A Probe for Batch-Composition Failures of SIGReg on Time Series

*A Statistical and Empirical Analysis for JEPA-Style Forecasting*

*Research Proposal*

---

## Abstract

SIGReg is a regularization method for self-supervised learning that enforces an isotropic Gaussian marginal on embeddings. It has shown promise in robotics and time series, but on temporally correlated data, the empirical marginal within a mini-batch can be badly unrepresentative of the overall data distribution. When a batch is constructed from contiguous windows of one or a few sequences, it concentrates in a small region of the latent space, creating a spuriously non-Gaussian batch marginal. We distinguish two distinct statistical effects: *effective sample size* controls the variance of the batch statistic, while *representativeness* controls its systematic bias. We prove that under homogeneous batch construction the representativeness bias persists under data-parallel training, gradient accumulation, and longer training, whereas effective-sample-size effects diminish. We further show, via a constructive example, that this bias can drive different dynamic regimes to overlap in the history embedding space, and that such overlap provably increases the minimal linear forecasting loss. The central deliverable is a *representativeness probe*: a lightweight two-sample statistic comparing each batch's empirical embedding distribution against a corpus-level reference, computable at or near initialization, intended to warn practitioners before harmful batch constructions cause damage. We plan to validate the mechanism and the probe on synthetic Markov-chain data with controlled regime structure, and to test whether the probe predicts final forecasting quality on real time-series benchmarks.

**Keywords**: self-supervised learning, JEPA, SIGReg, batch composition, representativeness, effective sample size, two-sample testing, temporal correlation

---

## 1. Introduction

Predictive self-supervised learning learns representations by predicting held-out structure from visible context. The Joint-Embedding Predictive Architecture (JEPA) is a prominent example: by predicting latent representations rather than raw inputs, JEPA avoids reconstructing irrelevant detail and focuses on predictive structure, with demonstrated success in images [@assran2023ijepa], video [@bardes2024vjepa], and robotics [@maes2026lewm].

Among methods for preventing representational collapse in JEPA, regularization-based approaches are attractive for their simplicity. SIGReg [@balestriero2025lejepa] enforces an isotropic Gaussian marginal on embeddings via the Epps--Pulley statistic [@epps1983epps]. Under i.i.d. sampling assumptions, the isotropic Gaussian is provably minimax-optimal for downstream linear prediction [@balestriero2025lejepa]. SIGReg has enabled stable end-to-end training in robotics [@maes2026lewm] and has been adopted for time series [@petersen2026hepa].

Time series, however, violate the i.i.d. assumption. SIGReg computes its statistic on the empirical marginal of embeddings within each mini-batch. When a batch is assembled from contiguous windows of one or a few sequences, adjacent samples are correlated, the batch concentrates in a small region of the state space, and its empirical marginal can appear strongly non-Gaussian even when the true data marginal is well matched to a Gaussian. The regularizer's gradient then applies a corrective pressure whose strength the true marginal would not warrant---a miscalibration that can distort latent structure the prediction objective is trying to learn. A related failure mode was observed in multi-task robotic learning, where Gaussianization compressed task-dependent latent clusters, motivating temporally centered SIGReg [@liu2026tclwm].

We hypothesize that the deciding factor is *batch representativeness*: how faithfully each batch's empirical embedding distribution reflects the corpus-level marginal. We separate this from *effective sample size*, which has been the primary lens in prior work. Our theoretical analysis shows that effective sample size mostly affects variance, while representativeness induces a bias that does not vanish with more data-parallel workers, larger batches, or longer training. This distinction is the foundation of our contribution.

We then ask whether the failure can be predicted early---even at initialization---and propose a *representativeness probe*: a lightweight two-sample statistic comparing each batch's empirical embedding distribution against a corpus-level reference. The probe is intended as a monitoring instrument, not as a universal batch-construction recipe. We plan to (a) establish the mechanism on synthetic Markov-chain data with controlled regime structure, (b) construct and validate the probe against ground truth, and (c) test whether the probe predicts final forecasting quality on real benchmarks.

## 2. Related Work

### 2.1. Joint-Embedding Predictive Architectures

Early self-supervised learning relied on contrastive objectives with negative samples [@chen2020simclr; @he2020moco]. Non-contrastive methods eliminated negatives through architectural asymmetry: BYOL [@grill2020byol] used a predictor network and stop-gradient, SimSiam [@chen2021simsiam] removed the momentum encoder, and DINO [@caron2021dino] used centering and sharpening. JEPA [@lecun2022jepa] unified this trajectory by predicting latent representations of unseen regions from visible context; I-JEPA [@assran2023ijepa] applied it to images, V-JEPA [@bardes2024vjepa] to video, and LeWM [@maes2026lewm] demonstrated stable end-to-end training from raw pixels.

Regularization-based variants prevent collapse through explicit constraints on the embedding space: Barlow Twins [@zbontar2021barlow] minimizes cross-correlation redundancy, VICReg [@bardes2022vicreg] regularizes variance and covariance, and W-MSE [@ermolov2021wmse] whitens embeddings via Cholesky decomposition. SIGReg [@balestriero2025lejepa] instead matches the full marginal to an isotropic Gaussian via the Epps--Pulley statistic on random 1D projections; its minimax optimality is proven under i.i.d. assumptions whose relaxation to dependent data is precisely our subject. Recent variants include KerJEPA [@zimmermann2025kerjepa], LpJEPA [@kuang2026lpjepa], and VISReg [@wu2026visreg].

For temporally correlated data, JEPA has been applied to video [@bardes2024vjepa], robotics [@maes2026lewm], and time series [@tsjepa2025; @petersen2026hepa]. Closest to our concern, TC-LeWM [@liu2026tclwm] observed that marginal Gaussianization compresses task-dependent latent clusters and proposed Gaussianizing temporally centered embeddings instead---a model-side fix that changes the regularization target. Klindt et al. [@klindt2026lejepa] proved LeJEPA linearly identifies latent variables if and only if the latent distribution is Gaussian, and reported a non-monotonic interaction between regularization strength and temporal correlation. Neither work analyzes the batch-level sampling mechanism behind these observations, which is the gap we address; our contribution is complementary to TC-SIGReg, as we discuss in Section 4.

### 2.2. Batch Statistics and Effective Sample Size

Batch-statistics estimators are known to be sensitive to what the batch contains. The BatchNorm literature [@singh2019evalnorm; @unravelingbn2023] documents train/test discrepancy and shows that batch composition---there, class diversity---determines estimation quality. Experience replay in reinforcement learning [@lin1992replay; @mnih2015dqn; @schaul2016prioritized] addresses a cousin of our problem: decorrelating consecutive samples to stabilize training.

For correlated data, classical statistics provides the notion of effective sample size: under autocorrelation $\rho_k$, $N_{\mathrm{eff}} = N/(1+2\sum_k \rho_k)$, and estimation error scales as $1/\sqrt{N_{\mathrm{eff}}}$ [@geyer1992mcmc; @flegal2010batchmeans; @stan2023ess]. These tools are standard in the MCMC literature but, to our knowledge, have not been applied to regularization statistics in SSL. Prior analyses of batch size in SSL [@vaessen2024batchsize] attribute batch-size effects to gradient noise rather than to representativeness of the batch sample.

### 2.3. Time-Series Representation Learning

Time-series representation learning has been dominated by contrastive methods [@yue2022ts2vec] and reconstruction-based masked modeling. TS-JEPA [@tsjepa2025] and HEPA [@petersen2026hepa] bring JEPA to time series, with HEPA adopting SIGReg-style regularization. Neither analyzes how batch construction interacts with the regularizer under temporal correlation.

## 3. Research Gap

Three gaps converge on this work. First, regularization-based JEPA is known to *sometimes* damage latent structure [@klindt2026lejepa; @liu2026tclwm], but the batch-level mechanism---when and why the regularizer's pressure becomes miscalibrated under temporal correlation---has not been analyzed. Second, the statistical tools for this analysis (effective sample size, two-sample discrepancy) exist but have not been connected to SSL regularization. Third, practitioners currently have no instrument to tell whether their data loader is harming the regularizer: the only available remedies are blind fixes (shuffle everything, enlarge batches) or modifying the loss itself [@liu2026tclwm].

Our positioning relative to TC-SIGReg deserves emphasis. TC-SIGReg changes *what* is Gaussianized; we leave the objective untouched and ask *when* the naive objective fails and how to detect it. The two are complementary: a representativeness probe is useful when the loss cannot be modified (fixed codebases, pretrained pipelines), when temporal centering would discard task-relevant slow structure, and as a monitoring tool regardless of which regularizer is used.

## 4. Problem Formulation

### 4.1. Setup

We consider JEPA-style self-supervised learning for time-series forecasting. An encoder maps each observation to a latent embedding $z_t = f_\theta(x_t)$; a context network aggregates history into $h_t = g_\phi(z_{1:t})$; the next embedding is predicted linearly as $\hat{z}_{t+1} = W h_t$. Training minimizes $\mathcal{L} = \mathcal{L}_{\text{pred}} + \lambda \cdot \mathcal{L}_{\text{reg}}$, where $\mathcal{L}_{\text{pred}} = \|W h_t - z_{t+1}\|^2$ and $\mathcal{L}_{\text{reg}}$ is the SIGReg regularizer enforcing $\mathcal{N}(0,I)$ marginality of $z_t$ via the Epps--Pulley statistic. We focus on the univariate case for analytical tractability, but the arguments generalize to multivariate embeddings.

### 4.2. An Ideal Terminal State Exists

Before diagnosing failures, we ask whether the three properties a practitioner wants---(i) marginal Gaussianity of $z_t$, (ii) decodability of $x_t$ from $z_t$, and (iii) linear predictability of $z_{t+1}$ from $h_t$---are mutually compatible at all. They are, under mild regularity. The following proposition formalizes the constructive argument.

**Proposition 1 (Existence of an ideal terminal state).** *Let the data-generating process admit a latent representation $\tilde{z}_t$ that is a sufficient statistic for $x_t$ and whose marginal is absolutely continuous with respect to the Lebesgue measure. Assume that the conditional expectation $m(\tilde{z}_{1:t}) = \mathbb{E}[\tilde{z}_{t+1} \mid \tilde{z}_{1:t}]$ is well-defined and can be approximated by a neural network. Then there exists an encoder $f_\theta$ and a context network $g_\phi$ such that:*

1. *$z_t = f_\theta(x_t)$ has marginal distribution $\mathcal{N}(0,I)$;*
2. *$x_t$ is recoverable from $z_t$ via an invertible transformation;*
3. *there exists a linear map $W$ with $\|W h_t - z_{t+1}\|^2$ arbitrarily small.*

The proof is given in Appendix A. The key idea is that any atomless distribution can be pushed forward to a standard Gaussian via an invertible differentiable map $F$. Setting $z_t = F(\tilde{z}_t)$ yields (i) and (ii); the context network can then learn to invert $F$ on the history, compute the conditional mean $m$, and map the result forward through $F$, placing the prediction in a designated subspace of $h_t$ for the linear readout. This demonstrates that Gaussianization *per se* does not preclude a good representation.

Two caveats delimit this argument. When the underlying structure is genuinely discrete (an atomic marginal), no continuous invertible push-forward to a Gaussian exists, and Gaussianization necessarily distorts cluster geometry; we therefore treat structural incompatibility as a separate, already-studied phenomenon [@klindt2026lejepa; @liu2026tclwm] and focus on the regime where a compatible configuration exists. Second, existence says nothing about whether gradient descent *finds* this state---which is where batch composition enters.

### 4.3. Batch Composition: Effective Sample Size versus Representativeness

SIGReg evaluates its statistic on the empirical marginal of embeddings within each mini-batch. When the batch is a representative sample of the training corpus, the empirical marginal approximates the true marginal and the regularizer's pressure is well-calibrated. When the batch is assembled from contiguous windows of one or a few sequences---standard practice for time series---it can concentrate in a small region of the state space. The batch then appears spuriously non-Gaussian, the Epps--Pulley statistic is inflated, and the resulting gradient pushes embeddings apart with a strength the true marginal would not warrant, potentially pulling the optimizer away from the ideal terminal state.

We define two measurable batch properties.

**Effective sample size.** Under autocorrelation $\rho_k$, the effective sample size is $N_{\mathrm{eff}} = N / (1 + 2\sum_k \rho_k)$, where $N$ is the nominal batch size. For a stationary, representative batch, the deviation of a regularizer statistic from its population value scales as $O(N_{\mathrm{eff}}^{-1/2})$ and is unbiased in expectation up to finite-sample corrections. Thus effective sample size controls the *variance* of the batch statistic.

**Representativeness.** We define representativeness operationally as the discrepancy between the batch's empirical embedding distribution and a corpus-level reference distribution, measurable by a two-sample statistic. A batch is representative if its empirical distribution is close to the reference; it is unrepresentative if it concentrates on a subset of dynamic regimes. Representativeness controls the *bias* of the batch statistic: even as the batch size grows, if each batch is drawn from a single regime, the expected statistic does not converge to the population statistic.

### 4.4. A Formal Decomposition of Bias and Variance

Let $P_r$ denote the latent distribution of regime $r$, and let $\pi_r$ be its corpus-level weight. The population marginal is $P = \sum_r \pi_r P_r$. Let $S$ be the Epps--Pulley statistic (or any nonlinear functional of the distribution). For a representative batch of size $N$, we have

$$
\mathbb{E}[S(\hat{P}_B)] = S(P) + O(N_{\mathrm{eff}}^{-1}).
$$

For a homogeneous batch of size $N$ drawn entirely from regime $r$, we have

$$
\lim_{N_{\mathrm{eff}}\to\infty} \mathbb{E}_{B\sim q}[S(\hat{P}_B)] = \sum_r \pi_r S(P_r) \neq S\Bigl(\sum_r \pi_r P_r\Bigr),
$$

where the inequality follows from the nonlinearity of $S$. Thus representativeness creates a non-vanishing bias, while effective sample size only creates a finite-sample error.

This distinction is formalized in Appendix B. We state a proposition:

**Proposition 2 (Bias vs. variance).** *Let $S$ be a nonlinear functional of the empirical distribution, such as the Epps--Pulley statistic. Under homogeneous batch construction, the expected batch statistic converges to $\sum_r \pi_r S(P_r)$, while under representative batch construction it converges to $S(P)$. The difference is non-zero for any non-trivial mixture, and it does not vanish as the batch size or number of batches increases.*

The proof follows immediately from the nonlinearity of $S$ and is given in Appendix B.

### 4.5. Bias Persistence under Data Parallelism and Gradient Accumulation

A common practitioner intuition is that larger batches, more GPUs, or gradient accumulation will fix batch statistics. The following proposition shows this is false for representativeness bias.

**Proposition 3 (Bias persistence).** *Consider a training setup where each local batch is homogeneous, and gradients are averaged across $M$ local batches either in data-parallel training or via gradient accumulation. As $M \to \infty$, the averaged gradient converges to*

$$
\nabla_\theta \sum_r \pi_r S(P_r(\theta)),
$$

*which is not equal to $\nabla_\theta S(\sum_r \pi_r P_r(\theta))$. Thus representativeness bias is not reduced by more workers or accumulation steps.*

The proof is in Appendix C. The key reason is that the gradient operator is linear in the chain-rule sense but nonlinear in the distribution argument: averaging gradients of $S$ over local batches is not the same as taking the gradient of $S$ on the aggregated batch.

### 4.6. An LDA Decomposition of the Embedding Space

To connect batch-level bias to forecasting performance, we decompose the learned embedding covariance by regime, in the spirit of linear discriminant analysis. Let the hidden regime indicator be $s_t \in \{1,\dots,K\}$ with stationary weights $\pi_s$. For a learned representation $u_t$ (which may be $z_t$ or $h_t$), define the regime-conditional means

$$
\mu_s = \mathbb{E}[u_t \mid s_t = s], \qquad \bar{u} = \sum_s \pi_s \mu_s.
$$

The total covariance decomposes as $\Sigma_T = \Sigma_W + \Sigma_B$, where

$$
\Sigma_W = \sum_s \pi_s\,\mathbb{E}[(u-\mu_s)(u-\mu_s)^\top \mid s], \qquad
\Sigma_B = \sum_s \pi_s\,(\mu_s - \bar{u})(\mu_s - \bar{u})^\top .
$$

Taking traces gives scalars $V_T = V_W + V_B$ with

$$
V_W = \mathrm{tr}(\Sigma_W), \qquad V_B = \mathrm{tr}(\Sigma_B) = \sum_s \pi_s \|\mu_s - \bar{u}\|^2 .
$$

Superposition is then characterized by a small between-regime scatter relative to within-regime scatter; e.g., the Fisher-like ratio $J = V_B / V_W$ is low. This decomposition serves two purposes: it identifies *which* statistic a biased regularizer attacks ($V_B$), and it motivates the variance-based probe of Section 6.

### 4.7. Mean-Shift Monotonicity of the Epps--Pulley Statistic

The mechanism by which a biased SIGReg gradient damages latent structure rests on a monotonicity property of the Epps--Pulley statistic under mean shifts.

**Lemma 1 (Mean-shift monotonicity).** *Let $S$ be the Epps--Pulley criterion. For any centered distribution $P_0$ with mean $0$, the function $\mu \mapsto S(T_\mu P_0)$ is nondecreasing in $\|\mu\|$, with its minimum at $\mu = 0$.*

The proof is in Appendix D. Intuitively, shifting a distribution multiplies its characteristic function by $e^{i\mu^\top t}$, and the induced oscillation moves the weighted $L^2$ distance to the Gaussian characteristic function away from its minimum.

### 4.8. From Biased Gradients to Shrunken Between-Regime Scatter

Under homogeneous batches, each local batch contains samples predominantly from one regime. By Lemma 1, SIGReg penalizes a regime-specific distribution more as its mean moves away from zero, so the regularizer gradient acts to shrink each regime's mean toward the origin, reducing $V_B$. In the extreme case of point-mass regimes, the homogeneous regularizer drives $\mu_s \to 0$ for all $s$, collapsing $V_B$ to zero; Proposition 4 below makes this precise for a tractable DGP.

In contrast, a representative batch contains the mixture $\sum_r \pi_r P_r$. For a symmetric two-point distribution $0.5\,\delta_{\mu} + 0.5\,\delta_{-\mu}$, the Epps--Pulley statistic attains its minimum at some $\mu^* > 0$, not at $\mu = 0$ (Appendix E). Thus the representative regularizer preserves a nonzero between-regime scatter.

### 4.9. A Constructive Example: Homogeneous Batches Force Collapse

To show that the biased gradient can indeed lead to harmful latent overlap, we analyze a tractable class of data-generating processes.

**Proposition 4 (Collapse under homogeneous batches).** *Consider a two-state deterministic cycle with observations $x_A = \mu_A$, $x_B = \mu_B$. Let the encoder be linear, $z_t = U x_t$, and let the context be the identity $h_t = z_t$. Then:*

1. *Under homogeneous batches (each batch containing only one state), the SIGReg gradient pushes $z_A$ and $z_B$ toward the origin, leading to a collapsed latent state in which both regimes map to the same point.*
2. *Under representative batches (each batch containing both states in equal proportion), the regularizer preserves a nonzero symmetric separation $z_A = -z_B \neq 0$.*

*Consequently, homogeneous batches create a stationary point with zero latent prediction loss but complete loss of linear decodability of the regime.*

The proof is in Appendix E. The mechanism is that the Epps--Pulley statistic for a point mass at $u$ is an increasing function of $\|u\|$, so the regularizer gradient points toward zero; for a symmetric two-point distribution $0.5\delta_u + 0.5\delta_{-u}$, the statistic has a nonzero minimizer, so the regularizer does not force $u\to 0$.

This example closes the link between biased batch gradients and harmful latent overlap for a specific DGP. It is not a universal theorem, but it demonstrates the mechanism concretely.

### 4.10. Overlap in History Space Hurts Forecasting

The final analytical piece is a lower bound showing that if different regimes' histories overlap in $h_t$ space, the minimal linear forecasting loss increases. We give two forms: an expectation-based bound phrased in the LDA quantities, and a norm-based bound phrased in pairwise history overlap.

**Theorem 1 (Lower bound on linear forecasting error).** *Suppose the data follows a regime mixture with regime-conditional next-step means $\mu_s^{\mathrm{future}} = \mathbb{E}[z_{t+1} \mid s_t = s]$. Let $h_t$ be the context embedding from which a linear predictor $W$ must predict $z_{t+1}$, and define the linear approximation error*

$$
\epsilon^2_{\mathrm{lin}} = \min_{W,b}\; \mathbb{E}\,\bigl\|\mu_{s_t}^{\mathrm{future}} - (W h_t + b)\bigr\|^2 .
$$

*Then the minimal linear forecasting loss satisfies*

$$
\mathcal{L}_{\mathrm{pred}} \;\ge\; \mathbb{E}\|\varepsilon\|^2 + \epsilon^2_{\mathrm{lin}},
$$

*where $\varepsilon$ is irreducible regime-conditional noise. Under complete superposition ($\mu_{s_t}^{\mathrm{future}}$ independent of $h_t$), the best linear predictor is the unconditional mean and $\epsilon^2_{\mathrm{lin}} = \sum_s \pi_s \|\mu_s^{\mathrm{future}} - \bar{\mu}^{\mathrm{future}}\|^2 = V_B^{\mathrm{future}}$.*

The proof is in Appendix F. Thus reducing between-regime scatter in $h_t$ directly raises the minimal forecasting error. The norm-based variant below gives the same conclusion without an expectation decomposition.

**Proposition 5 (Forecasting lower bound under history overlap).** *Let $h_A$ and $h_B$ be context vectors for two histories with distinct future conditional means $\mu_A \neq \mu_B$. Suppose $\|h_A - h_B\| \le \epsilon$. Then for any linear predictor $W$ with operator norm bounded by $M$, there exist inputs for which the prediction error is at least*

$$
\frac{1}{4}\left(\|\mu_A - \mu_B\| - M\epsilon\right)^2.
$$

*In particular, if $h_A = h_B$, no linear predictor can distinguish the two futures.*

The proof is in Appendix F. This is the rigorous version of the intuition that "trace crossing" or "superposition" hurts forecasting: not just raw latent overlap, but overlap of the history embeddings that the linear predictor receives.

### 4.11. The Core Problem

We can now state the research question concisely: *Does homogeneous batch composition drive the optimizer away from the ideal terminal state, and can we detect this deviation early with a lightweight two-sample probe?* The theoretical chain is:

$$
\text{non-representative batches} \longrightarrow \text{biased SIGReg gradient} \longrightarrow \text{shrinkage of regime means} \longrightarrow V_B \downarrow,\; J \downarrow \longrightarrow \text{increased forecasting loss}.
$$

The first arrow is proven (Propositions 2 and 3) and the last is proven (Theorem 1 and Proposition 5); the middle arrow is supported by the mean-shift mechanism (Lemma 1), proven for a tractable class (Proposition 4), and hypothesized generally. The probe is designed to monitor the first arrow without needing ground-truth regime labels.

## 5. Research Design

Our study proceeds in three stages: (1) a controlled synthetic investigation to establish the causal mechanism; (2) construction and validation of the representativeness probe; (3) a real-data test of whether the probe predicts final performance.

### 5.1. Stage 1: Synthetic Markov-Chain Experiments

We generate data from hidden Markov models with a finite set of discrete latent states and continuous emissions, chosen so that a Gaussian-compatible embedding configuration exists (verifiable via the push-forward construction of Proposition 1). This isolates the sampling mechanism from structural incompatibility. We construct batches in three ways:

1. **Representative**: windows drawn uniformly across regimes, so the batch matches the corpus mixture.
2. **Moderately unrepresentative**: windows from a subset of regimes.
3. **Highly unrepresentative**: a long contiguous segment dominated by one regime.

Architecture, learning rate, and $\lambda$ are held constant. We measure:

- reconstruction fidelity: $R^2$ of a linear probe from $z_t$ to the true state identity (or raw observation);
- forecasting performance: next-step prediction error via a linear probe on $h_t$;
- the SIGReg statistic value itself, as a function of batch construction.

We expect condition (C) to degrade reconstruction and forecasting relative to (A), and---critically---that condition (A) demonstrates that multiple dynamic regimes do not by themselves prevent SIGReg from working.

We also evaluate a round-robin scheme where each batch is still homogeneous but we cycle through regimes. According to Proposition 3, this should still suffer representativeness bias, though the effect may be smaller than using contiguous windows from a single sequence. This directly tests the theoretical claim that bias does not average out over mini-batches.

Baselines include shuffled-window batching (windows shuffled across sequences with within-window temporal order intact, preserving the context needed by the prediction loss) and, where applicable, TC-SIGReg [@liu2026tclwm].

### 5.2. Stage 2: Probe Construction and Validation

The reference distribution is obtained by caching embeddings of a large, shuffled sample of the training corpus. For each batch $B$, the probe computes a discrepancy $\Delta(B) = D(\hat{P}_B, \hat{P}_{\mathrm{ref}})$. Candidate statistics $D$:

- difference between batch and reference Epps--Pulley statistics (nearly free if already computed);
- kernel MMD with an RBF kernel;
- energy distance;
- variance-ratio or effective-rank measure.

We evaluate candidates on synthetic data by their correlation with final forecasting performance, selecting the best predictor.

**Variance discrepancy probe.** The LDA decomposition of Section 4 predicts that harmful batch composition specifically removes between-regime scatter, so the batch variance falls below the corpus variance. This motivates the variance discrepancy

$$
\Delta = \bigl| \overline{\mathrm{Var}}(B) - \mathrm{Var}(D) \bigr|,
$$

where $\overline{\mathrm{Var}}(B)$ averages per-batch variance over several batches drawn from the actual strategy, and $\mathrm{Var}(D)$ is the variance over a large, shuffled reference set. Variance is robust and cheap to estimate even with small batches, and the probe can be computed at initialization since even a random encoder approximately preserves relative spreads. To make the probe comparable across batch sizes and datasets, we optionally standardize it: estimate the mean $\mu_{\mathrm{ref}}$ and standard deviation $\sigma_{\mathrm{ref}}$ of the variance over many i.i.d. reference batches of the same size, and report the $z$-score $z = (\bar{V}_{\mathrm{actual}} - \mu_{\mathrm{ref}}) / (\sigma_{\mathrm{ref}} / \sqrt{K})$ for $K$ actual batches.

**Training-free hypothesis.** We test whether the probe works *training-free*. A randomly initialized encoder acts as an approximate random projection, which preserves pairwise distance structure in expectation (a Johnson--Lindenstrauss-type intuition); a homogeneous batch may therefore already appear concentrated at initialization. We treat this as a hypothesis to be tested, not a guarantee: we measure the probe's correlation with final performance at initialization and after a short warm-up ($\sim$100 steps). If the signal is weak at initialization, the warm-up version remains a practical early-warning instrument at negligible cost. In practice, a practitioner can estimate autocorrelation from the raw series as a proxy for assessing risk before training begins.

**Explicit TBD work.** The exact form of the probe is left as an explicit research task. The synthetic setup provides ground-truth labels for representativeness, allowing us to compare candidate probes not only against final performance but also against the true batch composition. We will report a decision curve: given a probe threshold, what is the false-positive and false-negative rate for detecting harmful batch construction?

### 5.3. Stage 3: Real-Data Validation

We test on univariate forecasting benchmarks with diverse autocorrelation profiles (e.g., electricity, traffic, weather). For each dataset we fix nominal batch size and vary batch construction: many short clips vs. few long clips, stride between windows, mixing across sequences. After warm-up (or at initialization, if Stage 2 supports it), we record the probe value, train to convergence, and measure the correlation between early probe value and final forecasting error across strategies. A quantitative prediction follows from the mechanism: effect sizes should grow with the dataset's estimated autocorrelation time.

We note two scope limits: real series are often non-stationary, and our probe's reference-based definition of representativeness is intended to remain meaningful in that regime (it compares batch to corpus, not to a stationary law); and shuffling windows across sequences incurs I/O overhead that must be weighed against the benefit, which we report.

### 5.4. Mapping to the Quantitative Questions

Stage 1 addresses Q1 (monotonicity in autocorrelation, sweeping transition-matrix persistence), Q2 ($N_{\mathrm{eff}}$ as governing parameter, sweeping batch size and correlation jointly), and Q3 (representativeness threshold, sweeping regime coverage). Stage 2 addresses Q4 in a weakened, testable form: rather than measuring each link of the causal chain independently, we test whether the mechanism's end-to-end prediction---early probe value predicts late performance---holds. Stage 3 tests external validity.

## 6. Research Impact

For practitioners, the deliverable is a monitoring instrument: compute the probe early in training, and if it is high, restructure the data loader (more distinct sequences per batch, larger stride) using domain expertise. This targets cases where modifying the loss is impossible or undesirable, and complements structural fixes such as TC-SIGReg [@liu2026tclwm]; it reduces the risk of silently degraded representations rather than claiming to improve state of the art.

For SSL researchers, we provide a mechanistic account of how batch-statistics regularizers can misfire on non-i.i.d. data, plus a reusable synthetic benchmark and validation protocol. The probe concept may extend to other batch-level regularizers (VICReg, Barlow Twins), but their statistics differ (variance, cross-correlation) and we make no claim that the same thresholds or fixes transfer without further study.

For the time-series community, the work argues that batch construction is a first-class design consideration alongside architecture and objective, and suggests evaluation protocols that report performance as a function of batching strategy. The broader message is methodological: SSL regularizers inherit classical statistics' sensitivity to sample representativeness, and classical tools---effective sample size, two-sample discrepancy---are practical instruments for diagnosing it.

---

## Appendix A: Existence of an Ideal Terminal State

**Proof of Proposition 1.** Let $\tilde{z}_t \in \mathbb{R}^d$ be a sufficient statistic for $x_t$ with atomless marginal $P_{\tilde{z}}$. By the probability-integral transform (or the Rosenblatt transform in the multivariate case), there exists an invertible differentiable map $F : \mathbb{R}^d \to \mathbb{R}^d$ such that $F(\tilde{z}_t) \sim \mathcal{N}(0,I)$. Define the encoder $f_\theta$ to first compute $\tilde{z}_t$ and then apply $F$, i.e., $z_t = F(\tilde{z}_t)$. Then property (i) holds by construction.

Since $F$ is invertible, $F^{-1}(z_t) = \tilde{z}_t$, and $\tilde{z}_t$ is sufficient for $x_t$; thus there exists a function $d$ such that $d(\tilde{z}_t) = x_t$ (almost everywhere). Define the decoder $\hat{x}_t = d(F^{-1}(z_t))$, which achieves exact reconstruction. Thus property (ii) holds.

For property (iii), let $m(\tilde{z}_{1:t}) = \mathbb{E}[\tilde{z}_{t+1} \mid \tilde{z}_{1:t}]$ be the optimal predictor in $\tilde{z}$-space. Define the context network $g_\phi$ to first invert $F$ on each element of the history, compute $m$, and then apply $F$ to the result; place this value in a fixed subset of the coordinates of $h_t$, with zeros elsewhere. Let $W$ be the linear projection onto those coordinates. Then $W h_t = F(m(F^{-1}(z_{1:t})))$, and for any sufficiently expressive approximation of $m$, the prediction error $\|W h_t - z_{t+1}\|^2$ can be made arbitrarily small. This establishes property (iii). $\square$

## Appendix B: Effective Sample Size versus Representativeness

Let $P_r$ be the latent distribution of regime $r$, with weights $\pi_r$, and let $P = \sum_r \pi_r P_r$. Let $S(\cdot)$ be the Epps--Pulley statistic, which is a nonlinear functional of the empirical distribution. For a representative batch of size $N$, the empirical distribution $\hat{P}_N$ is an unbiased estimate of $P$ in the sense that $\mathbb{E}[\hat{P}_N] = P$ under i.i.d. sampling. By the standard theory of U-statistics, $S(\hat{P}_N)$ converges to $S(P)$ as $N\to\infty$ with rate determined by the effective sample size; in particular, $\mathbb{E}[S(\hat{P}_N)] = S(P) + O(N_{\mathrm{eff}}^{-1})$.

Now consider a homogeneous batch construction where each batch is drawn from a single regime $r$, and regimes are selected with probability $\pi_r$. Let $q$ denote the distribution over such homogeneous batches. For a fixed $\theta$, the expected statistic is

$$
\mathbb{E}_{B\sim q}[S(\hat{P}_B)] = \sum_r \pi_r \mathbb{E}_{\hat{P}_B \sim P_r}[S(\hat{P}_B)].
$$

As the batch size within each regime grows, the inner expectation converges to $S(P_r)$. Therefore,

$$
\lim_{N_{\mathrm{eff}}\to\infty} \mathbb{E}_{B\sim q}[S(\hat{P}_B)] = \sum_r \pi_r S(P_r).
$$

Because $S$ is nonlinear, this is generally not equal to $S(\sum_r \pi_r P_r)$. For the Epps--Pulley statistic, the nonlinearity arises from the weighted integral of the squared difference between empirical and Gaussian characteristic functions, which does not commute with mixture averaging.

Thus the representativeness bias is the difference

$$
\Delta S = S\Bigl(\sum_r \pi_r P_r\Bigr) - \sum_r \pi_r S(P_r) \neq 0.
$$

This difference does not vanish as the number of batches increases. $\square$

## Appendix C: Bias Persistence under Data Parallelism and Gradient Accumulation

We prove Proposition 3. Let $\theta$ denote the encoder parameters, and let $S_B(\theta) = S(\hat{P}_B(\theta))$ be the regularizer loss for a local batch $B$. In a homogeneous batch construction, each local batch $B_i$ is drawn from a single regime $r_i$ with probability $\pi_{r_i}$. The gradient of the local regularizer is $\nabla_\theta S_{B_i}(\theta)$.

Data-parallel training averages these gradients over $M$ workers:

$$
g_M(\theta) = \frac{1}{M}\sum_{i=1}^M \nabla_\theta S_{B_i}(\theta).
$$

As $M\to\infty$, by the law of large numbers,

$$
g_M(\theta) \to \mathbb{E}_{B\sim q}[\nabla_\theta S_B(\theta)] = \nabla_\theta \mathbb{E}_{B\sim q}[S_B(\theta)] = \nabla_\theta \sum_r \pi_r S(P_r(\theta)),
$$

assuming the interchange of gradient and expectation is valid (which holds under standard regularity conditions).

The ideal gradient under a representative batch would be $\nabla_\theta S(\sum_r \pi_r P_r(\theta))$. Because $\sum_r \pi_r S(P_r(\theta)) \neq S(\sum_r \pi_r P_r(\theta))$ in general, the two gradients differ. Moreover, the difference is independent of $M$; increasing the number of workers only reduces variance, not the bias. The same argument holds for gradient accumulation across sequential micro-batches: the total update is the average of local gradients, which converges to the same biased expectation. $\square$

## Appendix D: Mean-Shift Monotonicity of the Epps--Pulley Statistic

**Proof of Lemma 1.** For a probability measure $P$ on $\mathbb{R}^d$, the Epps--Pulley statistic is

$$
S(P) = \int_{\mathbb{R}^d} \bigl| \phi_P(t) - e^{-\frac12 \|t\|^2} \bigr|^2 \, w(t)\, dt,
$$

where $\phi_P(t) = \mathbb{E}_{X \sim P}[e^{i t^\top X}]$ and $w(t)$ is a rotation-invariant positive weight. Let $P_0$ be centered and define $P_\mu = T_\mu P_0$. Then $\phi_{P_\mu}(t) = e^{i\mu^\top t} \phi_{P_0}(t)$, and since $w$ is rotation-invariant the integral depends on $\mu$ only through $\|\mu\|$. Expanding the squared modulus,

$$
S(P_\mu) = C + \int \Bigl( 1 - 2\,\mathrm{Re}\bigl[ e^{i\mu^\top t} \phi_{P_0}(t) e^{\frac12 \|t\|^2} \bigr] \Bigr) e^{-\|t\|^2} w(t)\, dt,
$$

where $C$ is independent of $\mu$. Because $\phi_{P_0}$ is the characteristic function of a centered distribution, its real part is even (it is real and even whenever $P_0$ is symmetric). The oscillatory factor $e^{i\mu^\top t}$ reduces the cross-term integral as $\|\mu\|$ grows, by the Riemann--Lebesgue lemma applied to the weighted $L^2$ norm. Hence the negative cross term becomes less negative and $S(P_\mu)$ increases; the minimum occurs at $\mu = 0$. $\square$

## Appendix E: Collapse under Homogeneous Batches

We prove Proposition 4 for the two-state deterministic cycle. Let $x_A = \mu_A$, $x_B = \mu_B$ be two distinct observation vectors, and let the encoder be linear: $z_t = U x_t$, with $U \in \mathbb{R}^{d\times d}$ invertible. The two latent representations are $z_A = U\mu_A$ and $z_B = U\mu_B$.

### Homogeneous batch gradient

A homogeneous batch contains either only $z_A$ or only $z_B$. The empirical distribution is $\hat{P}_A = \delta_{z_A}$ or $\hat{P}_B = \delta_{z_B}$. The Epps--Pulley statistic for a point mass at $u$ is known to be an increasing function of $\|u\|$ (for the univariate case this follows from the characteristic function; for the multivariate version applied to 1D projections, the same monotonicity holds after projection). Therefore, the gradient of $S(\delta_{z_A})$ with respect to $z_A$ is proportional to $z_A$, pointing radially outward from the origin in the direction of positive $\|z_A\|$. The regularizer loss $\lambda S(\delta_{z_A})$ thus pushes $z_A$ toward the origin: $z_A \leftarrow z_A - \eta \lambda \nabla_{z_A} S(\delta_{z_A})$, which decreases $\|z_A\|$.

The prediction loss for the deterministic cycle with linear predictor $W$ and context $h_t = z_t$ is zero if $W z_A = z_B$ and $W z_B = z_A$. At the symmetric fixed point $z_A = -z_B$, the optimal $W$ is $-I$. If $z_A$ and $z_B$ both approach zero, the prediction loss remains zero (since $W \cdot 0 = 0$), but the regularizer gradient continues to push them toward zero. Thus zero is a stationary point for the combined loss.

### Representative batch equilibrium

Now consider a representative batch containing both states in equal proportion: $\hat{P} = 0.5\,\delta_{z_A} + 0.5\,\delta_{z_B}$. The Epps--Pulley statistic for this symmetric two-point distribution, projected onto any 1D direction $v$, is a function of the distance between the projected points. For a projection $u = v^\top z_A$, we have $S(0.5\delta_u + 0.5\delta_{-u})$. As $u\to 0$, the distribution collapses to a point mass, which is maximally non-Gaussian in the sense of the Epps--Pulley criterion. As $u\to\infty$, the two points are far apart, which also departs from Gaussianity, but there is an intermediate value $u^* > 0$ where the statistic is minimized. The symmetric configuration $z_A = -z_B$ with $\|z_A\| = u^*$ is therefore a stable equilibrium of the regularizer, not collapse. Under the combined loss, the prediction loss reinforces this by requiring $z_A = -z_B$. Thus representative batches preserve a nonzero separation.

This proves that homogeneous batches force collapse while representative batches preserve separation, at least for this tractable class. $\square$

## Appendix F: Forecasting Lower Bounds

**Proof of Theorem 1.** Assume $z_{t+1} = \mu_{s_t}^{\mathrm{future}} + \varepsilon_{t+1}$ with $\mathbb{E}[\varepsilon_{t+1} \mid s_{1:t+1}] = 0$. The linear forecasting loss decomposes as

$$
\mathbb{E}\|z_{t+1} - (W h_t + b)\|^2 = \mathbb{E}\|\varepsilon\|^2 + \mathbb{E}\|\mu_{s_t}^{\mathrm{future}} - (W h_t + b)\|^2,
$$

because the cross terms vanish. The second term is minimized at $\epsilon^2_{\mathrm{lin}}$ by definition, which gives the bound. If $h_t$ contains no linearly decodable regime information, the best linear predictor of $\mu_{s_t}^{\mathrm{future}}$ is the unconditional mean $\bar{\mu}^{\mathrm{future}}$, and the error is $\sum_s \pi_s \|\mu_s^{\mathrm{future}} - \bar{\mu}^{\mathrm{future}}\|^2 = V_B^{\mathrm{future}}$. $\square$

**Proof of Proposition 5.** Let $h_A$ and $h_B$ be two context vectors whose associated next-step latent means are $\mu_A$ and $\mu_B$, with $\delta = \|\mu_A - \mu_B\|$. Let $W$ be a linear predictor with operator norm bounded by $M$: $\|W h\| \le M \|h\|$ for all $h$. Suppose $\|h_A - h_B\| \le \epsilon$. Then

$$
\|W h_A - W h_B\| \le M \epsilon.
$$

For any prediction $\hat{z}_A = W h_A$ and $\hat{z}_B = W h_B$, we have

$$
\|\hat{z}_A - \hat{z}_B\| \le M \epsilon.
$$

By the triangle inequality,

$$
\|\hat{z}_A - \mu_A\| + \|\hat{z}_B - \mu_B\| \ge \|\mu_A - \mu_B\| - \|\hat{z}_A - \hat{z}_B\| \ge \delta - M \epsilon.
$$

Therefore, at least one of the two prediction errors is at least $(\delta - M\epsilon)/2$; squaring and using $\|\cdot\|^2 \ge (\|\cdot\|)^2$ yields the lower bound on the squared error:

$$
\max(\|\hat{z}_A - \mu_A\|^2, \|\hat{z}_B - \mu_B\|^2) \ge \frac{1}{4}(\delta - M\epsilon)^2.
$$

If $h_A = h_B$, then $M\epsilon = 0$, and the lower bound becomes $\delta^2/4$. $\square$

---

## References

[References would be listed here using standard Markdown citation format or a bibliography section. Since GitHub does not natively support BibTeX, you may want to use a manual list or a tool like `pandoc` to convert.]
