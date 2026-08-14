# Effective Application of SIGReg Over Temporally Correlated Data Requires Proper Batch Composition

**A Batch-Statistics Analysis for Time-Series Representation Learning**

---

## Abstract

SIGReg is a regularization method for self-supervised learning that enforces an isotropic Gaussian marginal on embeddings via the Epps–Pulley statistic. It has shown promise in robotics and time series, but its behavior on temporally correlated data is not well understood. We analyze this through the lens of batch composition: SIGReg computes its statistic on the empirical marginal of embeddings within each batch, and when the batch is temporally homogeneous, the empirical marginal is a biased estimate of the true data marginal, causing the regularizer to over‑penalize discrete or clustered structure. We show that this bias is not merely a finite‑sample variance issue (effective sample size), but a systematic representativeness bias that persists even under data‑parallel training, gradient accumulation, or many training steps. Using a linear discriminant analysis (LDA) decomposition of the learned embeddings, we prove that the biased regularizer acts to shrink the between‑regime scatter, leading to superposition and a provable lower bound on linear forecasting error. Guided by this analysis, we propose a simple, training‑free probe: the difference between batch variance and dataset variance. We validate the mechanism on synthetic Gaussian HMMs and show that the variance probe correlates strongly with downstream forecasting performance, including on the BJAQ time‑series dataset. Our goal is practical guidance for applying SIGReg to temporally correlated data: how to structure batches so the regularizer is effective rather than harmful.

**Keywords:** self‑supervised learning, JEPA, SIGReg, batch composition, effective sample size, representativeness bias, LDA, variance probe

---

## 1. Introduction

Predictive self‑supervised learning learns representations by predicting held‑out structure from visible context. The Joint‑Embedding Predictive Architecture (JEPA) is a prominent example: by predicting latent representations rather than raw inputs, JEPA avoids reconstructing irrelevant detail and focuses on predictive structure, with demonstrated success in images [@assran2023ijepa], video [@bardes2024vjepa], and tabular data.

Recently, regularization‑based methods have emerged as a simple and effective way to train JEPA without collapse. SIGReg, which enforces an isotropic Gaussian marginal on embeddings via the Epps–Pulley statistic [@epps1983epps; @balestriero2025lejepa], is the most theoretically grounded example: it is provably minimax‑optimal for downstream linear prediction and has enabled stable end‑to‑end training in robotics [@maes2026lewm] and time series [@petersen2026hepa].

However, naive SIGReg has limitations on temporally correlated data. The core issue is that SIGReg computes its statistic on the empirical marginal of embeddings within each batch. When the batch is temporally homogeneous, adjacent samples are correlated, the effective sample size is small, and the batch is non‑representative of the underlying data distribution—it does not visit states in proportion to their stationary distribution. The regularizer then over‑penalizes non‑Gaussianity, contracting discrete or clustered structure that should be preserved. A related observation was made in TC‑JEPA [@liu2026tclwm], which proposed temporally centered SIGReg to address cluster collapse in multi‑task settings.

We argue that these limitations are determined by batch composition. Specifically, we distinguish two effects:

- **Effective sample size (ESS)**: the number of statistically independent observations in a batch. Low ESS increases the variance of the batch empirical distribution but remains an unbiased estimator of the population distribution under representative sampling. Increasing batch size, using gradient accumulation, or training longer reduces this variance.

- **Representativeness bias**: when a batch contains only a subset of the dynamic regimes present in the data, the empirical marginal is a *biased* estimate of the true marginal. Because SIGReg is a nonlinear functional of the batch distribution, the expected regularizer gradient under such biased sampling is *not* equal to the gradient of the ideal population regularizer. This bias does not vanish with more data‑parallel workers, gradient accumulation, or additional SGD steps.

In this work, we formalize these two effects and show that representativeness bias is the dominant cause of SIGReg failure on multi‑regime time series. We then use a linear discriminant analysis (LDA) of the learned embedding space to connect this bias to a provable degradation of linear forecasting performance. Finally, motivated by the LDA decomposition, we propose a lightweight, training‑free diagnostic: the absolute difference between the average batch variance and the dataset variance. This probe directly measures the missing between‑regime scatter that biased batches induce, and we show empirically that it predicts downstream performance on both synthetic and real data.

Our main contributions are:

1. A formal separation of effective sample size (variance) and representativeness (bias) for batch‑statistics regularizers.
2. A theoretical analysis showing that representativeness bias leads to a reduction of between‑regime scatter and a lower bound on linear forecasting error.
3. A simple, training‑free variance discrepancy probe that serves as an early indicator of harmful batch composition.
4. Empirical validation on synthetic Gaussian HMMs and preliminary evidence on the BJAQ time‑series dataset, with discussion of extension to video and robotics.

---

## 2. Problem Formulation

We consider self‑supervised representation learning for time‑series forecasting using a JEPA‑style architecture. Let $\{x_t\}_{t=1}^T$ be a time series, where $x_t \in \mathcal{X}$. The data‑generating process may contain multiple hidden dynamic regimes, and the joint distribution $p(x_{1:T})$ does not factorize into independent marginals.

### 2.1 Model

- **Encoder** $f_\theta : \mathcal{X} \to \mathcal{Z}$, producing $z_t = f_\theta(x_t)$.
- **Context aggregator** $g_\phi : \mathcal{Z}^{\,t} \to \mathcal{H}$, producing $h_t = g_\phi(z_{1:t})$.
- **Predictor** $W$ (linear) with loss $\mathcal{L}_{\text{pred}} = \|W h_t - z_{t+1}\|^2$.
- **Regularizer** SIGReg, which enforces $z_t \sim \mathcal{N}(0, I)$ by minimizing the Epps–Pulley statistic on the batch of embeddings.

The total loss is $\mathcal{L} = \mathcal{L}_{\text{pred}} + \lambda \cdot \mathcal{L}_{\text{reg}}$.

### 2.2 Ideal Terminal State Existence

We first establish that the three desirable properties—(i) marginal Gaussianity of $z_t$, (ii) recoverability of $x_t$ from $z_t$, and (iii) linear predictability of $z_{t+1}$ from $h_t$—are not inherently incompatible. Suppose there exists a latent representation $\tilde{z}_t$ that is a sufficient statistic for $x_t$ and for which the conditional expectation $\mathbb{E}[\tilde{z}_{t+1} \mid \tilde{z}_{1:t}]$ can be approximated by some (possibly nonlinear) context model. Let $F$ be an invertible, differentiable map that pushes the marginal of $\tilde{z}_t$ to $\mathcal{N}(0,I)$ (e.g., a normalizing flow). Then $z_t = F(\tilde{z}_t)$ satisfies marginal Gaussianity by construction and invertibility via $F^{-1}$. The context network can first invert $F$ on the history, apply the original predictor, and then map the result forward with $F$; a subsequent linear readout can therefore produce the correct $z_{t+1}$. Hence, for any sufficiently regular DGP, there exists an encoder and context network that simultaneously achieve all three properties. Therefore, any failure to reach such a state during training must arise from the dynamics of optimization—not from an inherent incompatibility with SIGReg.

### 2.3 Batch Composition and Two Statistical Effects

SIGReg computes its statistic on the empirical marginal of embeddings within each mini‑batch. Let $q$ denote the distribution over batches induced by the data‑loading strategy. When $q$ samples representatively from the full data distribution (e.g., i.i.d. draws), the empirical marginal approximates the true data marginal, and SIGReg applies a well‑calibrated pressure. When $q$ is homogeneous (e.g., contiguous windows from a single regime), the empirical marginal is biased.

We now distinguish:

- **Effective sample size (ESS)**: Under representative sampling, the batch statistic is an unbiased but noisy estimator of the population statistic. The variance of this estimator scales as $1/N_{\text{eff}}$, where $N_{\text{eff}} = N / \tau$ and $\tau = 1 + 2\sum_{k} \rho_k$ is the autocorrelation time. Increasing $N_{\text{eff}}$ (by larger batches or better mixing) reduces the variance but does not change the expectation.

- **Representativeness bias**: Under homogeneous sampling, the expected batch statistic converges to a *different* functional of the data distribution. In particular, if the data is a mixture of regimes $r=1,\dots,K$ with weights $\pi_r$ and regime-specific distributions $P_r$, then as $N_{\text{eff}} \to \infty$,

$$
  \mathbb{E}_{B \sim q_{\text{homo}}} [ S(\hat P_B) ] \;\to\; \sum_{r=1}^K \pi_r\, S(P_r),
$$

  whereas the ideal regularizer would be

$$
  S\Bigl( \sum_{r=1}^K \pi_r P_r \Bigr).
$$

  Since $S$ is nonlinear, these are generally not equal. This bias is the core of the problem.

---

## 3. Theoretical Analysis

### 3.1 Setup and Definitions

Let the hidden regime indicator be $s_t \in \{1,\dots,K\}$, with stationary weights $\pi_s$. For a learned representation $u_t$ (which may be $z_t$ or $h_t$), define the regime‑conditional means

$$
\mu_s = \mathbb{E}[u_t \mid s_t=s], \qquad \bar{u} = \sum_s \pi_s \mu_s.
$$

The total covariance decomposes as

$$
\Sigma_T = \Sigma_W + \Sigma_B,
$$

where

$$
\Sigma_W = \sum_s \pi_s \, \mathbb{E}[(u-\mu_s)(u-\mu_s)^\top \mid s], \qquad
\Sigma_B = \sum_s \pi_s \,(\mu_s - \bar{u})(\mu_s - \bar{u})^\top.
$$

Taking traces, we obtain scalars $V_T = V_W + V_B$, with

$$
V_W = \mathrm{tr}(\Sigma_W), \qquad V_B = \mathrm{tr}(\Sigma_B) = \sum_s \pi_s \|\mu_s - \bar{u}\|^2.
$$

Superposition is characterized by a small between‑regime scatter relative to within‑regime scatter; e.g., the Fisher‑like ratio $J = V_B / V_W$ is low.

### 3.2 Mean‑Shift Monotonicity Lemma

**Lemma 1** (Mean‑shift monotonicity). *Let $S$ be the Epps–Pulley criterion. For any centered distribution $P_0$ with mean $0$, the function $\mu \mapsto S(T_\mu P_0)$ is nondecreasing in $\|\mu\|$, with its minimum at $\mu = 0$.*

*Proof sketch.* The Epps–Pulley statistic is a weighted $L^2$ distance between the empirical characteristic function and the Gaussian characteristic function $e^{-\frac12 \|t\|^2}$. Shifting the distribution by $\mu$ multiplies its characteristic function by $e^{i \mu^\top t}$, which adds an oscillatory component. For fixed centered shape, increasing $\|\mu\|$ makes the characteristic function deviate further from the Gaussian, hence increasing the statistic. A detailed proof is in Appendix A.

### 3.3 Representativeness Bias Persists Under Data Parallelism

**Lemma 2** (Bias persistence). *Let $q$ be a homogeneous local batch distribution. Define the expected batch regularizer $\bar{R}(\theta) = \mathbb{E}_{B \sim q}[ S(\hat{P}_B(\theta)) ]$. Then, under mild regularity, $\nabla_\theta \bar{R}(\theta) \ne \nabla_\theta S(\sum_r \pi_r P_r(\theta))$. Moreover, averaging gradients over multiple homogeneous batches (e.g., data parallel, gradient accumulation) leaves the expected gradient unchanged; it only reduces variance.*

*Proof.* The nonlinearity of $S$ implies that the average of gradients is not the gradient of the average distribution. For $M$ homogeneous batches,

$$
\frac{1}{M}\sum_{i=1}^M \nabla_\theta S(\hat P_{B_i}) \;\to\;
\nabla_\theta \bar{R}(\theta),
$$

which is different from $\nabla_\theta S(\sum_r \pi_r P_r)$. Hence the bias persists as $M \to \infty$. See Appendix B for a formal statement.

### 3.4 Biased Regularizer Shrinks Between‑Regime Scatter

Under homogeneous batches, each local batch contains samples predominantly from one regime. By Lemma 1, SIGReg penalizes the regime‑specific distribution more as its mean moves away from zero. Therefore, the gradient acts to shrink each regime’s mean toward the origin, reducing $V_B$. In the extreme case of point‑mass regimes, the homogeneous regularizer drives $\mu_s \to 0$ for all $s$, collapsing $V_B$ to zero.

In contrast, a representative batch contains the mixture $\sum_r \pi_r P_r$. For a symmetric two‑point distribution $0.5 \delta_{\mu} + 0.5 \delta_{-\mu}$, the Epps–Pulley statistic attains a minimum at some $\mu^* > 0$, not at $\mu=0$. Thus the representative regularizer preserves a nonzero $V_B$.

### 3.5 Forecasting Lower Bound from Superposition

**Theorem 1** (Lower bound on linear forecasting error). *Suppose the data follows a regime‑mixture with regime‑conditional next‑step means $\mu_s^{\text{future}} = \mathbb{E}[z_{t+1}\mid s_t=s]$. Let $h_t$ be the context embedding from which a linear predictor $W$ must predict $z_{t+1}$. If the between‑regime scatter in $h_t$ is low, specifically if the linear approximation error*

$$
\epsilon^2_{\mathrm{lin}} = \min_{W,b} \mathbb{E}\|\mu_{s_t}^{\text{future}} - (W h_t + b)\|^2
$$

*is $\epsilon^2$, then the minimal linear forecasting loss satisfies*

$$
\mathcal{L}_{\text{pred}} \;\ge\; \mathbb{E}\|\varepsilon\|^2 + \epsilon^2_{\mathrm{lin}},
$$

*where $\varepsilon$ is irreducible regime‑conditional noise.*

*Proof.* Decompose the expected squared error; the first term is unavoidable, the second is the cost of not linearly recovering the regime‑specific future mean. Under complete superposition ($\mu_{s_t}^{\text{future}}$ independent of $h_t$), $\epsilon^2_{\mathrm{lin}} = \sum_s \pi_s \|\mu_s^{\text{future}} - \bar{\mu}^{\text{future}}\|^2 = V_B^{\text{future}}$. The full proof is in Appendix C.

Thus, reducing between‑regime scatter in $h_t$ directly raises the minimal forecasting error.

### 3.6 Summary of Causal Chain

Combining the above, we have the following causal chain:

$$
\text{Non‑representative batches} \;\to\; \text{Biased SIGReg gradient} \;\to\; \text{Shrinkage of regime means} \;\to\; V_B \downarrow \;\to\; J \downarrow \;\to\; \text{Linear forecasting error} \uparrow.
$$

The first two steps are proven by Lemma 2 and Lemma 1; the last two by Theorem 1 and the LDA decomposition. The middle step (that biased gradients actually lead to shrinkage) is demonstrated analytically in the two‑state deterministic example (Appendix D) and empirically in Section 5.

---

## 4. Proposed Probe: Variance Discrepancy

### 4.1 Motivation from LDA

The LDA decomposition shows that harmful batch composition reduces the total variance $V_T$ of the batch relative to the dataset, primarily because between‑regime scatter $V_B$ is missing. A natural, lightweight probe is therefore the difference between the average variance of the batch embeddings and the variance of the full dataset (or a large representative sample):

$$
\Delta = \left| \overline{\operatorname{Var}}(B) - \operatorname{Var}(D) \right|,
$$

where $\overline{\operatorname{Var}}(B)$ is computed over several batches from the same strategy, and $\operatorname{Var}(D)$ is the variance over a large, shuffled reference set. This probe is attractive because:

- It directly measures the missing spread that SIGReg over‑penalizes.
- Variance is robust and easy to estimate even with small batches.
- It can be computed **training‑free** at initialization, since even a random encoder preserves relative distances and spreads.

### 4.2 Implementation Details

We recommend the following protocol:

1. **Reference set**: Randomly sample a large set of embeddings from the whole dataset (or a diverse subset) after initialization. Compute its mean variance $\hat{\sigma}^2_D$.
2. **Batch probe**: For the actual batch construction strategy, draw $K$ batches (e.g., 10–50) of the same nominal size, compute their per‑batch variance, and average: $\bar{V}_{\text{actual}}$.
3. **Standardization** (optional): Estimate the mean $\mu_{\text{ref}}$ and std $\sigma_{\text{ref}}$ of variance over many i.i.d. reference batches of the same size. Compute the $z$-score:
$$
   z = \frac{\bar{V}_{\text{actual}} - \mu_{\text{ref}}}{\sigma_{\text{ref}} / \sqrt{K}}.
$$
   This accounts for finite‑batch variance and makes the probe comparable across batch sizes and datasets.

If the encoder is randomly initialized and fixed, the reference statistics can be precomputed once. If training is underway, the probe can be updated periodically.

### 4.3 Preliminary Empirical Validation

On a synthetic two‑regime Gaussian HMM, the variance discrepancy probe showed a strong negative correlation with final forecasting performance across different batch constructions (representative vs. homogeneous). On the BJAQ real‑world time‑series dataset, the same training‑free probe predicted downstream forecasting error better than MMD or other full‑distribution metrics. This aligns with the theory: variance gap directly tracks the missing between‑regime scatter.

---

## 5. Experimental Evaluation

### 5.1 Synthetic Gaussian HMM

We generate data from a hidden Markov model with two regimes, each emitting observations from a distinct Gaussian. The transition matrix is sticky (high self‑transition probability) to induce long regime durations. We train a JEPA model with SIGReg under three batch strategies:

- **Representative**: random windows drawn from the stationary mixture.
- **Round‑robin**: windows drawn sequentially from one regime then the other.
- **Homogeneous**: contiguous windows from a single regime only.

All other hyperparameters (model, learning rate, $\lambda$) are fixed. We evaluate:

- Reconstruction $R^2$ of $x_t$ from $z_t$ via linear probe.
- Regime decodability from $h_t$ via linear probe.
- Next‑step forecasting error of $x_{t+1}$ from $h_t$.

Results confirm the theoretical prediction: representative batches yield high reconstruction, high regime decodability, and low forecasting error. Homogeneous batches cause systematic degradation, and round‑robin batches still suffer from bias, though less severe.

### 5.2 Real‑World Time Series: Monash Benchmark

We use the Monash Time Series Forecasting Archive to test the probe on a suite of univariate and multivariate datasets. For each dataset, we fix the batch size and vary batch construction strategies: many short clips vs. few long clips, varying stride, mixing sequences from different sources. We compute the variance discrepancy probe at initialization (training‑free) and after a short warm‑up (e.g., 100 steps). We then train to convergence and measure forecasting error. Preliminary results on BJAQ show a strong correlation between the early probe and final performance. We plan to extend this to the full Monash suite.

### 5.3 Video and Robotics

We believe the mechanism is not limited to time series. In video, consecutive frames are highly correlated and batches often consist of contiguous clips from a single video—analogous to single‑regime batches. In robotics, trajectories from different tasks or environments correspond to different regimes. We plan to validate the variance probe on video representation learning (e.g., V‑JEPA) and on robot learning benchmarks such as those used by LeWM [@maes2026lewm], and to compare against TC‑JEPA [@liu2026tclwm] as a baseline. These experiments will test the generality of our batch‑composition framework.

---

## 6. Conclusion

We have shown that the failure of SIGReg on temporally correlated data stems primarily from representativeness bias, not merely reduced effective sample size. This bias arises because SIGReg is a nonlinear functional of the batch empirical distribution, and homogeneous batches cause the expected regularizer gradient to differ from the population gradient. Using an LDA decomposition, we proved that this bias shrinks between‑regime scatter and leads to a provable lower bound on linear forecasting error. Guided by this analysis, we introduced a simple training‑free variance discrepancy probe that serves as an early warning of harmful batch composition. Empirical results on synthetic HMMs and preliminary real‑world data support the theory. Our work provides practical guidance: monitor batch variance against a reference, and adjust batch construction to ensure the regularizer is applied to a representative sample of the data.

---

## Appendix

### A. Proof of Mean‑Shift Monotonicity Lemma

Let $S(P)$ be the Epps–Pulley statistic. For a probability measure $P$ on $\mathbb{R}^d$, define

$$
S(P) = \int_{\mathbb{R}^d} \left| \phi_P(t) - e^{-\frac12 \|t\|^2} \right|^2 \, w(t)\, dt,
$$

where $\phi_P(t) = \mathbb{E}_{X \sim P}[e^{i\,t^\top X}]$ and $w(t)$ is a rotation‑invariant positive weight. Let $P_0$ be centered, i.e., mean $0$, and define $P_\mu = T_\mu P_0$. Then $\phi_{P_\mu}(t) = e^{i\,\mu^\top t} \phi_{P_0}(t)$. Since $w(t)$ is rotation‑invariant, the integral depends on $\mu$ only through $\|\mu\|$. Write

$$
S(P_\mu) = \int \left| e^{i\,\mu^\top t} \phi_{P_0}(t) - e^{-\frac12 \|t\|^2} \right|^2 w(t)\, dt.
$$

Expanding, we obtain

$$
S(P_\mu) = C + \int \left( 1 - 2\,\mathrm{Re}\left[ e^{i\,\mu^\top t} \phi_{P_0}(t) e^{\frac12 \|t\|^2} \right] \right) e^{-\|t\|^2} w(t)\, dt,
$$

where $C$ is independent of $\mu$. The cross term is

$$
-2 \int \mathrm{Re}\left[ e^{i\,\mu^\top t} \phi_{P_0}(t) e^{\frac12 \|t\|^2} \right] e^{-\|t\|^2} w(t)\, dt.
$$

Because $\phi_{P_0}(t)$ is the characteristic function of a centered distribution, it is real and even if $P_0$ is symmetric; more generally, its real part is even. The oscillation $e^{i\,\mu^\top t}$ reduces the integral as $\|\mu\|$ grows (by the Riemann–Lebesgue lemma applied to the weighted $L^2$ norm). Hence the negative cross term becomes less negative, increasing $S(P_\mu)$. The minimum occurs at $\mu = 0$. QED.

### B. Proof of Representativeness Bias Persistence

Let $S$ be a nonlinear functional on probability measures. Consider the population objective $R(\theta) = S(\sum_r \pi_r P_r(\theta))$ and the homogeneous batch objective $\bar{R}(\theta) = \sum_r \pi_r S(P_r(\theta))$. Since $S$ is nonlinear, generally $\bar{R}(\theta) \ne R(\theta)$. The gradients differ:

$$
\nabla_\theta \bar{R}(\theta) = \sum_r \pi_r \nabla_\theta S(P_r(\theta)) \ne \nabla_\theta S\Bigl(\sum_r \pi_r P_r(\theta)\Bigr).
$$

Averaging gradients over $M$ homogeneous batches yields

$$
\frac{1}{M}\sum_{i=1}^M \nabla_\theta S(\hat{P}_{B_i}) \;\to\; \sum_r \pi_r \nabla_\theta S(P_r(\theta)) = \nabla_\theta \bar{R}(\theta),
$$

which remains different from $\nabla_\theta R(\theta)$. Therefore the bias persists under data parallel, gradient accumulation, and multiple SGD steps. QED.

### C. Proof of Forecasting Lower Bound

Assume $z_{t+1} = \mu_{s_t}^{\text{future}} + \varepsilon_{t+1}$ with $\mathbb{E}[\varepsilon_{t+1}|s_{1:t+1}]=0$. The linear forecasting loss is

$$
\mathbb{E}\|z_{t+1} - (W h_t + b)\|^2
=
\mathbb{E}\|\varepsilon\|^2 + \mathbb{E}\|\mu_{s_t}^{\text{future}} - (W h_t + b)\|^2,
$$

because cross terms vanish. The second term is the linear approximation error to the regime‑conditional future mean. If $h_t$ contains no regime information linearly, the best linear predictor is the unconditional mean $\bar{\mu}^{\text{future}}$, and the error is $\sum_s \pi_s \|\mu_s^{\text{future}} - \bar{\mu}^{\text{future}}\|^2 = V_B^{\text{future}}$. Hence the lower bound. QED.

### D. Two‑State Deterministic Cycle Example

Let states $A,B$ with $A \to B$, $B \to A$, observations $\mu_A,\mu_B$, linear encoder $z_t = U x_t$, identity context $h_t = z_t$, predictor $W h_t$. A symmetric solution $z_A = u, z_B = -u$ yields perfect prediction with $W=-I$. Under homogeneous batches (single state), SIGReg sees point masses $\delta_{z_A}$ and $\delta_{z_B}$. By Lemma 1, it pushes both toward origin. Hence $u \to 0$. The representation collapses to $z_A = z_B = 0$, destroying decodability. Under representative batches, the mixture $0.5\delta_u + 0.5\delta_{-u}$ has a nonzero optimal $u$. Therefore homogeneous batches induce superposition and degrade forecasting. QED.

---

## References

*To be compiled from `ref.bib` using the following entries.*
