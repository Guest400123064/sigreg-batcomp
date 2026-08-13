# LeWorldModel (LeWM)

## Citation

- **Title:** LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels
- **Authors:** Lucas Maes* (Mila & Université de Montréal), Quentin Le Lidec* (NYU), Damien Scieur (Mila & Samsung SAIL), Yann LeCun (NYU), Randall Balestriero (Brown University). *Equal contribution. Correspondence: lucas.maes@mila.quebec
- **Venue / year:** arXiv preprint, March 2026 (v1: 2026-03-13; v3: 2026-06-03)
- **arXiv:** [2603.19312](https://arxiv.org/abs/2603.19312) ([PDF](https://arxiv.org/pdf/2603.19312), [HTML v3](https://arxiv.org/html/2603.19312v3))
- **Code:** [github.com/lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) (official; built on the authors' `stable-worldmodel` and `stable-pretraining` libraries). Pretrained checkpoints on Hugging Face (`quentinll/lewm-{pusht,cube,tworooms,reacher}`); baseline checkpoints (PLDM, DINO-WM, LeJEPA, IVL, IQL, GCBC) on Google Drive.
- Local copy: `paper.pdf` (arXiv v-current, 28 pages).

## Summary

### Problem setting

Fully offline, **reward-free** world-model learning from raw pixel trajectories `(o_1:T, a_1:T)` — no rewards, no task specification, no privileged state. The goal is a task-agnostic latent world model that can later be used for planning or analysis. The central challenge for end-to-end JEPAs is **representation collapse**: the encoder can map all inputs to a constant to trivially minimize the prediction loss. Prior work avoided collapse with EMA target encoders + stop-gradient (I-JEPA/V-JEPA), frozen pretrained encoders (DINO-WM), or a 7-term VICReg-derived objective (PLDM) with known instabilities. ([abs page](https://arxiv.org/abs/2603.19312), paper §1–2)

### Architecture (~15M parameters total)

- **Encoder:** ViT-Tiny (~5M params), patch size 14, 12 layers, 3 heads, hidden dim 192, input 224×224. The observation embedding is the final-layer **[CLS] token (192-dim) followed by a 1-layer MLP projector with BatchNorm** — the projector is required because the ViT's final LayerNorm would otherwise prevent the anti-collapse objective from being optimized effectively.
- **Predictor:** transformer (~10M params), 6 layers, 16 heads, 10% dropout, causal temporal masking. Actions enter via **zero-initialized Adaptive LayerNorm (AdaLN)** at each layer, so action conditioning ramps up progressively. It takes a history of N frame embeddings (N=3 on PushT/Cube, 1 on TwoRoom) and predicts the next embedding autoregressively. Followed by the same kind of projector as the encoder.
- Encoder and predictor are trained **jointly, end-to-end**: no stop-gradient, no EMA, no frozen or pretrained components. (paper §3.1, App. D)

### Training objective (two terms, one effective hyperparameter)

`L_LeWM = L_pred + λ · SIGReg(Z)`, with:

- `L_pred = || ẑ_{t+1} − z_{t+1} ||²₂` — teacher-forced MSE between predicted and encoded next-step embeddings.
- **SIGReg** (Sketched-Isotropic-Gaussian Regularizer, from Balestriero & LeCun 2025): forces the embedding distribution toward an isotropic Gaussian `N(0, I)`. Because high-dimensional normality testing is intractable, embeddings are projected onto **M = 1024 random unit-norm directions**; along each 1-D projection the **Epps–Pulley test statistic** is minimized — an integrated squared difference between the empirical characteristic function of the projected embeddings and that of a standard Gaussian (quadrature: trapezoid nodes in [0.2, 4]). By the **Cramér–Wold theorem**, matching all 1-D marginals implies matching the joint distribution, giving the weak-convergence guarantee `SIGReg(Z) → 0 ⇔ P_Z → N(0, I)`. (paper §3.1, App. A)

Defaults: λ = 0.1, M = 1024. The number of projections and integration knots have negligible impact, so **λ is the only effective hyperparameter** (vs. six for PLDM), tunable by bisection search in O(log n) rather than O(n⁶) grid search. (paper §3.1, §4.3)

### Stability results

- Training curves on PushT are smooth and monotonic: prediction loss decreases steadily; SIGReg drops sharply early then plateaus (latents quickly approach the Gaussian target). PLDM's seven-term objective is noisy and non-monotonic across components. (paper §4.3, Figs. 18–19)
- Ablations show robustness: insensitive to SIGReg internals (projections, knots); performance saturates quickly with embedding dimension; **works with a ResNet-18 encoder as well as ViT** — "largely agnostic to the choice of vision encoder". (paper §4.3, App. G)
- **Known limitation:** on the simplest environment (Two-Room, low-diversity low-intrinsic-dimension data), LeWM plans *worse* than PLDM. Authors' explanation: matching a high-dimensional isotropic-Gaussian prior is hard when data diversity is low, degrading latent structure. Probing (below) shows the Two-Room representation itself is as informative as PLDM's, so the planning gap comes from dynamics/planning, not representation content. (paper §4.2, App. F.2)

### Planning usage

- **Latent MPC:** encode initial observation and goal image; roll the predictor forward over horizon H; cost = terminal latent distance `||ẑ_H − z_g||²₂`; optimize action sequences with **CEM** (300 samples, 30 iterations on PushT / 10 elsewhere, top-30 elites); receding-horizon MPC, frame-skip 5 (horizon 5 latent steps = 25 env steps). World model weights frozen during planning. (paper §3.2, App. B/D)
- Results: **+18% success over PLDM on PushT** (and beats DINO-WM there even when DINO-WM gets extra proprioceptive input); competitive with DINO-WM overall across PushT, OGBench-Cube, TwoRoom, Reacher. **Planning up to 48× faster than DINO-WM** (single 192-dim token vs ~200× more tokens), full plan in under one second. Under fixed FLOPs, LeWM clearly beats DINO-WM on PushT and OGBench-Cube. (paper §4.1–4.2, Fig. 3)

### Scale / training-cost claims

- ~15M parameters total (ViT-Tiny encoder ~5M + predictor ~10M).
- "Trainable on a **single GPU in a few hours**" (abstract). Concretely: **10 epochs** per environment, batch size 128, sub-trajectories of 4 frames at 224×224, frame-skip 5. Datasets: 10k–20k episodes per environment (TwoRoom 10k × ~92 steps; PushT 20k × ~196 steps; Cube 10k × 200; Reacher 10k × 200). 10 epochs was empirically sufficient to match best reported performance. The paper does not state an exact GPU model or wall-clock number beyond "a few hours". (abstract, App. D–E)

## Analysis / probing methodology (§5, App. F–H)

This is the part most relevant to our project; LeWM includes an explicit "what is in the embedding?" analysis suite:

1. **Physical-quantity probing.** Train **linear and MLP (non-linear) probes** from the frozen latent embedding to ground-truth physical state variables; report MSE and Pearson r. Linear probes test whether information is *linearly accessible*; MLP probes test whether it is *present but entangled*. Variables: agent 2-D position (TwoRoom); agent position, block position, block angle (PushT); cube position, end-effector position, joint velocity, end-effector yaw, block quaternion/yaw (OGBench-Cube). Findings: LeWM ≥ PLDM on essentially all quantities and competitive with DINO-WM (whose DINOv2 encoder saw ~124M images, ~2 orders of magnitude more data); **all methods struggle with fine-grained rotation** (block quaternion/yaw) — rotational information is poorly encoded in compact latents regardless of training strategy. TwoRoom probing shows LeWM ≈ PLDM despite LeWM's worse planning there.
2. **A-posteriori decoder visualization.** A lightweight transformer decoder (cross-attention from the 192-dim [CLS] token to 196 learnable patch queries) is trained *after the fact* to reconstruct 224×224 images from single latents — reconstruction is never part of training. Decoded images recover global scene structure (proving state info is retained in the latent) but lose fine detail such as end-effector angle at longer horizons, consistent with probing. Early in training, decoded images show "slow features".
3. **t-SNE visualization** of PushT latents: latent space preserves spatial neighborhood structure of the environment.
4. **Violation-of-expectation (VoE) / surprise.** Borrowed from developmental psychology (as in V-JEPA-2-style intuitive-physics evals): surprise = **latent prediction error** (MSE between predicted and encoded next embedding) along a trajectory. Three trajectory types per environment: unperturbed; *visual* perturbation (abrupt object color change); *physical* perturbation (object teleported to a random position). LeWM shows a pronounced surprise spike for teleportation (paired t-test, p < 0.01 across all three environments) but a weaker, non-significant response to color changes — i.e., the latent dynamics encode physical continuity rather than surface appearance.
5. **Temporal latent path straightening.** Cosine similarity between consecutive latent velocity vectors rises over training — latent trajectories straighten as an *emergent* phenomenon (no explicit regularizer), and LeWM ends up straighter than PLDM despite PLDM's dedicated temporal-smoothness term. (App. H)

## Relevance

LeWM is the closest published analogue of what a LeJEPA-style time-series model does: end-to-end JEPA (encoder + action/conditioned predictor) trained with next-embedding MSE + SIGReg, with **no reconstruction, no EMA, no stop-gradient**. It matters for our probing study in several concrete ways:

- **Methodology template:** their §5 is essentially a ready-made probing battery for JEPA latents — linear vs. MLP probes (linearly-accessible vs. entangled information), a-posteriori decoding as an information-retention diagnostic, t-SNE for geometry, VoE/surprise as a functional test of what the *dynamics* (not just the encoder) captured, and latent path straightening as a dynamics-quality metric. All transfer directly to time-series embeddings (physical state variables → task-relevant signal properties).
- **What JEPA latents do and do not keep:** a single 192-dim vector per frame suffices to linearly decode positions (r ≈ 0.97–1.0) and even reconstruct the scene, yet **rotational/fine-grained dynamical quantities are poorly encoded by all JEPA variants** — a concrete hypothesis for what information a time-series JEPA embedding may drop.
- **Representation–planning dissociation:** on TwoRoom, LeWM's embeddings probe as well as PLDM's but plan worse — evidence that *probe accuracy does not imply downstream usability*, a caveat we must build into our own analysis (probes alone can overstate embedding quality).
- **SIGReg as a known inductive bias:** the isotropic-Gaussian constraint shapes what information can be stored (it hurt in a low-diversity, low-intrinsic-dimension environment). If our time-series setup uses SIGReg-like regularization, this paper documents both its stability benefits and its failure mode under low data diversity — directly relevant when probing what survives the Gaussianity pressure.
- **Follow-up literature signal:** later work ([arXiv 2607.26924](https://arxiv.org/html/2607.26924v2)) reports that marginal SIGReg can leave a LeWM representation *globally non-collapsed yet compressing/entangling task-relevant structure* under multi-task training — further motivation for probing rather than trusting anti-collapse metrics.
