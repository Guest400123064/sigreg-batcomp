# V-JEPA (and V-JEPA 2, with I-JEPA lineage)

Notes on the video Joint-Embedding Predictive Architecture line of work from Meta FAIR /
NYU / Inria: I-JEPA (images, 2023) -> V-JEPA (video, 2024) -> V-JEPA 2 (scaled video +
planning, 2025). These are the canonical references for feature-space (latent) masked
prediction as a stand-alone self-supervised objective, and for the *attentive probe* frozen
evaluation protocol.

Files in this folder:

- `ijepa.pdf`  — I-JEPA, arXiv:2301.08243 (17 pages)
- `vjepa.pdf`  — V-JEPA, arXiv:2404.08471 (23 pages)
- `vjepa2.pdf` — V-JEPA 2, arXiv:2506.09985 (48 pages)

---

## 1. V-JEPA — "Revisiting Feature Prediction for Learning Visual Representations from Video"

### Citation

- Authors: Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat,
  Yann LeCun, Mahmoud Assran, Nicolas Ballas (Assran & Ballas joint last authors).
  Note: first author is Bardes, not Assran.
- Venue: Transactions on Machine Learning Research (TMLR), 2024.
  OpenReview: https://openreview.net/forum?id=QaCCuDfBk2
- arXiv: https://arxiv.org/abs/2404.08471 (v1 Feb 2024; HTML of v1:
  https://arxiv.org/html/2404.08471v1)
- Code & checkpoints: https://github.com/facebookresearch/jepa
  ("PyTorch code and models for V-JEPA self-supervised learning from video")

### Objective / loss

- Pure **feature prediction**: predict the representation of a masked spatio-temporal
  region `y` of a video from the representation of the visible context `x`, conditioned on
  positional mask tokens `Δ_y` specifying where `y` is.
- Loss: L1 regression
  `min ||P_φ(E_θ(x), Δ_y) − sg(Ē_θ(y))||₁`
  where `sg` is stop-gradient and `Ē_θ` (the y-encoder / target encoder) is an
  **exponential moving average** of the context encoder `E_θ`. The EMA target encoder +
  stop-gradient + predictor is the BYOL-style collapse-prevention strategy; no negative
  examples, no reconstruction, no pretrained image encoders, no text, no human labels.
  ([abstract](https://arxiv.org/abs/2404.08471), [§3.1](https://arxiv.org/html/2404.08471v1))
- The L1 choice (vs. I-JEPA's L2) was found more stable. Theoretical motivation: under L1,
  the optimal predictor outputs the conditional *median* of the target representation given
  `E_θ(x)`, so the encoder-side gradient minimizes the conditional median absolute
  deviation (MAD) of the target — i.e. the encoder is pushed to make the target as
  predictable (low-deviation) as possible given the context, which means capturing as much
  relevant information about the video as possible. The EMA keeps the target network
  slower-moving than the predictor so the predictor stays near-optimal. (§3.1)

### Masking / prediction task (§3.2, §4.4)

- Video clip: 16 frames, frame-skip 4 (~3 s), 224 px (384 for ViT-H/16-384).
  Patchified into tubelets of 2 frames × 16×16 px; masking = dropping tokens.
- **Multi-block masking**: sample several possibly-overlapping spatially contiguous blocks
  (aspect ratio in (0.75, 1.5)) and **repeat each spatial block across the full temporal
  extent** of the clip (limits information leakage from spatio-temporal redundancy);
  `x` is the complement. Two mask types mixed per batch:
  - *short-range*: union of 8 blocks covering 15% of each frame;
  - *long-range*: union of 2 blocks covering 70% of each frame.
  Average masking ratio ~90%.
- Ablation (Table 4, ViT-L/16, attentive probe): multi-block > causal multi-block (context
  restricted to first p frames) > random-tube masking (90% random tubes gives
  "features of low-semantic quality" under the feature-prediction objective). Large
  contiguous blocks are what makes the task hard enough to force semantic/temporal
  representations.

### Architecture (§3.3)

- Encoder: standard ViT (ViT-L/16 300M, ViT-H/16 630M params) processing tubelet tokens;
  masking applied at the *input* of the x-encoder (efficiency) and at the *output* of the
  y-encoder (contextualized targets, as in data2vec).
- Predictor: narrow transformer, 12 blocks, embedding dim 384; takes encoder outputs +
  learnable mask tokens with positional embeddings for the masked positions.
- Pretraining data: **VideoMix2M** = HowTo100M + Kinetics-400/600/700 + SSv2,
  decontaminated vs. the corresponding validation sets, ~2M videos. Batch 3072, 90K
  iterations for ablations.

### Frozen evaluation: attentive probing (§4.3, §11.1)

This is the paper's central evaluation methodology and its key methodological export:

- Because the JEPA loss is unnormalized, there is no reason for the encoder output to be
  linearly separable; hence **average pooling + linear probe undersells the representation**.
- The **attentive probe**: freeze the backbone; train on top a cross-attention layer with a
  learnable query token that pools the full token feature map, add the result back to the
  query token (residual), then a 2-layer MLP (single GeLU) + LayerNorm + linear classifier.
- Attentive pooling beats average pooling by **+17.3 pts on K400 and +16.1 pts on SSv2**
  (Table 3). It also improves baselines (Appendix 12), so all frozen comparisons in the
  paper use it (including re-probing DINOv2, OpenCLIP, I-JEPA — e.g. DINOv2-g/14 K400
  improves from a previously reported 78.4% linear-probe to 83.4% with an attentive probe).
- Label efficiency (Table 7): gap between V-JEPA and pixel-prediction baselines *grows* as
  probe training labels shrink (5%/10%/50% of train set, 3 splits each).

### Key results

- ViT-H/16-384 frozen backbone + attentive probe: **81.9% K400, 72.2% SSv2, 77.9%
  ImageNet-1K** ([abstract](https://arxiv.org/abs/2404.08471)).
- Frozen evaluation vs. pixel-prediction video models (VideoMAE, OmniMAE, Hiera; Table 5):
  V-JEPA wins on all tasks at ViT-L scale while seeing far fewer pretraining samples
  (270M samples / 90K iters vs. up to 2400M / 1170K for OmniMAE) — ~2× wallclock speedup
  over large pixel-prediction models (Fig. 5). Fine-tuning is competitive with pixel
  prediction.
- vs. image models (DINOv2, OpenCLIP, I-JEPA): +21 pts on SSv2 (motion), while slightly
  behind on appearance-heavy ImageNet — motion information is what video pretraining buys.

### Analysis: what do the representations capture? (§6)

- The paper's direct "what's in the embedding" analysis is a **decoding probe**: freeze
  encoder + predictor, train a **conditional diffusion decoder** to map the predictor's
  predicted features for the *masked* regions back to pixels. The decoder never sees the
  unmasked context, so whatever is common across decoder samples is information actually
  encoded in the predicted representation.
- Finding: predictions are spatially and temporally coherent with the masked-out content
  and capture consistent motion over time — i.e. the latent predictions encode
  scene/object layout and dynamics, not exact pixel detail (V-JEPA is not generative).
- Indirect evidence of content: strong SSv2 (motion) vs. slightly weaker ImageNet
  (appearance) under identical probing shows the embedding prioritizes predictable
  dynamics over static appearance minutiae; feature-space prediction consistently beats
  pixel-space prediction under the same frozen protocol (Table 1).

---

## 2. V-JEPA 2 — "Self-Supervised Video Models Enable Understanding, Prediction and Planning"

### Citation

- Authors: Mahmoud (Mido) Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell
  Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha,
  Artem Zholus, … Yann LeCun, Michael Rabbat, Nicolas Ballas (FAIR at Meta; Mila).
- arXiv preprint, June 2025: https://arxiv.org/abs/2506.09985
  (HTML: https://arxiv.org/html/2506.09985v1)
- Code, checkpoints (ViT-L/H/g/g-384), trained attentive probes, V-JEPA 2-AC:
  https://github.com/facebookresearch/vjepa2 (also on HuggingFace, e.g.
  `facebook/vjepa2-vitg-fpc64-256`).

### Objective / architecture (§2)

- Same mask-denoising feature-prediction loss as V-JEPA:
  `min ||P_φ(Δ_y, E_θ(x)) − sg(E_θ̄(y))||₁`, L1, EMA target encoder, loss only on masked
  patches. Changes:
  - **3D-RoPE** (rotary position embeddings split across time/height/width) instead of
    absolute sincos — stabilizes training at 1B scale.
  - Encoder scaled ViT-L (300M) -> **ViT-g/16 (~1B params)**; predictor ~ViT-S-sized.
  - Data scaled 2M -> **22M samples (VideoMix22M)**: SSv2 + Kinetics 400/600/700 +
    HowTo100M + **YT-Temporal-1B** (cluster-retrieval-curated against a target
    distribution of Kinetics/SSv2/COIN/EpicKitchens train sets) + ImageNet images
    (temporally duplicated as static 16-frame clips). >1M hours of video.
  - Warmup-constant-decay LR schedule, 252K iterations; **progressive resolution**:
    warmup+constant phases at 16 frames / 256², cooldown (12K iters) at up to 64 frames /
    384² — 8.4× GPU-time saving vs. full-resolution training; 64-frame training helps even
    when evaluating on 16 frames (+0.7 avg).
- Cumulative scaling gains on a 6-task frozen-probe average (SSv2, Diving-48, Jester,
  Kinetics, COIN, ImageNet): +1.0 (data 2M->22M), +1.5 (300M->1B), +0.8 (90K->252K iters),
  up to +4.0 total with resolution/duration scaling (Fig. 3).

### Evaluation methodology: probing what the embedding contains (§5, §6, §7)

- **Attentive probe, scaled up**: frozen encoder, train a **4-layer** attentive probe —
  four transformer blocks, the last replacing self-attention with cross-attention with a
  learnable query token; logits averaged over multiple sampled clips at inference (§5).
  Trained probe checkpoints are released in the repo (SSv2, Diving48, EK100).
- Tasks deliberately split into **motion understanding** (SSv2, Diving-48, Jester) vs.
  **appearance understanding** (K400, COIN, ImageNet) to characterize *what kind* of
  information the representation encodes — motion tasks need multi-frame information;
  appearance tasks are (mostly) solvable from a single frame.
- Results: SOTA motion understanding — **77.3 top-1 SSv2**, 90.2 Diving-48 (attentive
  probe, ViT-g/16-384); competitive appearance understanding.
- **Prediction probe**: Epic-Kitchens-100 action anticipation with an attentive probe,
  **39.7 recall@5** — SOTA, +44% relative over previous best (PlausiVL 27.6,
  per repo table). Evidence the embedding encodes near-future dynamics, not just the
  current state.
- **LLM alignment as a probe**: aligning the frozen-ish encoder with an 8B LLM gives SOTA
  video-QA at that scale (PerceptionTest 84.0, TempCompass 76.9, MVP 44.5, TemporalBench
  36.7, TOMATO 40.3), showing language-free pretraining retains temporally-grounded,
  physically meaningful content that is extractable by a language model.
- **Planning as the ultimate probe of state information** (§3–4): freeze encoder, train
  **V-JEPA 2-AC**, a ~300M block-causal transformer predictor, on <62 h of *unlabeled*
  Droid robot video: teacher-forcing L1 next-frame feature prediction conditioned on
  end-effector state + action (7-DoF delta), plus a 2-step rollout loss. Planning = CEM
  minimization of `||P(â_{1:T}; s_k, z_k) − z_goal||₁` in latent space, receding horizon.
  Zero-shot Franka pick-and-place in two unseen labs (e.g. 80% cup / 50% box
  pick-and-place vs. 10%/10% for Octo and 0%/0% for Cosmos; repo table). The paper is
  explicit that "the capabilities of a representation-space world model … are inherently
  limited by the state information encoded in the learned representation space" — planning
  success *is* evidence about embedding content. They also visualize the latent **energy
  landscape** over actions (Fig. 9): smooth, locally convex, minimum near the ground-truth
  action.

---

## 3. Lineage: I-JEPA — "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"

### Citation

- Authors: Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent,
  Michael Rabbat, Yann LeCun, Nicolas Ballas.
- Venue: CVPR 2023, pp. 15619–15629. arXiv: https://arxiv.org/abs/2301.08243
- (No arXiv HTML version; facts below extracted from the PDF in this folder.)

### Key ideas (the template everything above follows)

- Non-generative SSL from images: predict the representations of several target blocks in
  an image from a single context block — in representation space, so no hand-crafted
  data-augmentation invariances and no pixel detail.
- **Masking strategy is the core design choice** (abstract): (a) target blocks must be
  large enough to be semantic, (b) the context must be informative and spatially
  distributed. Concretely: sample M=4 target blocks, scale (0.15, 0.2), aspect ratio
  (0.75, 1.5); sample 1 context block, scale (0.85, 1.0), unit aspect ratio; remove
  target/context overlap. Targets are obtained by masking the *output* of the target
  encoder, "crucial to ensure target representations of a high semantic level".
- Architecture: ViT context encoder (visible patches only), ViT target encoder updated by
  **EMA** (momentum 0.996 ramped up), narrow ViT predictor conditioned on positional mask
  tokens. Loss = average **L2** distance between predicted and target patch
  representations (V-JEPA later switched to L1).
- Evaluation/analysis:
  - Linear probing on ImageNet and **1%-label semi-supervised** evaluation as the main
    sensitivity benchmark for masking ablations (Tables 6, 8–12: block size, context size,
    number of targets, predictor depth).
  - Low-level transfer linear probes: CLEVR object counting and depth prediction —
    I-JEPA matches/beats MAE/data2vec and beats augmentation-based methods (DINO, iBOT),
    showing latent prediction keeps low-level spatial information too.
  - Scalability: ViT-H/14 trained on ImageNet in <72 h on 16 A100s.
  - Predictor-content visualization: decode predictor outputs (conditioned only on
    positional mask tokens) with a generative model (R-CDM decoder) — the ancestor of
    V-JEPA's diffusion-decoder analysis; shows position-aware, semantic predictions.

---

## Relevance

Why this matters for probing LeJEPA-style time-series embeddings:

- **Methodological template for latent-prediction objectives.** V-JEPA/I-JEPA define the
  exact design axes a LeJEPA time-series model inherits: masking strategy (large
  contiguous blocks vs. random masking; in video, blocks repeated over time to prevent
  leakage through temporal redundancy), feature-space vs. pixel/sample-space targets,
  EMA target encoder + stop-gradient as collapse prevention, and a narrow predictor
  conditioned on positional mask tokens. The V-JEPA ablations (multi-block > causal >
  random-tube; feature > pixel targets under frozen probing) are directly transferable
  hypotheses for time-series masking design.
- **Attentive probing is the reference frozen-evaluation protocol.** Both papers show
  linear probes *under-measure* JEPA representations (unnormalized loss ⇒ no guaranteed
  linear separability; +17 pts on K400 just from attentive pooling). Any probe suite for
  LeJEPA embeddings should therefore include attentive/cross-attention probes alongside
  linear ones, and report low-label regimes — the V-JEPA label-efficiency protocol (5/10/50%
  of probe labels, multiple splits) is a ready-made design.
- **How to ask "what information is in the embedding?"** This line of work answers with
  three probe families we can mirror: (i) *task probes* deliberately split by information
  type (motion vs. appearance — the time-series analogue could be trend/frequency/shape/
  anomaly probes); (ii) *decoding probes* — a conditional diffusion decoder trained on
  frozen predictions to visualize what latent predictions actually contain (works for V-JEPA
  and I-JEPA; a time-series variant could decode masked-interval forecasts); (iii)
  *downstream-coupling probes* — anticipation (EK100), LLM alignment, and MPC planning as
  evidence of dynamical/state information. V-JEPA 2's argument that a latent world model's
  planning ability bounds and reveals the state information in the embedding applies
  verbatim to forecasting/control uses of time-series embeddings.
- **Content vs. objective trade-off.** V-JEPA's headline finding — feature prediction
  yields embeddings strong on dynamics/motion but slightly weaker on static appearance than
  image-pretrained models — is an existence proof that the JEPA objective preferentially
  encodes *predictable dynamics* and discards unpredictable detail. Characterizing the
  analogue (which temporal structures survive, which are discarded) is precisely the
  question of our probing project.
- **Practical assets.** Official code and released checkpoints/probes exist for both V-JEPA
  generations (github.com/facebookresearch/jepa, /vjepa2) — useful as reference
  implementations for masking collators, EMA target encoders, and attentive-probe training.

### Caveats / unverified items

- V-JEPA 2 (2506.09985) is an arXiv preprint (June 2025); I found no peer-reviewed venue
  as of this writing. A successor "V-JEPA 2.1" (dense features) exists in the same repo
  but is out of scope here.
- Some V-JEPA 2 details (e.g. exact probe hyperparameters in §12, full robot success-rate
  tables) were only partially extracted; the robot numbers quoted above come from the
  official GitHub repo table, not the PDF tables.
- I-JEPA has no arXiv HTML; all I-JEPA facts were extracted from the downloaded PDF text.
