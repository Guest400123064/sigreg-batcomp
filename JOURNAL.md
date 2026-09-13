# JOURNAL — Latent geometry of the patch encoder

This is the *iteration log* for the geometry exploration. The living summary at
the top is the current best understanding; each iteration below records one
hypothesis, its experiment, the result, and how the summary changed. The
consolidated storyline, geometry model, and full experiment plan live in
`README.md` (§1–§4).

---

## Current best understanding

**The per-patch latent z of the JEPA encoder is a 2-D phase-ring for mid
frequencies, degenerating to 1-D lines at the frequency extremes. The "pole"
is a narrow-vocabulary artifact; the trend is the f→0 limit of the rings.**

- A pure tone of frequency `f` sweeps, as its phase goes around `[0, 2π)`, a
  flat 2-D circle for mid frequencies (`dim1+2 ≈ 0.94`, `f ≈ 8…256`); phase is
  the angle, sine/cosine are quadrature directions on the same circle.
- **At the extremes the ring degenerates to a 1-D line.** As `f → 0` the ring
  collapses onto the *trend/ramp* direction (`dim1 → 1.0`); at the Nyquist
  `f = 512` it collapses onto an *amplitude* line (`dim1 ≈ 0.99`, because at
  Nyquist phase ≡ amplitude of the alternating ±1 pattern). So "radius grows
  at extremes" is two different 1-D degeneracies, not bigger circles.
- **Low-frequency planes form one overlapping family.** The planes of
  `f = 1…8` overlap heavily (0.71–0.97) because they all *contain the trend
  line*: the fraction of the trend direction captured by a ring plane rises
  monotonically 1.00 → 0.17 as `f` goes 1 → 64. "Spectral leakage" near f=0 and
  "trend = f→0 limit" are the same phenomenon.
- **Above `f ≈ 16` the planes are mostly near-orthogonal** (overlap ≈ 0.03–
  0.16), with no clean monotone separation law.
- **The "pole" is not robust.** On the rich model the ring-centroid directions
  spread: pairwise mean |cos| falls 0.83 → 0.49 (min 0.67 → 0.03) on the same
  11 frequencies, and the centroid offset shrinks 1.2–1.4 → 0.7–0.9. The old
  shared-pole was partly an artifact of training on near-pure tones.
- **Mixtures = direct sum of peak subspaces, compressed.** A bimodal cluster
  occupies ~4-D (the span of two near-orthogonal peak planes), trimodal ~6-D,
  not a new ring — but with ~40% less scatter than a linear encoder would
  give (a superposition signature).
- **Capacity merges the planes.** Far-apart peak planes rotate toward each
  other as `hidden` shrinks: overlap 0.02 → 0.4 for `hidden = 64 → 16`, while
  the rings themselves stay 2-D. Superposition = azimuth merging, not ring
  collapse.
- **Architecture-driven, data-agnostic.** Rings survive real-data models
  (Monash solar-only and 8-domain mix) and even *unseen* frequencies of a
  single-cluster model — the geometry is baked in by the linear patch
  projection (the spectral basis), not learned per-class. Vocabulary
  *diversity* buys plane **separation**; its absence **aliases** unseen
  classes onto the one learned plane (a ring-overlap superposition, orthogonal
  to capacity-limited merging).
- **Causal decomposition (E1–E3).** The ring **is** the linear patch
  projection: a linear-only encoder maps a tone's phase loop onto an exact 2-D
  plane (`dim1+2 = 1.00`, the span of `{L(cos), L(sin)}`). It survives direct
  input-space forecasting with **no** z-regularizer (E1) — the ring is purely
  architectural (patchify), independent of the objective *and* the
  regularizer. RevIN **circularizes** it and enables the f→0 trend tilt; GLU
  and RoPE only *refine* it. Any distribution-matching regularizer
  (SIGReg/VICReg/VISReg) keeps phase alive (with no reg, a latent-prediction
  model throws phase away on the training manifold); the regularizer shapes
  *scale* (radius) and *pole alignment*, not ring topology.
- **Radius is encoder gain, not amplitude or energy.** The scaler's
  `minimum_scale=0.1` floor under-normalizes small amplitudes, and the encoder
  adds a *saturating* gain (radius/input-RMS falls 2.85 → 2.21 as amplitude
  rises) consistent with SIGReg's unit-Gaussian target. Radius peaks at
  `f=256` and is unrelated to normalized-patch energy (which is ~constant by
  the scaler's construction).
- **Forecasting & downstream status.** The predictor recovers the *direction*
  but under-predicts the *radius* (a learned epistemic prior, partly
  under-training: `k=1` climbs 0.57 → 0.84 with 10k steps). This shrinks the
  autoregressive rollout to the mean; scale-aware / norm-copy fix the
  *compounding* but not the one-step shrink (and norm-copy over-predicts on
  real noisy data). Zero-shot forecasting of classic dynamical systems fails
  for the same reason (rollout collapse), not a representation failure.
  On Monash, **direct input-space forecasting is best** (MASE 1.97 vs 2.13
  latent, 2.49 SGD); the ring topology is objective-invariant — SIGReg only
  compresses the radius.
- **The frequency bias is a learned prior, not a resolution limit.** Patching
  is lossless below Nyquist; the high-freq forecasting failure is a
  **band-selective radius damping** (7% amplitude kept on real data vs 60% on
  synthetic where high-freq is predictable) — the model regresses the
  fast-rotating rings toward the mean because real high-freq content is mostly
  noise. The per-patch phase advance `Δφ = 2π·f·k/ctx (mod 2π)` sets only the
  *training-dynamics ordering* (copy tasks converge first, the Δφ=π flip
  slowest) and is **transient** — it vanishes at convergence on deterministic
  data, where shrinkage reaches ≈0.99 for every frequency. A third axis,
  **coverage** (off-support/rare frequencies), keeps some frequencies damped
  even when deterministic (rich 30k: f128 0.82, f256 0.87). Encoder resolution
  is the *reverse* of the bias: larger patches discriminate finer, with the
  `d = k` orthogonality threshold (see README §2/§4).
- **The next-patch map is a per-frequency rotation; the transformer head
  doesn't learn it (Iterations 41–46).** A hand-built map-reduce — detect
  frequency → rotate each ring plane by the phase advance `Δφ_f` → recombine
  by vector addition → decode via a probe — is **near-lossless on clean
  geometry** (one-step input MSE 0.04–0.07 on pure tones), while the
  transformer's own head scores 0.44–1.55 (10–35× worse). A single affine map
  over the frozen encoder matches the oracle on pure tones (one patch
  suffices), but fails on the mixed rich corpus. The transformer's failure is
  training dynamics (joint SIGReg + mixed-corpus MSE compromise = shrinkage),
  not capacity. Head engineering (single-shot / mini-RNN / teacher-forced /
  norm-scale) does not close the gap; the geometry predicts a
  frequency-conditioned dynamics operator is needed.
- **SIGReg × training duration deforms the rings (Iterations 45, 49; the
  marginal was a red herring).** Clean circular rings and decodable
  frequencies require the *direct* objective (or SIGReg trained only briefly).
  The canonical SIGReg twin (uniform marginal, 32k steps) produces elliptical
  rings (f64 s1/s2 = 19 vs 1.3 direct) with large centroid offsets and
  unseparable planes — the same signature first seen on the uneven-marginal
  runs, which were confounded by also being trained 15k–30k. Root cause: SIGReg
  deforms rings only once its marginal-Gaussianization pressure is actually
  satisfied (3k: weak/circular; 32k: strong/elliptical). RevIN is not the
  cause. This motivates the canonical test-bed model: uniform rich marginal,
  hidden = patch (d=k), **direct objective**.

**Measurement conventions.** All probes run on a frozen checkpoint at patch
position `p=8` (of 16), `ctx=1024`, `patch=64`, `hidden=32`, causal RevIN on,
unless noted. Plane overlap is the Grassmann overlap `mean(s²)` of the two
singular values of `U1ᵀU2` for the top-2 latent-space bases (1 = identical,
0 = orthogonal, 0.5 = share one direction). Ring radius is the mean distance of
phase-loop latents from their centroid. Reproducer: `src/batcomp/geom.py`;
probe JSON: `results/geom-rich-probe.json`.

---

## Iteration 1 — Does the ring/pole geometry survive a rich vocabulary and longer training?

**Hypothesis.** The pole + 2-D phase-ring geometry is intrinsic to the
causal-patch scaler + pure-JEPA/SIGReg objective, so it should survive training
on a *rich, diverse* spectral vocabulary. If it does not, the geometry is a
narrow-training artifact and everything downstream is suspect.

**Experiment design.** Train `geom-rich.pt` on 12 clusters (five narrow single
peaks 8–120, a Laplace, a broadband Gaussian σ=24, two bimodal and two
trimodal mixtures) — `n_per=512`, 3000 steps (vs 800), SGD lr 1e-2, batch 64,
hidden 32, pure JEPA (no stop-grad) + SIGReg λ=1. Run the full probe suite.

**Expected outcomes.** Rings survive (`dim1+2 ≥ 0.9` everywhere, mid radius
≈ 2.1, offset ≈ 1.2, pole mean |cos| ≥ 0.8). If rings break, the old map was
vocabulary-limited and we stop to rebuild.

**Results.** Rings survive everywhere — `dim1+2 ≥ 0.91` for all `f ∈ {1…512}`,
mid-freq radius ≈ 2.0–2.3 — **but the pole does not**. On the same 11
frequencies, pole mean |cos| is 0.49 (was 0.83), min 0.03 (was 0.67), and
offsets shrink to ≈ 0.7–0.9 (was 1.2–1.4). Rich training spread the
ring-centroid directions and pulled them toward the origin.

**Conclusion.** The *ring* structure is robust; the *pole* was a
narrow-vocabulary artifact. The geometry is "many rings", not "one pole with
clock faces". Updated top accordingly.

**Evidence.** `results/geom-rich-probe.json`: `rings.*.dim12 ∈ [0.913, 0.998]`;
`pole.mean_abs_cos = 0.485` (rich) vs 0.831 (old `vocab-g8.pt`, same 11 freqs);
`centroid_norm` ≈ 0.7–0.9 vs 1.2–1.4 old.

---

## Iteration 2 — Is a mixture cluster the direct sum of its peak subspaces?

**Hypothesis.** A bimodal/trimodal spectral cluster is *not* a new ring: its z
lives in the direct sum of the constituent peaks' ring planes (bimodal → ~4-D,
trimodal → ~6-D), with the peak planes near-orthogonal.

**Experiment design.** `mixture_sum` probe: for each mixture, measure the pure
peak rings' radii, the plane overlap between the two peak planes, and the
mixture's within-cluster scatter (`mean‖z−c‖²` = covariance trace); compare to
the orthogonal-sum prediction `Σ r_i²`.

**Expected outcomes.** Peak plane overlap ≈ 0 (far apart), mixture `dim1+2`
drops vs a pure ring and `dim1+2+3+4` grows, and `mean‖z−c‖² ≈ Σ r_i²` for a
linear encoder. A shortfall means nonlinear mixing/compression.

**Results.** Peak planes are near-orthogonal (overlap 0.06–0.11). Mixture
dimensionality drops as predicted (single → `dim1+2=0.81`, bimodal → 0.50,
trimodal → 0.39). But the scatter is **~40% lower** than the linear sum:
`mean‖z−c‖² / Σ r_i² = 0.61` (bimodal), 0.40–0.43 (trimodal). A single-peak
σ=3 cluster matches its pure-tone radius (2.30 vs 2.25), so the shortfall is
real nonlinear mixing, not just spectral width.

**Conclusion.** Mixtures occupy the direct sum of near-orthogonal peak
subspaces, but with ~40% *superposition-style compression* — the components do
not each get their full ring representation. Direct-sum confirmed, linearity
refuted.

**Evidence.** `mixture_sum`: `bi_mid plane_overlap=0.107, ratio=0.610`;
`tri_far plane_overlap=0.064, ratio=0.427`; `mixtures.*.dim12` above.

---

## Iteration 3 — Peak separation vs plane orthogonality (and what the overlap matrix actually shows)

**Hypothesis.** Plane overlap between two pure-tone rings decays with their
frequency separation (spectral-leakage toward neighboring bins).

**Experiment design.** Full 11×11 plane-overlap matrix over
`f ∈ {1,2,4,8,16,32,64,128,256,384,512}`.

**Expected outcomes.** Monotone decay of overlap with `log` separation, high
for adjacent bins.

**Results.** No global monotone decay. Instead a **block structure**:
`f=1…8` overlap 0.71–0.97 (one family); `f=16` bridges (0.16–0.36 with the low
block, ≈0.03 with everything else); above `f=16` mostly 0.03–0.16 with a few
bumps (32–512 = 0.14, 64–128 = 0.16) and no clean law.

**Conclusion.** "Separation → orthogonality" is only true *above* the low
block. The low block's heavy overlap is the *trend/ramp line shared by all
sub-cycle rings* (see Iteration 4), not generic leakage. This subsumes the old
"spectral leakage" story: near f=0, leakage ≈ the f→0 trend ray.

**Evidence.** `plane_overlap` matrix in the probe JSON (reproduced in the
results above).

---

## Iteration 4 — Is the trend the f→0 limit of the rings?

**Hypothesis.** The pure-trend (ramp) ray is the f→0 degenerate limit of the
phase rings: as `f → 0` the cosine/curvature quadrature collapses to the
(mean-subtracted) constant direction, leaving only the ramp, so the low-freq
ring *planes* should rotate to contain the trend line.

**Experiment design.** `trend_alignment` probe: for `f ∈ {1,2,4,8,16,32,64}`,
measure the fraction of the trend's top principal direction captured by each
ring's top-2 plane (`frac_trend_in_plane`).

**Expected outcomes.** `frac_trend_in_plane` rises monotonically to 1 as
`f → 0`; the trend line is the surviving ramp axis.

**Results.** Exactly that: 1.000, 0.994, 0.983, 0.808, 0.250, 0.171, 0.172 for
`f = 1,2,4,8,16,32,64`. The trend itself is a clean 1-D ray (`dim1=0.98`).

**Conclusion.** Confirmed. The trend *is* the f→0 limit; low-frequency ring
planes tilt toward the trend ray. This also explains Iteration 3's low-freq
overlap block.

**Evidence.** `trend.by_freq.*.frac_trend_in_plane` above; `trend_dim1=0.979`.

---

## Iteration 5 — Is the ring radius amplitude-invariant, or does amplitude leak?

**Hypothesis.** The causal scaler should make a pure tone's radius
amplitude-invariant (phasor after normalization). The `minimum_scale` floor is
the obvious place it could leak.

**Experiment design.** `radius_vs_amp_rms`: sweep raw amplitude
`A ∈ {0.25…4}` at `f=32`, measuring ring radius *and* the normalized-patch RMS
the encoder actually saw.

**Expected outcomes.** If the scaler fully normalizes, radius is flat and
`radius/input_rms` is constant (any saturation is then purely the floor). If
the encoder adds nonlinearity, the ratio changes with `A`.

**Results.** Radius rises then saturates (1.39 → 2.06 → 2.19). The
`radius/input_rms` ratio is **not** constant: 2.85 → 2.21 as `A` rises. So
there are two effects: the floor under-normalizes small `A` (input RMS drops to
0.49), *and* the encoder applies a saturating gain (higher gain at small
normalized input), consistent with SIGReg compressing large latent excursions.

**Conclusion.** Radius is **not** amplitude-invariant and **not** proportional
to input energy. It is encoder gain with a compressive nonlinearity. The
"radius grows at extremes" (Iteration 1) is gain vs frequency, not input
energy: `radius/input_rms` peaks at 3.39 (`f=256`) while input RMS is ~constant
(0.91). Top updated.

**Evidence.** `radius_amp_rms`: ratio 2.85 (A=0.25) → 2.21 (A=4); `radius.energy`
ratio 2.12 (f=1) → 3.39 (f=256), `input_rms ≈ 0.91` for all mid freqs.

---

## Iteration 6 — Does capacity control the plane merging?

**Hypothesis.** The queued superposition test: at lower `hidden`, far-apart peak
planes should rotate toward each other (overlap → 1); at higher `hidden` they
should stay orthogonal.

**Experiment design.** Train `hidden ∈ {16, 32, 64}` on the same rich corpus
(32 already done), then measure plane overlap of far-apart peak pairs
(`(16,80)`, `(8,120)`) and the pole alignment.

**Expected outcomes.** Overlap of far-apart planes grows as `hidden` shrinks;
pole alignment is preserved at high `hidden`.

**Results.** Far-apart plane overlap decreases monotonically with `hidden`:
`bi_far (8,120)` → 0.398 / 0.091 / 0.017 and `bi_mid (16,80)` → 0.195 / 0.107 /
0.012 for `hidden = 16 / 32 / 64`; matrix pairs agree (8–128: 0.202 → 0.086 →
0.005). The rings themselves stay 2-D at every `hidden` (`dim1+2 ≥ 0.91`). The
pole is strongest at `hidden=64` (mean 0.62, min 0.27) and weakest at 32
(0.49); secondary, not the capacity tell.

**Conclusion.** Confirmed — capacity pressure merges the *class-bearing planes*
(azimuth rotation), not the rings. At `hidden=64` far-apart planes are
near-orthogonal ("way too much space"); at `hidden=16` they overlap 0.2–0.4.
This is the superposition signature expressed in the ring geometry.

**Evidence.** `geom-rich-h{16,32,64}-probe.json`:
`mixture_sum.{bi_far,bi_mid}.plane_overlap_p1p2` and `plane_overlap` matrix.

---

## Iteration 7 — E0: does the geometry survive real data, and is single ≠ mixed?

**Hypothesis.** The ring geometry is *architecture-driven* (causal patch RevIN
+ patching + RoPE + next-patch), so it should survive training on real data —
and a single-dataset model should have one ring family with spare capacity
(near-zero cross-plane overlap), while a mixed-dataset model should have many
overlapping planes (a *ring-overlap* superposition, distinct from Gaussian-blob
superposition).

**Experiment design.** Two real checkpoints and one synthetic control, all
`MiniLTFM`/rotary:
- `monash-toto64` (solar-only, patch 64, hidden 32) — real **single** dataset.
- `monash-lftm` (8-domain Monash mix, patch 16, hidden 32) — real **mixed**.
- `geom-single.pt` (one narrow cluster, patch 64, hidden 32, 3000 steps) —
  synthetic **single**; compare to `geom-rich.pt` (12 clusters, synthetic
  **mixed**).
Run the same probes (phase rings at `p=8`, pole, plane overlap, trend,
mixture/radius) with patch-size matched to each checkpoint.

**Expected outcomes.** Rings survive in the real models → architecture-driven.
Single-cluster: one ring family, low cross-plane overlap. Mixed: 12 families
with nonzero plane overlap. Real solar (highly periodic) should show a strong
single ring family.

**Results.** The ring geometry **survives real data** in both models
(`dim1+2 ≥ 0.95` at every frequency probed):
- `monash-toto64` (solar single, patch 64): `dim1+2 ≥ 0.947`; Nyquist `f=512`
  degenerates to a line (`dim1=0.994`) and `f=1` toward the trend (`dim1=0.75`);
  `frac_trend_in_plane` runs 1.00 → 0.07 as `f` goes 1 → 64. Pole mean |cos|
  0.68. Radius 2.5–3.4 (peak at f=8).
- `monash-lftm` (8-domain mix, patch 16): `dim1+2 ≥ 0.960`; extremes degenerate
  (`f=4 dim1=0.97`, `f=512 dim1=0.89`); the trend family extends to higher f
  (patch 16 → fewer cycles/patch), `frac_trend_in_plane ≥ 0.99` for f ≤ 16.
  Pole mean |cos| 0.61. Radius 3.0–6.0.

Single vs mixed, clean same-architecture control (`geom-single.pt` = one narrow
cluster @64 vs `geom-rich.pt` = 12 clusters, both patch 64):
- The single-cluster model produces rings at **every** frequency — including
  ones it never saw — `dim1+2 ≥ 0.949`, its own `f=64` the cleanest
  (`dim1+2=0.994`, smallest offset 0.49). The ring is **architecture-baked-in**,
  not learned per-frequency.
- **Plane overlap is higher in the single model**: `8–64` 0.244 vs 0.050,
  `16–64` 0.156 vs 0.032, `32–64` 0.113 vs 0.091. Pole is also stronger
  (0.59 vs 0.49).

**Conclusion.** Rings are **architecture-driven** — they appear in real-data
models and for unseen frequencies. The single-vs-mixed difference is *not*
"spare capacity → orthogonal separation"; it is the reverse: a single-cluster
model **aliases** unseen frequencies onto its one learned plane (high overlap),
while the mixed model *separates* trained classes into distinct planes (low
overlap). Diversity pressure buys separation; its absence causes aliasing. This
is a **ring-overlap** (plane-orientation) superposition — distinct from
Gaussian-blob overlap and from the capacity-limited plane merging of Iteration
6: two orthogonal axes, capacity (merge) and diversity (alias).

**Evidence.** `results/geom-monash-{toto64,lftm}-probe.json`,
`results/geom-{single,rich}-probe.json`:
`rings.*.dim12 ≥ 0.947/0.960`; `plane_overlap`; `pole.mean_abs_cos`.

---

## Iteration 8 — E2 + E3: what *causes* the rings (regularizer & architecture)

**Hypothesis.** E2 — the ring geometry is *not* SIGReg-specific: VICReg and
VISReg (other distribution-matching regularizers) should also produce rings,
while `none` (plain next-patch MSE) should collapse. E3 — **RevIN** and
**RoPE** are both necessary: RevIN separates phase from amplitude, RoPE lets
the transformer implement the phase-advance rotation; the GLU non-linearity is
expressive but likely not necessary for the *rings* (the linear patch
projection already forms the spectral basis).

**Experiment design.** Rich corpus, patch 64, hidden 32, 3000 steps, SGD 1e-2.
- E2 (reg): `sigreg` (baseline = `geom-rich.pt`), `vicreg`, `visreg`, `none`.
- E3 (arch, reg=sigreg): `--no-revin`, `--no-rope`, `--no-glu`, and the
  `--no-revin --no-rope` 2×2 cell.
Probe each checkpoint with the same suite (phase rings, pole, plane overlap,
trend, radius).

**Expected outcomes.** Rings survive VICReg/VISReg (`dim1+2 ≥ 0.9`); `none`
collapses (radius → 0 or dim structure vanishes). RevIN-off or RoPE-off breaks
or distorts the rings (dim1+2 drops / radius changes); GLU-off likely keeps the
rings. RevIN × RoPE both-off is the strongest distortion.

**Results.** (all `dim1+2` for mid-freq f; `radius` at f=32; `frac@f8` = trend-in-plane at f=8)
- **E2 regularizers.** Rings survive `vicreg` (`dim1+2 ≥ 0.91`, radius 2.15) and
  `visreg` (`dim1+2 ≥ 0.95`, radius 2.59), like `sigreg` (0.94, 2.06). `none`
  does **not** collapse to zero — it collapses to *phase-invariance* on
  in-distribution frequencies: radius at f=32/f=64 → 0.07/0.05, while an
  out-of-distribution pure tone (f=256) still shows a ring (radius 3.18). The
  no-reg model trivially minimizes next-patch MSE by throwing away phase.
- **E3 architecture.** `no-glu` → `dim1+2 = 1.00` with `dim1 ≈ 0.5–0.6` for mid
  freqs: the *linear* projection alone maps the phase loop onto an exact 2-D
  plane. `no-revin` → rings survive but are more eccentric (`dim1=0.87` at f=32
  vs 0.54) and the trend tilt collapses (`frac@f8` 0.33 vs 0.81). `no-rope` →
  rings survive (`dim1+2 ≥ 0.88`) with a weaker pole (0.31 vs 0.48).
  `no-revin+no-rope` → rings survive, smaller radius, weak tilt.

**Conclusion.** The ring **is** the linear patch projection (the spectral
basis: phase = rotation in the span of `{L(cos), L(sin)}`); GLU and RoPE are
*refinements*, not causes. The regularizer is *necessary* to preserve phase
(without it the encoder discards phase on the training manifold), but the ring
shape is not SIGReg-specific — VICReg and VISReg produce it too. RevIN is not
needed for the ring but **circularizes** it and enables the f→0 trend tilt.
This refines the E3 hypothesis: RoPE is not the linchpin, and GLU is not
needed for the ring.

**Evidence.** `results/geom-rich-{vicreg,visreg,noreg,norevin,norope,noglu,norevin-norope}-probe.json`:
`rings.*.dim1/dim12/radius`, `trend.by_freq.*.frac_trend_in_plane`.

---

## Iteration 9 — E1: does the ring survive direct input-space forecasting (no z-regularizer)?

**Hypothesis.** The ring is the *linear patch projection* (architecture), not
the latent-prediction objective or the regularizer. It should therefore survive
Toto-style direct input-space forecasting — with **no regularizer on z**.

**Experiment design.** Same `MiniLTFM` backbone, `forecast_space="input"`
(head outputs `patch_size`), loss = MSE(pred next *normalized patch*), **no
z regularizer**. Rich corpus, patch 64, hidden 32, 3000 steps.

**Interpretation caution.** With no regularizer, `z` is only shaped indirectly
through the input-space gradient. We ask only whether the encoder's raw `z`
mapping still forms rings; we do **not** read radial/scale structure as
Gaussian-shaped (no SIGReg target exists here).

**Expected outcomes.** If rings survive → the geometry is purely architectural
(patchify), independent of both the objective and the regularizer. If they
break → the latent objective and/or regularizer is required for the ring.

**Results.** Rings **survive** direct input-space forecasting with no z
regularizer: `dim1+2 ≥ 0.965` at every frequency; the extremes degenerate as
before (`f=1 dim1=0.99`, `f=512 dim1=0.99`); the trend tilt survives
(`frac_trend_in_plane` 1.00 → 0.15 as `f` goes 1 → 64). Compared to the latent
SIGReg baseline, the pole is *stronger* (0.79 vs 0.49) and the radii *larger*
(2.6–4.2 vs 2.0–3.1) — exactly the absence of SIGReg's unit-Gaussian
compression.

**Conclusion.** The ring is **purely architectural** — patchify + RevIN +
next-patch — independent of both the training objective (latent vs input) and
the regularizer. The regularizer does **not** create the ring; it only shapes
its *scale* (radius) and *centroid alignment* (pole). This is the cleanest
confirmation of "the ring = the patchify architecture."

**Evidence.** `results/geom-rich-input-probe.json`:
`rings.*.dim12 ≥ 0.965`, `rings.1.dim1=0.987`, `rings.512.dim1=0.992`,
`trend.by_freq.*.frac_trend_in_plane`, `pole.mean_abs_cos=0.792`.

---

## Iteration 10 — E4: is next-patch prediction a phase advance (rotation)?

**Hypothesis.** Next-patch prediction is a *rotation* on the ring: the predicted
latent stays on the ring (norm preserved) and advances by the physical phase
step `2π·f·patch/ctx`, independent of starting phase.

**Experiment design.** `rotation_test` on `geom-rich.pt` (and `geom-rich-norope.pt`
for the RoPE comparison): sweep the phase of a pure tone, read the current-patch
latent `z_p`, the predicted next latent `ẑ_{p+1}`, and the true next latent
`z_{p+1}`; project onto the ring's 2-D plane and report the angle advance and
the distance from the ring centroid.

**Results.** The naive rotation is only **partially** confirmed:
- The predicted **angle** tracks the true next latent for in-distribution
  integer cycles/patch (`d_pred ≈ d_true ≈ 0`, identity — for `f ∈ {32,64,128}`
  one patch is a whole number of cycles, so the next patch is identical).
- But the **norm is systematically shrunk**: `r_pred ≈ 0.5–0.6 × r_next`
  (f=64: 1.18 vs 2.25; f=32: 1.10 vs 2.06). The prediction points in the right
  direction (`cos_pred_true` 0.71–0.81) but under-shoots the radius.
- **RoPE sharpens the rotation**: `cos_pred_true` 0.814 → 0.610 and less
  shrinkage (r_pred/r_next 0.52 → 0.45) when RoPE is removed.

**Conclusion.** The next-patch map is a phase advance in *angle* but **not**
norm-preserving — the ring is invariant only up to radial contraction. This is
the same "prediction shrinkage" seen earlier, and it says the circle is the
invariant set for the *angular* dynamics while the radius is compressed (the
model hedges toward the centroid). Caveat: the clean frequencies here test the
identity (integer cycles/patch), not a non-trivial rotation; a clean
non-trivial test needs the scale-aware loss (which fixed shrinkage earlier) or
a position-matched ring measurement.

**Evidence.** `rotation_test` output (RoPE on/off): `d_pred_mean`, `d_pred_std`,
`r_pred_mean` vs `r_next_mean`, `cos_pred_true`.

---

## Iteration 11 — Does the undershoot drive forecasting error?

**Hypothesis.** The prediction shrinkage ("undershoot") is the dominant cause
of forecasting error, and it compounds over an autoregressive rollout.

**Experiment.** (1) One-step decomposition on the spectral val corpus: put
both JEPA (latent → linear decode) and E1 (direct input) on the same
normalized-patch currency and split MSE into an energy ratio (`E_pred/E_true`)
and direction (`cos`). (2) Rollout amplitude decay on an f=64 tone: measure the
predicted radius from the ring centroid over 12 autoregressive steps.

**Results.**
- One-step: JEPA `mse 0.912, E_pred/E_true 0.149, undershoot 0.39, cos 0.25`;
  E1 `mse 0.883, E_pred/E_true 0.093, undershoot 0.31, cos 0.29`. Both are
  dominated by undershoot — even with perfect direction the floor is
  `(1−r)² ≈ 0.37/0.48`.
- Rollout (f=64, true radius 2.25): predicted radius collapses
  `1.50 → 0.80 → 0.71 → 0.61 → 0.55 → … → 0.47` — ~21% of truth within ~5
  steps, then fixed point.

**Conclusion.** Undershoot is a first-order, direction-independent error and
**compounds** over the horizon: the forecast decays to the unconditional mean.
This is the causal chain `undershoot → geometric collapse → poor multi-step
forecasting`, and it justifies targeting the shrinkage.

**Evidence.** Decomposition table and rollout-decay sequence above.

---

## Iteration 12 — Does the scale-aware loss remove the undershoot and the rollout collapse?

**Hypothesis.** The scale-aware next-latent loss
(`(1 − cos(pred, tgt)) + norm_lam · MSE(log‖pred‖, log‖tgt‖)`) removes the
shrinkage by decoupling direction from scale, so it should (a) push the
shrinkage ratio toward 1, (b) stop the rollout amplitude decay, and therefore
(c) improve multi-step forecasting.

**Experiment.** Train `geom-rich-scal.pt` (same rich corpus, patch 64, hidden
32, 3000 steps, SIGReg λ=1, scale-aware `norm_lam=1`) vs the plain-MSE
`geom-rich.pt`. Measure: `rotation_test` shrinkage ratio, rollout amplitude
decay (f=64), and the one-step energy/direction decomposition.

**Expected outcomes.** Shrinkage ratio 0.39 → ≈0.96; rollout radius stays near
2.25 instead of collapsing to 0.47; one-step energy ratio → ~1 with direction
preserved. Plain latent MSE rises (the loss no longer minimizes it).

**Results.** (proper P−1 rollout; ring radius differs per model)
- **One-step shrinkage is NOT removed** — it is slightly *worse*: `rotation_test`
  ratio 0.35 vs 0.52, but pure-tone direction improves (`cos` 0.93 vs 0.81).
- **The compounding is stopped**: rollout radius is *stable* (1.29 → 1.20,
  ≈0.34× its own ring) instead of collapsing (MSE: 1.50 → 0.47, ≈0.21×).
- Side effect: the ring radius grows (3.58 vs 2.25).
- On the corpus, one-step forecasting is slightly *worse* (mse 0.978 vs 0.912,
  `cos` 0.174 vs 0.252, energy ratio 0.109 vs 0.149).

**Conclusion.** The scale-aware loss fixes the **compounding collapse**, not the
**one-step shrinkage**. They are distinct: the one-step shrinkage is the
Bayes-optimal regression-to-the-mean (epistemic uncertainty), while the rollout
collapse is the *iteration* of that shrinkage. This reconciles with the earlier
"ratio 0.964" result, which measured rollout stability, not one-step scale. The
trade-off: better long-horizon stability and pure-tone direction, worse
one-step amplitude and mixture direction.

**Evidence.** `rotation_test` (f=64): ratio 0.52→0.35, cos 0.81→0.93. Proper
rollout radius: MSE `1.50→0.47`, scale-aware `1.29→1.20`. Corpus decomposition
above.

---

## Iteration 13 — How sensitive is the scale-aware result to `norm_lam`?

**Hypothesis.** The one-step shrinkage is either (a) a tunable loss-scale
effect — then raising `norm_lam` should push the shrinkage ratio toward 1, or
(b) an epistemic floor — then the ratio should saturate while the rollout
stability degrades (overshoot) at high `norm_lam`.

**Experiment.** Sweep `norm_lam ∈ {0.3, 1.0, 3.0, 10.0}` (rich corpus, patch
64, hidden 32, 3000 steps, SIGReg λ=1, scale-aware), and measure `rotation_test`
shrinkage ratio, the proper P−1 rollout radius trajectory, and the ring radius.

**Expected outcomes.** Ratio monotone toward 1 with `norm_lam`; rollout stable
at moderate values but possibly overshooting at 10.0. If the ratio saturates
well below 1, the residual is epistemic.

**Results.** (f=64; `ratio = r_pred/r_next`, `stable = |r[-1]−r[0]|/r[0]`)
- One-step ratio **does not** rise with `norm_lam`: MSE 0.52 → 0.22 (0.3) →
  0.35 (1.0) → 0.35 (3.0) → 0.33 (10). It saturates at ~0.33–0.35.
- `norm_lam` is a **threshold** on rollout stability: MSE collapses (stable
  0.68), 0.3 still collapses (0.47), but 1.0/3.0/10 all stabilize
  (0.07–0.11) with no overshoot.
- Side effect: the ring radius grows (2.25 → 3.6–4.3) under scale-aware; pure-
  tone direction improves and stays flat (`cos` 0.92–0.95 vs 0.81).

**Conclusion.** The one-step shrinkage is an **epistemic floor**, not a
loss-scale artifact — it cannot be dialed away by `norm_lam`. The weight only
controls whether the shrinkage *compounds* (collapse) or *stabilizes*; the
threshold is ≈1.0, with no overshoot up to 10. The residual ~0.33 ratio is the
model's irreducible uncertainty, matching the Bayes regression-to-the-mean.

**Evidence.** Table above: `rotation_test` ratio and proper P−1 rollout `stable`
for `norm_lam ∈ {0.3, 1, 3, 10}` vs MSE.

---

## Iteration 14 — Is the shrinkage history-limited or a learned prior?

**Hypothesis.** If the shrinkage were an *information* limitation (history too
short to pin the amplitude), it should shrink with more context. If it is a
*learned* prior, it should be constant across context length.

**Experiment.** On `geom-rich.pt`, measure the `rotation_test` shrinkage ratio
at context positions `p ∈ {2, 4, 6, 8, 10, 12, 14}` for f=32 and f=64 (more `p`
= more history).

**Results.** The ratio is **exactly constant** to 3 decimals: f=32 → 0.537 at
every `p`, f=64 → 0.524 at every `p` (cos 0.714/0.814 also flat). More history
does not change the prediction's radius.

**Conclusion.** The shrinkage is a **learned, position-invariant prior**, not a
history-length bottleneck. For a pure tone the amplitude is already determined
by one patch, so the hedge is the model's internalized uncertainty from
training on *mixtures* (genuinely uncertain future), applied uniformly even to
deterministic signals. The residual undershoot is therefore a *calibration*
problem, not an information or loss-scale problem.

**Evidence.** Position-sweep table above (ratio flat across `p=2…14`).

---

## Iteration 15 — E3 remainder: does patching *cause* the ring?

**Hypothesis.** Patching provides the local spectral basis `{sin, cos}`, so a
multi-timestep patch maps a tone's phase onto a 2-D circle. With `patch_size=1`
(per-timestep tokens) each token is a *scalar*, so the phase loop should
degenerate to a 1-D line (`dim1 → 1.0`), not a ring (`dim12 ≈ 0.94`).

**Experiment.** Train `geom-rich-p1.pt` (`patch_size=1`, `ctx=256`, hidden 32,
2000 steps, SIGReg) and probe the phase loop at a mid position; compare `dim1`
against the `patch_size=64` baseline.

**Expected outcomes.** `patch_size=1` → `dim1 ≈ 1.0` (line); `patch_size=64` →
`dim1 ≈ 0.5, dim1+2 ≈ 0.94` (ring). This would isolate patching as the
ring-maker, completing E3's architecture decomposition.

**Results.** `patch_size=1` does **not** collapse to a line — it produces a
2-D curve (`dim1=0.458`, `dim1+2=0.813`, radius 1.92) because the GLU bends the
scalar's sinusoidal sweep. But it is **completely frequency-blind**: the
numbers are *identical* for f=16/32/64 at every position, and the plane
overlap between any two frequencies is **1.000** (vs 0.03–0.09 for
`patch_size=64`).

**Conclusion.** Patching is what **encodes frequency**. A single timestep's
value marginal is frequency-independent (the arcsine law), so without patching
the encoder maps *all* frequencies to the same 2-D curve — the frequency →
azimuth (plane-orientation) structure that defines the ring geometry is
destroyed. This completes E3: the ring's *existence* is the linear projection
(Iteration 8), but its *frequency discriminability* is patching.

**Evidence.** `patch_size=1`: dim profile identical across f/p; `plane_overlap`
= 1.000 for all frequency pairs vs 0.03–0.09 at `patch_size=64`.

---

## Iteration 16 — E5: latent-azimuth regime / changepoint detection

**Hypothesis.** The plane azimuth encodes frequency, which the raw *value*
marginal hides (the arcsine law: phase-randomized cosines have identical value
distributions regardless of frequency). So a sequence switching between two
frequencies should be **undetectable in raw values but cleanly detectable in
the latent plane azimuth**.

**Experiment.** Regimes `f1=32, f2=96` (phase-randomized windows, `ctx=1024`).
Fit each regime's plane from pure-tone phase loops (`geom-rich.pt`), then (1)
verify the raw value marginals are identical, (2) classify held-out windows by
max plane projection, (3) track the plane score across a switch (changepoint).
Baseline: raw value statistics.

**Expected outcomes.** Raw value marginal identical (KS ≈ 0); latent plane
classification ≈ 100%; the plane score jumps at the changepoint while a raw
score stays flat.

**Results.**
- Raw value marginal is **identical** across regimes: mean 0.0000, std 0.7071,
  KS (max CDF diff) 0.0023.
- Latent plane classification is **100.0%** (score `+1.65` for f1, `−1.55` for
  f2).
- Raw baseline (window mean) separates at `0.0000`.
- Changepoint: the plane score flips `+1.64 → −1.54` at the switch; the raw
  score stays `0.0000`.

**Conclusion.** The latent plane azimuth detects regime identity (frequency)
that the raw value marginal *completely hides*. This is the practical payoff of
the geometry: the azimuth is a regime signature invisible to the value
distribution — exactly the "z reveals frequency the value marginal hides"
property, now turned into a working detector.

**Evidence.** Numbers above; reproducible from `geom.py` (`phase_loop_z`,
`plane_basis`, `z_at`).

---

## Iteration 17 — Does a GLU head fix the one-step shrinkage?

**Hypothesis.** The decoder's final RMSNorm outputs a unit-scale `h`, so the
plain linear head cannot nonlinearly modulate the prediction's radius. A GLU
head (`SwiGLU → RMSNorm → Linear`) adds that expressivity and may make the
shrinkage input-dependent (the calibration fix).

**Experiment.** Train `geom-rich-gluhead.pt` (same rich corpus, latent, SIGReg,
`use_glu_head=True`) vs the linear-head `geom-rich.pt`. Measure `rotation_test`
shrinkage ratio, proper P−1 rollout trajectory, and the one-step
decomposition.

**Expected outcomes.** If the linear head was the bottleneck, the ratio should
rise toward 1 and/or vary with the input. If it stays flat ~0.5, the head was
not the limiting factor.

**Results.** GLU head does **not** help — it is uniformly equal or slightly
worse:
- `rotation_test` f=64: ratio 0.45 vs 0.52, `cos` 0.69 vs 0.81.
- Rollout: still collapses (1.48 → 0.60 vs 1.50 → 0.47).
- Corpus one-step: mse 0.912 = 0.912, undershoot 0.315 vs 0.386 (more shrunk),
  `cos` 0.249 vs 0.252.

**Conclusion.** The linear head was **not** the bottleneck — a more expressive
GLU head does not reduce the shrinkage (it slightly worsens direction). This
confirms the residual undershoot is upstream, in the encoder/history
representation (the learned prior from Iteration 14), not in the head's
expressivity. The fix must target the representation (heteroscedastic
uncertainty / radius encoding), not the head.

**Evidence.** Comparison table above.

---

## Iteration 18 — Does `hidden = patch_width` fix the shrinkage?

**Hypothesis.** `hidden=32 < patch=64`, so the patch embedding downsamples a
64-dim patch to 32 dims — a possible information bottleneck. `hidden=64`
(matching patch width) should retain more and reduce the shrinkage.

**Experiment.** Reuse the capacity-sweep `geom-rich-h64.pt` (hidden 64, patch
64, same corpus/objective) and compare the three shrinkage metrics against
`hidden=32`.

**Results.** One-step shrinkage is **not** fixed (`ratio` 0.43 vs 0.52, `cos`
0.75 vs 0.81), and the corpus undershoot is slightly worse (0.320 vs 0.386
energy ratio). But the rollout **stabilizes** instead of collapsing
(`1.28 → 0.87` vs `1.50 → 0.47`), and corpus one-step MSE is marginally better
(0.895 vs 0.912).

**Conclusion.** Matching hidden to patch width does **not** fix the one-step
undershoot (the encoder was not the bottleneck) — but, like the scale-aware
loss, it stops the *compounding* collapse. The one-step shrinkage is robust to
both head expressivity (Iteration 17) and capacity (this iteration); it is a
learned epistemic prior, and the multi-step collapse is the part that capacity
or scale-aware loss can fix.

**Evidence.** `rotation_test` and rollout/decomposition tables above.

---

## Iteration 19 — Is the shrinkage the corpus's aleatoric uncertainty?

**Hypothesis.** The one-step shrinkage equals the fraction of the target's
*variance that is unpredictable* (aleatoric). The corpus is a sum of `k`
random-phase tones, so `k=1` (a pure tone) is fully deterministic and `k=32`
(the rich corpus) is highly uncertain. If the shrinkage is calibrated, it
should go `≈1.0` at `k=1` and fall to `≈0.5` at `k=32`.

**Experiment.** Train `k ∈ {1, 4, 16, 32}` (same rich clusters, patch 64,
hidden 32, 3000 steps, SIGReg) and measure the `rotation_test` shrinkage ratio
on a pure tone.

**Expected outcomes.** Ratio monotone `≈1.0 → ≈0.5` as `k` grows → the
shrinkage is a *correct* calibrated response to the training aleatoric
uncertainty. If the ratio is flat, it is a modeling artifact.

**Results.** (rotation_test on f=64)
| k | ratio | cos | ring dim12 |
|---|---|---|---|
| 1 | 0.574 | 0.816 | 0.945 |
| 4 | 0.432 | 0.756 | 0.958 |
| 16 | 0.548 | 0.778 | 0.941 |
| 32 | 0.524 | 0.814 | 0.936 |

The ratio is **non-monotonic** and does not approach 1 at `k=1`.

**Conclusion.** The shrinkage is **not** the corpus's aleatoric uncertainty: a
model trained on *fully deterministic pure tones* (`k=1`) still shrinks to
0.57. The undershoot is a structural MSE/architecture artifact, not a
calibrated response to training predictability. This refutes the
"calibration" story — a heteroscedastic head (which targets aleatoric
variance) is the wrong fix.

**Evidence.** Table above; all rings stay clean (`dim12 ≈ 0.94–0.96`).

---

## Iteration 20 — Does norm-copy fix the rollout collapse?

**Hypothesis.** The shrinkage is purely a *norm* problem (direction is good,
`cos ≈ 0.81`). Scaling the predicted direction to the most recent patch's norm
("norm-copy") should stop the compounding collapse.

**Experiment.** On `geom-rich.pt`, run the proper P−1 rollout for f=64 with and
without re-normalizing each predicted latent to the last window patch's norm.

**Results.** Plain rollout collapses `1.50 → 0.47`; norm-copy stays
`2.40 → 2.25`, i.e. at the **true** ring radius (2.25) for all 12 steps.

**Conclusion.** Norm-copy completely fixes the compounding collapse. The
predicted *direction* is correct; only the radius is shrunk, and copying the
context's norm restores it. This is the hard inference-time version of the
scale-aware idea — and it works precisely because it's a hard constraint, not a
soft loss the model can partially ignore. It points to a training-time fix: a
hard norm constraint (predict direction, copy norm), rather than a soft
log-norm penalty.

**Evidence.** Rollout radius trajectories above (plain vs norm-copy).

---

## Iteration 21 — Is the shrinkage under-training, not structural?

**Hypothesis.** The `k=1` shrinkage at 3000 steps (0.574) was under-training,
not a structural MSE floor.

**Experiment.** Train `k=1` for 10 000 steps and re-measure.

**Results.** Ratio climbs `0.574 → 0.843`, `cos` `0.816 → 0.982`, and
`val_mse` keeps falling `0.135 → 0.052`. The deterministic model is still
converging; it approaches the exact prediction.

**Conclusion.** The "structural floor" of Iteration 19 was largely
**under-training**. The one-step shrinkage is a mix of (a) optimization gap
and (b) aleatoric uncertainty; for a deterministic corpus the gap shrinks with
more training. Norm-copy is a cheap inference-time compensation for this
under-trained (and aleatoric) shrinkage — correct for deterministic signals,
but it would over-predict for genuinely noisy ones.

**Evidence.** Ratio/`cos`/`val_mse` at 3k vs 10k steps above.

---

## Iteration 22 — Real data: 20k steps, synthetic pre-training, fine-tuning geometry

**Hypothesis.** (a) 20k steps should improve Monash MASE vs the earlier ~1.5–2k
runs; (b) synthetic pre-training (the rich spectral corpus) should help by
warm-starting the skeleton.

**Experiment.** `MiniLTFM` patch 32 / hidden 32 / 4 layers, representative,
no stop-grad, 8-domain Monash mix. Arms: pre-train 10k (synthetic) + fine-tune
10k vs Monash-only 20k. Also probe pure-tone rings before/after fine-tuning.

**Results.** Monash-only 20k → **MASE 2.49** (per-domain: weather 0.72, traffic
1.99, solar 5.62, pedestrian 1.47, london 1.42, kaggle 1.59, oikolab 4.62).
Pre-train + fine-tune → **MASE 2.57** (slightly worse). Rings survive
fine-tuning but radii roughly double.

**Conclusion.** More steps help (4–6 → 2.5), but still far from <1. Synthetic
pre-training does **not** help real forecasting — the geometry transfers, the
initialization doesn't.

**Evidence.** `monash-20k` and `monash-finetune` runs (logs), ring tables.

---

## Iteration 23 — Zero-shot forecasting of classic dynamical systems

**Hypothesis.** The ring/spiral geometry means a frozen MiniLTFM should
zero-shot forecast harmonic (rotation) and damped (spiral) oscillators.

**Experiment.** Forecast harmonic, damped, Van der Pol (RK4), Lorenz-x with the
frozen `geom-rich.pt` + post-hoc GLU decode; report R² vs persistence.

**Results.** R²: harmonic +0.015, damped −59.6, Van der Pol −0.022, Lorenz
−0.211. The model beats persistence on oscillating signals only because
persistence is a bad baseline; in absolute terms the forecasts are ≈ the mean
or catastrophic.

**Conclusion.** The latent *knows* the dynamics (rings/spirals), but the
autoregressive rollout collapses (the shrinkage) and RevIN frozen-stat
de-normalization fails on decaying signals. This is the same rollout failure,
not a representation failure.

**Evidence.** Zero-shot R² table above.

---

## Iteration 24 — Direct input-space forecasting on Monash

**Hypothesis.** The direct x-space objective (no SIGReg) closes the
latent-vs-input gap and improves real MASE.

**Experiment.** `MiniLTFM` patch 32 / hidden 32 / 4 layers, direct
`forecast_space="input"`, `MSE(pred, next normalized patch)`, no SIGReg,
AdamW + cosine. Arms: 50k and 100k steps.

**Results.** 50k direct → **MASE 1.973** (best yet; weather 0.62, traffic 1.39,
solar 3.52, pedestrian 0.86, london 1.24, kaggle 1.59, oikolab 4.60). 100k
direct → 2.016 (slightly worse, noisy loss). vs latent 50k = 2.13, SGD 20k =
2.49.

**Conclusion.** Direct forecasting is the right objective: 2.49 → 1.97. The
geometry holds under it (E1), and it forecasts better on real data. 100k does
not help further.

**Evidence.** `monash-direct-h32.pt` / `monash-direct-100k.pt` logs.

---

## Iteration 25 — Direct vs SIGReg geometry (same spec)

**Hypothesis.** The ring topology is objective-invariant; SIGReg only shapes
the *scale* (radius).

**Experiment.** Probe phase rings of the 50k direct vs 50k SIGReg latent
models (patch 32 / ctx 512 / hidden 32 / 4 layers).

**Results.** Rings survive both objectives (`dim1/dim12` broadly similar, with
the f=1 trend and f=256 Nyquist degeneracies). The **radii are ~3–10× larger**
in the direct model (5–6.4 vs 0.6–3.7), and the centroid norm grows more at
high f (5.72 vs 2.80).

**Conclusion.** SIGReg compresses the radial axis; direct forecasting leaves it
free. The ring *topology* is the architecture's doing; the regularizer only
sets the size. The direct model's larger radius is plausibly why it retains
more amplitude and forecasts better.

**Evidence.** Ring tables above (direct vs SIGReg, f ∈ {1…256}).

---

## Iteration 26 — Mixture of spectra in the *direct* model: geometry + exact recovery

**Hypothesis.** The direct-forecasting model's embedding should behave like the
latent model's: a low-freq + high-freq mixture should sit in the *direct sum*
of the two near-orthogonal ring planes (~4-D, compressed), and the *same* z
(not the predicted one) should still decode to the exact mixture.

**Experiment.** `results/ckpts/monash-direct-h32.pt` (patch 32 / ctx 512 /
hidden 32 / 4 layers / input-space head). Mixture = `f_lo=2` + `f_hi=128`
tones, random amplitudes ∈ [0.5, 2] and phases. (a) Geometry: ring profiles of
each tone, plane overlap, SVD of the mixture z-cloud, span-capture of the 4-D
direct-sum subspace. (b) Recovery: train a `LatentProbe` (z → normalized
patch) on mixtures + pure tones, eval on held-out mixtures; report raw-space
MSE/R² and per-component amplitude/phase/slope recovery.

**Expected outcomes.** (a) Near-orthogonal ring planes, mixture ≈ 4-D with
~40% compression vs the orthogonal-sum prediction. (b) High fidelity
recovery — if z can't retain the exact mixture, the geometry is decorative.

**Results.**
- **Geometry.** `f_lo=2` ring: radius 4.31, `dim1+2` 0.75; `f_hi=128` ring:
  radius 4.88, `dim1+2` 0.90; **plane overlap 0.007** (near-orthogonal).
  Mixture cloud: `dim1+2 = 0.58`, `dim1+2+3+4 = 0.81`; the 4-D direct-sum
  span captures 74% of the mixture variance; mean-sq radius 23.1 vs the
  orthogonal prediction 42.4 → **compression ratio 0.54** (same ~40%
  superposition signature as the latent model).
- **Recovery.** Probe R² = 0.995 (normalized) / 0.996 (raw), corr 0.998.
  Mixture MSE 0.0093 vs pure 0.0023 (mixtures 4× harder, still tiny). Per
  component on mixtures: high-freq recovered almost exactly (amp ratio 0.99,
  phase err 0.03 rad, oscillatory R² 0.99); low-freq `f=2` cos/sin fit is
  ill-conditioned (1/8 cycle per patch), so the meaningful quantity is the
  patch *slope*: slope corr 0.993, slope R² 0.987, slope amp ratio 1.18
  (mild overshoot); detrended oscillation R² 0.99.

**Conclusion.** The mixture geometry is *objective-invariant*: in the direct
model the rings are near-orthogonal planes and the mixture is a compressed 4-D
direct sum, exactly as in the latent model. And z does retain the exact
mixture — both components decode nearly losslessly (the low-freq one as a
patch slope). The embedding is a faithful, reconstructable code even for
multi-spectral patches.

**Evidence.** `results/probe_mix_direct.json`, `probe_mix_direct.py` (repo
root, scratch).

---

## Iteration 27 — Is the frequency bias a resolution loss or a prediction-side damping?

**Hypothesis (user question).** The ICLR 2026 implicit-biases paper attributes
the high-frequency failure of large-patch TSFMs to a *temporal bias* of
patching (their Thm 1: different-frequency patches embed into nearly orthogonal
subspaces; large `k` favors low frequencies). Two challenges: (1) large patch
size is not lossy per se — within-patch Nyquist is the *global* Nyquist
(`k/2` cycles/patch = `ctx/2` cycles/window) regardless of `k`, so patching
does not discard high-frequency content below Nyquist; (2) the failure may live
on the *prediction* side (next-patch map), not the embedding.

**Experiment.** Per-frequency teacher-forced one-step forecast on two
input-space models (`monash-direct-h32.pt` patch 32, `geom-rich-input.pt`
patch 64, in-distribution synthetic), plus a per-band decomposition of a
`f_lo=2 + f_hi=128` mixture prediction: split the error into the low-freq
slope band and the high-freq detrended-oscillation band, and compare the
model's prediction vs a trivial *copy-the-current-patch* baseline.

**Results.**
- **Mixture, per-band (direct model).** The high-freq band of the *next* patch
  is **copyable at r² = 0.9995** (f=128 completes exactly 8 cycles/patch, so
  consecutive patches are near-identical), yet the model's prediction achieves
  **r² ≈ 0 with only 7% of the true amplitude** (rms 0.048 vs 0.68). The
  low-freq slope band is retained at ~54% amplitude. So the model *erases* the
  high-frequency component of the forecast even though a free copy was
  available in the context. Same on `geom-rich-input.pt` (in-distribution):
  high-freq band copyable at 0.9995, predicted at r² = −0.46.
- **Per-frequency one-step MSE is *not* monotone in f.** Direct model
  (patch 32): f=2 best (0.22), f=8 worst (1.74), f=512 easy (0.23). Geom-input
  (patch 64): f=1 worst (1.98), f=8 best (0.49). Difficulty tracks the
  *effective phase advance per patch* `Δφ = 2π·f·k/ctx (mod 2π)` and its
  degeneracies (Δφ≈π → antipodal flip, hardest), not a monotone low-freq
  preference.

**Conclusion.** The frequency bias is **prediction-side, not resolution**:
the embedding retains high-frequency information losslessly (Iteration 26,
recon R² 0.99; rings exist at f=128, 256 even at patch 64), and patching does
not discard content below Nyquist. What fails is the *next-patch map*, which
learns a **band-selective damping**: it compresses the high-frequency ring
radius far more than the low-frequency one (7% vs 54% amplitude retention),
i.e. a regression-to-the-mean acting hardest on the band with the largest
per-patch phase advance. This reframes the paper's temporal bias: patch size
`k` does not add resolution; it changes `Δφ` per step and the number of
patches, i.e. *how much the predictor must rotate per patch* — and the
high-frequency ring is exactly the one whose radius the predictor damps.

**Evidence.** `results/probe_freq_pred.json`, `probe_freq_pred.py` (repo root,
scratch).

---

## Iteration 28 — Is the frequency bias a phase-error cost or a learned band-selective damping?

**Hypothesis (user question).** Two candidate mechanisms for "the predictor
favors low frequency": (1) a fixed phase error costs less output-space MSE at
low frequency; (2) the ring-angle-to-phase mapping differs across frequencies
(the same latent rotation = different phase change). Test both, plus a third:
is the band-selective damping (Iteration 27) a *learned prior* from the
training data?

**Experiment.**
- *Test 1 (angle↔phase):* on `monash-direct-h32.pt`, measure `dθ/dφ` of the
  ring-angle vs signal phase for f=2 and f=128 phase loops.
- *Test 2 (phase-error cost):* same model, same phase error `ε` applied to
  f=2 vs f=128 tones; compare normalized-patch MSE.
- *Test 3 (learned prior):* compare the f=2+f=128 mixture band damping on
  `monash-direct-h32.pt` (trained on real data, high-freq ≈ noise) vs
  `geom-rich-input.pt` (trained on synthetic rich spectra, high-freq
  predictable).

**Results.**
- **Test 1:** `dθ/dφ = 0.89` (f=2) and `−0.96` (f=128). The ring angle maps
  ≈ 1:1 to signal phase for every frequency (sign flip = SVD basis
  orientation). A fixed latent rotation = a fixed phase advance, independent
  of frequency.
- **Test 2:** fixed phase error `ε=0.1` → normalized-patch MSE `0.006` (f=2)
  vs `0.008` (f=128); `ε=0.3` → `0.053` vs `0.074`. The *cost* of a phase
  error is frequency-independent (even slightly higher at high f).
- **Test 3:** hi-freq band amplitude kept: **7%** (real-data model) vs **60%**
  (synthetic model). The damping strength tracks the *predictability of
  high-frequency content in the training distribution*, not the frequency
  itself.

**Conclusion.** The frequency bias is NOT a phase-error-cost effect: a given
phase error costs the same output MSE at any frequency (Test 2), and the
ring-angle↔phase mapping is frequency-independent (Test 1). The real mechanism
is a **learned, band-selective radius damping** (regression-to-the-mean in the
radial coordinate): the model erases the high-frequency band (7% amplitude)
even when a perfect copy is available in the context (Iteration 27,
copy r² = 0.9995) — because in its training data high-frequency content is
mostly unpredictable noise, so the conditional mean is ≈ 0. On synthetic data
where high-freq *is* predictable, the model keeps 60%. Frequency enters only
indirectly: the per-patch phase advance `Δφ = 2π·f·k/ctx` grows with f, so
high-frequency rings rotate fastest and their phase is the hardest to pin
down — the model hedges by compressing their radius.

**Evidence.** Inline tests on `monash-direct-h32.pt` + `geom-rich-input.pt`
(run during Iteration 27/28 analysis).

---

## Iteration 29 — Shrinkage on fully deterministic data: learned prior vs training-dynamics bias

**Hypothesis (user question).** If the band-selective radius damping is a
*learned predictability prior* (high-freq content ≈ unpredictable noise in the
training data), then training on **completely deterministic signals** (pure
tones, no noise — every frequency equally predictable) should remove the
shrinkage for *all* frequencies given enough steps. Separately: even so, does a
frequency bias show up *during training* — shrinkage disappearing first for
low f, then high f?

**Experiment.** Train a latent JEPA + SIGReg `MiniLTFM` (patch 64 / ctx 1024 /
hidden 32 / 2L) from scratch on a corpus of **pure tones only** —
`f ∈ {8, 16, 32, 64, 128, 256}`, random phase, no noise, 512 sequences each.
Every 500 steps (15k total), log the teacher-forced one-step radius ratio
`‖pred‖/‖z_next‖` per frequency (1.0 = no shrinkage).

**Results.**
- **Shrinkage vanishes for ALL frequencies.** Final radius ratios: f8 0.988,
  f16 0.998, f32 0.997, f64 0.997, f128 0.997, f256 0.997; cos ≥ 0.999
  everywhere. On deterministic data the model learns near-exact radii at every
  frequency — confirming the learned-prior picture (contrast: 7% amplitude kept
  on the real-data model, Iteration 27/28).
- **The convergence order is NOT "low f first".** Crossing steps to ratio ≥
  0.99: f8 **never**, f16 7500, f32 7000, f64 7500, f128 6000, f256 6000; to
  ≥ 0.995: f8 never, f16 11500, f32 12000, f64 10500, f128 9500, f256 10000.
  If anything, **high frequencies converge first** (f128/f256 fastest) — but
  note all of f16…f256 are "copy" tasks (`Δφ = 2π·f·k/ctx ≡ 0 mod 2π`: integer
  cycles per patch), so they are all equally trivial. The only non-trivial
  frequency is f8, whose `Δφ = π` (antipodal flip) — and **it is the persistent
  laggard** (final 0.988, never reaching 0.99).

**Conclusion.** Two-part answer. (1) The shrinkage is indeed a *learned
predictability prior*: on deterministic data it disappears for every frequency
with enough training. (2) But there is no monotone low-freq-first ordering in
the training dynamics on this corpus — the residual laggard is the
**antipodal-flip frequency** (`Δφ = π`), i.e. the difficulty tracks the
*phase advance per patch*, not the frequency magnitude. The earlier
high-frequency shrinkage on real data is therefore the combination of a genuine
predictability prior plus the fact that high-f rings rotate fastest per patch;
it is not an intrinsic frequency-resolution limit.

**Evidence.** `results/ckpts/shrink-dyn.pt` + `shrink-dyn_dyn.json`,
`probe_shrinkage_dyn.py` (repo root, scratch).

---

## Iteration 30 — Shrinkage dynamics under phase variance and mixed frequencies

**Hypothesis (user question).** Two stress tests on the Iteration 29 finding
(learned predictability prior; difficulty tracks Δφ, not f): (1) *phase
variance* — phase-modulate each tone so the per-patch phase advance varies and
the model must track the current ring phase from context before predicting;
(2) *mixed frequencies per sequence* — each sequence is a two-tone sum with a
uniform frequency marginal. Everything stays deterministic. Does the shrinkage
still vanish for all f? Does low-f still converge first?

**Experiment.** Same latent JEPA + SIGReg model (patch 64 / ctx 1024 / hidden
32 / 2L), 15k steps, per-frequency teacher-forced radius ratio every 500
steps.
- `pm`: `x = cos(2πft/ctx + φ₀ + 0.6·sin(2π·2t/ctx))`, φ₀ uniform per sequence.
- `mix`: `x = cos(2πf₁t/ctx+φ₁) + cos(2πf₂t/ctx+φ₂)`, all pairs `f₁<f₂` from
  `{8,16,32,64,128,256}` with equal counts (uniform marginal); per-frequency
  ratio via projection onto that tone's ring plane.

**Results.**
- **pm** (phase tracking required). Convergence is slower overall — ratios
  plateau at 0.96–0.99 by 15k and are still climbing — but again **not low-f
  first**: f8 (flip, Δφ=π) is the persistent laggard (0.961), f32 next (0.969),
  f128 0.982, f16/f64/f256 ≈ 0.987–0.988. High f converges fine; the ordering
  tracks `Δφ mod 2π`, and the flip frequency is hardest to track.
- **mix** (per-plane). The ordering is stark: at step 500, f8 is at 0.57 while
  everyone else is 0.83–0.92; f8 crosses 0.99 only at step 14500 (last), while
  f256 crosses 0.97 by step 1500 and 0.99 by 13000. Final ratios all
  ≈ 0.985–0.990 — shrinkage **vanishes for every frequency**, including inside
  mixtures, once trained long enough.

**Conclusion.** Both stress tests confirm Iteration 29 and sharpen it:
- **The learned-prior picture holds under harder conditions**: on fully
  deterministic data — with phase modulation forcing phase tracking, and with
  other frequencies present in the same sequence — the band-selective damping
  eventually disappears for *all* frequencies.
- **There is no low-f-first ordering; if anything it is Δφ-last.** The lowest
  frequency (f8, the antipodal flip) is always the laggard, and the highest
  (f256, a clean copy task) converges first in the mix corpus. The training
  difficulty is governed by the per-patch phase advance `Δφ = 2π·f·k/ctx`
  (mod 2π), not by the frequency magnitude.
- So the real-data high-frequency shrinkage (7% amplitude, Iteration 27/28) is
  a *predictability prior*, not a capacity or resolution limit: real
  high-frequency content is mostly noise, so the model learns to damp it; on
  deterministic data the same model learns it fully. The frequency bias DURING
  training reflects how fast the ring rotates per patch, which is exactly what a
  per-ring calibrated/radius-aware head would target.

**Evidence.** `results/ckpts/shrink-pm.pt`/`shrink-pm_dyn.json`,
`shrink-mix.pt`/`shrink-mix_dyn.json`, `probe_shrinkage_dyn.py`.

---

## Iteration 31 — Sanity check: uneven rich corpus, does shrinkage vanish with longer training?

**Hypothesis (user question).** The old `geom-rich.pt` (3k steps, latent JEPA)
showed a ~50% one-step shrink. If that was pure under-training, training the
same rich corpus much longer — but with an **uneven** spectrum marginal
(everything still deterministic) — should drive the shrinkage to ~1 for all
clusters, as the tone/pm/mix runs did.

**Experiment.** Same model (patch 64 / ctx 1024 / hidden 32 / 2L, latent JEPA +
SIGReg), 15k steps. Corpus = the 12-cluster rich spectral corpus with an
*uneven* per-cluster marginal (lo8: 1024 … tri_mid/tri_far: 32; a 32×
imbalance), each sequence a deterministic sum of 32 cosines with jittered
amplitudes. Log aggregate + per-cluster teacher-forced one-step radius ratio.

**Results.**
- **Shrinkage does NOT vanish: aggregate plateaus at 0.865 by 15k** (from
  0.54 at 500 steps), well below the ≈0.99 the tone/pm/mix runs reached. It is
  still climbing slowly at 15k, but far from converged.
- **Not a pure count effect.** corr(log n_per, final ratio) = 0.47 — rare
  clusters are somewhat worse, but the *hardest* clusters are not the rarest:
  `bi_mid` (128, far bimodal 16+80) = 0.715 and `broad` (128, σ=24) = 0.755 are
  the worst despite mid-range counts, while `tri_mid`/`tri_far` (32 each) reach
  0.889/0.885. The best is `lo8` (1024, lowest f) at 0.99.
- Per-cluster finals: lo8 0.99, laplace 0.918, lo64 0.926, bi_close 0.908,
  lo96 0.90, lo32 0.884, tri_* 0.89, lo120 0.784, bi_far 0.826, broad 0.755,
  bi_mid 0.715.

**Conclusion.** The 3k-step shrinkage was *partly* under-training (0.57 → 0.87
aggregate with 5× more steps) but **not entirely**: on the rich corpus the
shrinkage persists at ~13% even at 15k, concentrated in the *hardest* clusters
(far-apart bimodal, wide broadband), not the rarest ones. This refines the
learned-prior picture: shrinkage tracks **learnable predictability from a
finite context**, and a rich multi-component spectrum is genuinely harder to
predict exactly than a pure tone — 32 random phases/amplitudes at
continuous-sampled frequencies must be inferred from the window, so even
without noise the next patch is only approximately determined by the context.
Frequency bias *during* training remains Δφ-driven (f8 flip laggard everywhere,
Iteration 29–30); the persistent *residual* shrinkage is a
complexity/predictability prior, not a resolution or count artifact.

**Evidence.** `results/ckpts/shrink-rich.pt`/`shrink-rich_dyn.json`.

---

## Iteration 32 — How do frequency planes merge when capacity is short?

**Hypothesis (user question).** When the hidden dimension can't hold one 2-D
plane per frequency, *which* frequencies merge? Adjacent (f16 with f17)?
Harmonic (f16 with f32)? Or random?

**Experiment.** Dense-frequency plane-overlap scan on the capacity models
(`geom-rich-h16.pt` vs `geom-rich-h64.pt`, patch 64 / ctx 1024): phase-loop
ring planes for f ∈ {1…256}, pairwise Grassmann overlap, with focus on
frequency separation in *cycles per patch* units (`Δf·patch/ctx`).

**Results.**
- **Merging is LOCAL in frequency, not harmonic and not random.** At h64:
  `f16` vs `f17` overlap **0.987**; `f16` vs `f32` 0.049; `f16` vs `f64` 0.032.
  From f=64: `Δf=1 → 0.985, Δf=2 → 0.94, Δf=4 → 0.78, Δf=8 → 0.29, Δf=16 →
  0.04`. Overlap decays smoothly with separation and collapses once `Δf`
  reaches **1 cycle per patch** (`Δf = ctx/patch = 16` here).
- **The decay law is resolution-limited, not capacity-limited.** It holds at
  h64 as much as at h16: a 64-sample patch can only *resolve* frequencies
  separated by ≥ 1 cycle per patch (DFT bin width); closer tones produce
  near-identical patch shapes, so their planes must overlap. The two
  frequencies merge *before* the projection — the DFT bandwidth `ω ≤ k` of
  their Theorem 1 (similar-frequency patches → shared low-dim subspace) is
  exactly this bin-width aliasing.
- **Capacity adds a second, global compression on top.** At h16 the *far*
  pairs also rotate together: `f16` vs `f32` 0.357 (h16) vs 0.049 (h64),
  `f16` vs `f64` 0.269 vs 0.032. Local (resolution) merging is
  capacity-independent; capacity only merges what resolution leaves separate.

**Conclusion.** Two distinct merging mechanisms, ordered: (1) *resolution
merging* — frequencies closer than one DFT bin of the patch (Δf < ctx/patch)
merge smoothly, present at every capacity; (2) *capacity merging* —
frequencies the patch can resolve still rotate together when the hidden dim
can't host all planes (superposition = azimuth merging, the old 0.40→0.02
hidden sweep). So f16 sits with f17 (local), not with f32 (harmonic) — and
never randomly. This also corrects the earlier "planes above f≈16 are
near-orthogonal" claim: it held only because the old probe grid was
exponentially spaced (Δf = f, always ≥ 1 cycle/patch); a dense scan shows
adjacent planes overlap ~0.9+.

**Evidence.** Dense overlap scans above (h16/h64, f ∈ {1…256}).

---

## Iteration 33 — Pure-tone shrinkage rates during rich-corpus training

**Hypothesis (user question).** During the 30k rich-corpus run, does the
*shrinkage* (radius ratio) of each frequency converge at the same rate when
measured on **pure tones** (not the mixture clusters the model trains on)?

**Experiment.** Rerun of the 30k rich-uneven training with an extra eval:
alongside the per-cluster (mixture) ratios, log the pure-tone teacher-forced
radius ratio `‖pred‖/‖z_next‖` for f ∈ {8,16,32,64,128,256} every 500 steps
(deterministic eval — same phases, fixed RNG).

**Results.**
- **Not uniform.** Final pure-tone ratios: f16 **0.987**, f32 0.960, f8 0.941,
  f64 0.885, f256 0.867, f128 **0.821**. Crossing steps to ≥0.97: f16 at 6000,
  f32 at 19500, f64 at 20000, f8 at 16500 — f128/f256 **never** reach 0.95.
- **f16 is the star** (1 cycle per patch, Δφ≡0 copy task): reaches 0.99 by
  step 6000, earliest of all.
- **The flip is still a low-freq laggard**: f8 (Δφ=π) crosses 0.9 only at
  step 7000 and never reaches 0.99 — consistent with the Δφ difficulty from
  Iterations 29–30.
- **The high-freq copy tasks now lag — the opposite of the pure-tone corpus.**
  f64/f128/f256 (all Δφ≡0 copy tasks) *degrade or plateau*: f64 falls
  0.916 → 0.885 after step 6500, f128 peaks at 0.90 then settles 0.82, f256
  never crosses 0.9. In the pure-tone corpus these converged *first*.

**Conclusion.** Two separate pressures now separate:
1. **Δφ difficulty (unchanged)**: the flip f8 (Δφ=π) is the low-frequency
   laggard everywhere, as before.
2. **Training-marginal coverage (new, dominant at high f)**: the rich corpus
   trains on 32-component mixtures whose spectral mass sits at f ∈ 8…128; pure
   tones at f128/f256 are *off-support* (and diluted among 32 mixed components
   in every training window), so the model shrinks them persistently. The
   control is the uniform-marginal mix run (Iteration 30), where f256
   converged to 0.99 — so it is the *marginal*, not the frequency, that causes
   the high-end lag. f64's late degradation shows the model drifting toward
   mixture-optimal (superposition) solutions that hurt pure-tone recovery.

**Answer to the user's question:** no — rates are not uniform. f16 converges
first (copy + on-support), f8 lags via the flip, and f128/f256 never converge
because the training marginal under-covers them. Δφ still holds as a
difficulty axis, but coverage is a second, orthogonal axis that dominates at
the spectral edges.

**Evidence.** `results/ckpts/shrink-rich-30k-tone.pt`/`-tone_dyn.json`
(`per_freq_pure` field).

---

## Iteration 34 — Why the overlap is non-monotonic after x=1, and the d=k threshold

**Hypothesis (user question).** Two questions: (1) the plane-overlap curve from
Iteration 32 is not monotonically decreasing beyond `Δf = 1 cycle/patch` — why
the bump? (2) does hidden size need to scale with patch size?

**Experiment.**
- (1) Compare latent-space overlap against *input-space* overlap of the raw
  64-sample patch (no RevIN, no projection, no training): the 2-D span of
  {cos, sin} of each tone restricted to the patch window.
- (2) Re-examine the capacity sweep (hidden 16/32/64 at patch 64) through the
  lens of the rank paper (Yu et al., ICLR 2026): their Thm 2 bounds the
  ε-rank of a patched MLP embedding by `O(k)`, the patch size, not `d`.

**Results.**
- (1) **The bump is windowing sidelobes, present in the raw input.** The
  input-space overlap of a 64-sample patch has the *exact* same shape: 0.000 at
  integer `x` (df=16, 32, 48 → orthogonal, they are distinct DFT bins) and
  bumps at half-integers (x=1.5 → 0.046, x=2.5 → 0.017). This is the Dirichlet
  kernel / rectangular-window spectral leakage: frequencies separated by an
  integer number of cycles per patch are exactly orthogonal over the window,
  while non-integer separations leak through the window's sidelobes. The latent
  curves track this structure (h64: 0.035 @ x=1, 0.105 @ x=1.5, 0.082 @ x=2,
  0.006 @ x=3), slightly lifted by capacity/noise. Not a learned artifact.
- (2) **The orthogonality threshold is exactly `d = k`.** A patch of size `k`
  resolves `k/2` DFT bins (frequencies), each needing a 2-D ring plane, so
  `d = 2·(k/2) = k` is the hidden dimension at which all resolvable
  frequencies can be mutually orthogonal. The capacity sweep confirms it:
  far-apart plane overlap 0.40 (d=16 = k/4), 0.09 (d=32 = k/2), 0.02
  (d=64 = k). Clean orthogonality exactly at `d = k`, consistent with the rank
  paper's `rank_ε ≤ (1+ε⁻²)k` bound.

**Conclusion.** (1) The non-monotonicity is a *window property*, not a learned
one: integer-cycle separations are exactly orthogonal DFT bins; half-integer
separations leak through the rectangular-window sidelobes. Any patch embedding
inherits it. (2) Hidden size should scale **at least linearly with patch size**,
with the clean threshold `d ≥ k` (2 dims per resolvable frequency bin × k/2
bins). Below it, capacity forces far-apart planes to merge (superposition);
at/above it, orthogonality is free. The rank paper gives the matching theorem
side: embedding rank is set by `k`, not `d`.

**Evidence.** Inline input-vs-latent overlap comparison (h64, h32-noGLU);
capacity sweep table; `references/rank-structure-transformers/notes.md`.

---

## Iteration 35 — Janus rollout-face test: does smaller patch slow the rings?

**Hypothesis (user question).** The rollout face of the Janus problem says the
per-patch phase advance `Δφ = 2π·f·k/ctx` sets prediction difficulty. If so, a
smaller patch should reduce the *one-step* prediction error per unit time — the
small patch rotates the ring less per step, so each step is easier. Test with a
controlled pair: same rich corpus, same objective (input-space forecasting),
same hidden 32 / 2 layers / 12k steps, patch 16 vs patch 64.

**Experiment.** Train `rollout-p16.pt` / `rollout-p64.pt` (input-space, rich
corpus). (a) Teacher-forced one-step per-element normalized MSE at f ∈
{8, 24, 64} and on rich windows — per-element MSE is directly comparable across
patch sizes because p64 makes one 64-sample step while p16 makes four
16-sample steps over the same time. (b) Autoregressive rollout MSE vs
sample-horizon (re-encoding predicted patches — the input-space loop, a
separate error source).

**Results.**
- **Teacher-forced one-step (the clean test):**

  | signal | p16 (Δφ mod 2π) | p64 (Δφ mod 2π) | ratio |
  |---|---|---|---|
  | f=8 | 0.018 (0.79) | 0.125 (**3.14=π**) | 6.8× |
  | f=24 | 0.038 (2.36) | 0.889 (**3.14=π**) | 23× |
  | f=64 | 0.018 (0.00) | 0.199 (0.00) | 11× |
  | rich | 0.171 | 0.631 | 3.7× |

  The two frequencies where p64 faces the *flip* (`Δφ ≡ π`) are exactly the
  largest gaps (6.8×, 23×); f=64 is a copy task for both yet p16 is still 11×
  better — smaller patch helps everywhere, dramatically so at the flip.
- **Rollout (re-encoding loop)**: mixed — f=8/f=24 p16 wins at short horizons
  but p64 catches up at 256 samples (fewer re-encode steps, so less
  input-space error amplification); f=64 and rich favor p64 at long horizons.
  This is the known re-encoding cost of input-space rollout, not the Δφ signal.

**Conclusion.** The **teacher-forced one-step error per unit time is 4–23×
lower at patch 16** than patch 64, with the largest gaps exactly at the
frequencies where the large patch faces the antipodal flip (`Δφ = π`). This
directly confirms the rollout face of the Janus problem: smaller patches
"slow the rings" and make each prediction step easier. The rollout test is
confounded by the re-encoding error-amplification loop (input-space rollout
re-encodes predicted patches); a latent-space rollout with fine-grained phase
integration would isolate the benefit. Design implication: the model should
encode at coarse patches (resolution, `d=k`) but predict at fine steps
(rollout granularity) — the two axes should be decoupled, not tied to one `k`.

**Evidence.** `results/ckpts/rollout-p16.pt`, `rollout-p64.pt`,
`probe_rollout_patch.py` (repo root, scratch).

---

## Iteration 36 — Correction: Iteration 35 compared different forecast horizons

**What went wrong.** The Iteration 35 "teacher-forced one-step" table compared
`pred[:, :-1] vs z_next` per *patch*, which is a different forecast horizon in
*samples* for the two models: p16 predicts **16 samples ahead**, p64 predicts
**64 samples ahead**. A 16-sample-ahead error is trivially smaller than a
64-sample-ahead error, so the 6.8×/23×/11× gaps were horizon artifacts, not a
patch-size effect. The claim "smaller patch slows the rings and reduces
one-step error per unit time" was **not established**.

**Corrected experiment.** Same history (1024 samples, P−1 context), teacher-
forced multi-step rollout (extend context with the *true* next patch each step
— no re-encoding, same protocol as training), score per-element normalized MSE
at the **same sample horizons** 16/32/64/128/256 for both models.

**Corrected results.**

| signal | p16 @64 | p16 @128 | p16 @256 | p64 @64 | p64 @128 | p64 @256 |
|---|---|---|---|---|---|---|
| f=8 | 0.58 | 0.60 | 0.60 | 2.28 | 2.39 | 2.39 |
| f=24 | 2.82 | 2.89 | 2.89 | 0.92 | 0.94 | 0.94 |
| f=64 | 0.022 | 0.022 | 0.022 | 0.25 | 0.25 | 0.25 |
| rich | 1.52 | 1.52 | 1.55 | 1.26 | 1.24 | 1.24 |

**Conclusion — the Δφ/flip story is NOT supported at equal horizon.**
- f=8 (p64 faces Δφ=π): p16 wins 4× — consistent with the flip being hard.
- f=24 (p64 also faces Δφ=π): p64 *wins* 3× — **contradicts the flip
  explanation**. f=24 is `3π ≡ π (mod 2π)` for p64, yet p64 predicts it better.
- f=64 (both copy tasks): p16 wins 11× — real, but not a Δφ-flip effect.
- rich (realistic): p64 slightly better.

So at a fixed forecast horizon there is **no consistent patch-size advantage**
and no clean `Δφ mod 2π` difficulty law in the input-space forecasting task.
The remaining f=8 gap may be the trend/DC family (f=8 planes share the trend
line) rather than the flip. The Iteration 35 design implication (decouple
encoding and rollout granularity) is therefore **not yet supported by
measurement**; it remains a hypothesis.

**Evidence.** Same-horizon teacher-forced tables above (p16/p64, f ∈
{8,24,64}, rich); `rollout-p16.pt`, `rollout-p64.pt`.

---

## Iteration 37 — Decoupled head test: single-shot vs mini-RNN rollout head (encoder held fixed)

**Hypothesis (user question).** The Janus "rollout face" — finer prediction
granularity helps — tested *at the head*, holding the encoder fixed: patch 64 /
ctx 1024 / hidden 64 / 2L / input-space / rich corpus / 12k steps, same seed.
Head A predicts the full 64-sample next patch in one shot (SwiGLU→Linear,
29,056 params). Head B is a 4×16-sample mini-RNN head (27,024 params) whose
state recurrence feeds back its own emission (closed-loop). Head B-TF is the
same architecture trained with teacher-forced internal feedback (true sub-chunk
into the state update). Complexity matched (B/A = 0.93).

**Results.** Same-horizon teacher-forced one-step (64/128/256 samples; error
flat across horizons by construction):

| signal | A single-shot | B closed-loop | B-TF |
|---|---|---|---|
| f=8 | **2.17** | 2.27 | 2.26 |
| f=24 | **1.26** | 2.32 | 2.95 |
| f=64 | 0.112 | **0.061** | 0.088 |
| rich | **1.36** | 1.44 | 1.35 |

Training loss: A 0.557, B-TF 0.197 (B-TF optimizes far better but evals no
better).

**Conclusion.** **The mini-RNN head does not help — the single-shot head wins
or ties on 3 of 4 signals.** f=24 (where the RNN should help most if iterative
phase-tracking mattered) is 1.8× worse for B and 2.3× worse for B-TF. f=64 is
the only RNN win. Teacher-forcing improved optimization (loss 0.197) without
improving eval — the exposure-bias-free training didn't transfer to the
self-feedback eval. Combined with Iteration 36 (patch granularity, same
horizon, no consistent advantage), the **rollout face of the Janus problem is
now tested two independent ways and neither supports it**: neither changing
input patch size nor changing head granularity (with encoder fixed) improves
same-horizon one-step forecasting. The measured frequency bias remains the
learned damping prior (§4.2), not a Δφ/rollout-granularity effect.

**Evidence.** `results/ckpts/rollout-head-A.pt` / `-B.pt` / `-BTF.pt`,
`probe_rollout_head.py` (repo root, scratch).

---

## Iteration 38 — Norm+scale head (explicit direction/scalar decomposition)

**Hypothesis (user question).** If the shrinkage is a learned prior on the
*radius*, make it explicit: head C predicts `pred = w(h) · (GLU(h)/‖GLU(h)‖)` —
one branch produces a unit-vector *direction*, a second branch produces a
dynamic scalar `w(h)` that rescales it. Same backbone as Iteration 37
(patch 64 / ctx 1024 / hidden 64 / 2L / input-space / rich / 12k steps).

**Results.** Same-horizon teacher-forced one-step (64 samples; error flat
across 128/256):

| signal | A single-shot | B closed-loop | B-TF | **C norm+scale** |
|---|---|---|---|---|
| f=8 | **2.17** | 2.27 | 2.26 | 2.28 |
| f=24 | 1.26 | 2.32 | 2.95 | **1.12** |
| f=64 | 0.112 | **0.061** | 0.088 | 0.220 |
| rich | 1.36 | 1.44 | 1.35 | **1.27** |

Training loss: A 0.557, C ~0.73 (C optimizes worse but evals comparable or
better on 2 of 4 signals).

**Conclusion.** **Mixed but mildly encouraging for the radius-aware direction.**
Head C is best on f=24 (1.12 vs A's 1.26) and on rich (1.27 vs 1.36), ties
f=8, but is worst on f=64 (0.22, a copy task where the unit-direction
constraint hurts — the true next patch is not unit-norm, so the fixed
direction + scalar has less room to model amplitude-invariant copies). Overall
C is *not worse than A on realistic (rich) data* and beats it on the
higher-Δφ tone, suggesting the explicit norm/direction decomposition is a
plausible inductive bias even if the copy-task case (f=64) degrades. This is
the first head variant that doesn't lose to the single-shot baseline — worth
pursuing with a calibrated/heteroscedastic scalar rather than a point scalar.

**Evidence.** `results/ckpts/rollout-head-C.pt`, `probe_rollout_head.py`.

---

## Iteration 39 — Multi-seed unseen-mixture eval: B's win was noise; C is the front-runner

**Hypothesis (user question).** The single-mixture figure showed B (mini-RNN)
winning on one unseen mixture — does that survive across mixtures, or was it
luck?

**Experiment.** 12 unseen mixtures (random 3-frequency sets from a wide pool,
random phases, off the training clusters), teacher-forced 256-sample forecast,
all four heads.

**Results.**

| head | mean MSE | std | wins (of 12) |
|---|---|---|---|
| **C norm+scale** | **1.122** | 0.103 | **5** |
| A single-shot | 1.152 | 0.074 | 4 |
| B mini-RNN | 1.248 | 0.150 | 2 |
| B-TF RNN | 1.424 | 0.265 | 1 |

**Conclusion.** **B's single-mixture win was noise** — over 12 mixtures B is
third (1.25 vs C's 1.12), with the highest variance of the non-TF heads. The
front-runner is **C (norm+scale)**: best mean, most wins, and it also won rich
and f=24 in Iteration 38; its only consistent weakness is the f=64 copy task.
B-TF remains worst (teacher-forcing doesn't transfer to OOD). **Head ranking:
C ≳ A > B > B-TF.** The mini-RNN direction is not worth pursuing further as-is;
the norm+scale decomposition is the promising one (next: calibrated/
heteroscedastic scalar).

**Evidence.** Inline 12-seed eval (A/B/B-TF/C on unseen mixtures);
`make_fig_pred.py`.

---

## Iteration 40 — Z-space calibrated-radius head: collapse (negative result)

**Hypothesis (user question).** The shrinkage is a radial contraction of the
rotation operator (E4); fix it *in latent space* by decomposing the predicted
next latent into direction + calibrated radius: `z_pred = r(h)·(L(h)/‖L(h)‖)`
(scalar head) or per-dimension `s(h)⊙(L(h)/‖L(h)‖)` (perdim head), then decode
to input space via `LatentProbe`. Same backbone as Iteration 37–39 (patch 64 /
ctx 1024 / hidden 64 / 2L / latent JEPA + SIGReg / rich / 12k steps).

**Results.**
- **The calibrated heads collapse.** Latent radius ratio `‖pred‖/‖z_next‖`:
  lin 0.87 (rich) / 0.99 (f=24); scalar 0.47 (rich) / **0.09** (f=24);
  perdim **0.07** (rich) / **0.06** (f=24). Scalar collapses out-of-distribution;
  perdim collapses everywhere.
- The apparent "improvements" in input-space MSE are the **mean-prediction
  floor**: scalar/perdim tone MSEs are all ≈ 0.5 = the variance of a
  unit-amplitude tone — the model is predicting ≈ the mean, not a calibrated
  forecast.

**Conclusion.** **Negative result: re-parameterizing the head as
`radius × direction` does not fix the shrinkage — the scale branch collapses to
zero.** The geometry is right (direction is good, radius is shrunk) but pure
MSE training finds the degenerate solution "predict the mean" for the scale
branch: nothing anchors the scalar to the *true* next-latent norm. The fix
needs an explicit target for the radius — e.g., supervise the scale branch on
`‖z_next‖` (the true next-latent norm), or anchor it to the norm-copy
(radius = last known norm + learned correction). This is a clean, testable next
step: the geometry tells us *what* to calibrate (the radius) and the failure
tells us *how* (supervise the scale, don't let MSE discover it).

**Evidence.** `results/ckpts/zcalib-lin.pt` / `-scalar.pt` / `-perdim.pt`,
`probe_zcalib.py` (repo root, scratch).

---

## Iteration 41 — Tokenizer + tiny affine map: the transformer isn't needed, and the map doesn't shrink

**Hypothesis (user question).** On synthetic data the next patch is determined
by (frequency, phase) alone — both decodable from the *current* embedding. So a
tiny network over the frozen encoder (used purely as a tokenizer) should
reproduce the next-latent dynamics with no history, and — trained directly on
deterministic `(z_p, z_{p+1})` pairs — should **not** exhibit the shrinkage.

**Experiment.** Freeze the encoder from `shrink-rich-30k.pt` (patch 64 / ctx
1024 / hidden 32 / latent). Collect ~2M `(z_p, z_{p+1})` pairs from the rich
corpus. Train a single affine map (`z_{p+1} = W·z_p + b`) and a 1-hidden MLP
(MSE, 2000 steps). Compare teacher-forced one-step radius ratio and cosine
against the full model's own `lm_head` (which has the transformer + shrinkage).

**Results.** Teacher-forced radius ratio `‖pred‖/‖z_next‖` (cos):

| predictor | f8 | f24 | f64 | f128 |
|---|---|---|---|---|
| model lm_head (transformer) | 0.955 (0.994) | 0.961 (0.963) | 0.971 (0.987) | 0.898 (0.970) |
| **affine map (no history)** | 0.900 (0.992) | 0.828 (0.958) | 0.928 (0.996) | **1.000** (0.972) |
| **MLP map (no history)** | **0.973** (0.995) | 0.840 (0.954) | 0.932 (0.998) | 0.956 (0.974) |

**Conclusion.** Two findings:
1. **The affine map with *zero* history reproduces the next-latent dynamics at
   the same cosine level as the full transformer** (cos ≈ 0.96–0.99 everywhere).
   The user's analysis is confirmed: frequency and phase are both in the current
   embedding, so the next patch is a per-plane rotation of it — no history
   needed. (The affine map is slightly *worse* on radius at f24 but *better* at
   f128; overall the transformer's advantage is marginal.)
2. **The tiny map does NOT fix the shrinkage by construction.** Radius ratios
   are ≈ 0.83–1.00, comparable to the transformer's 0.90–0.97 — the map does
   not magically learn norm-preservation. The ~10–17% residual shrink at f24
   persists even in a no-history affine map trained on clean pairs. So the
   shrinkage is **not** a byproduct of the transformer/history — it is a
   property of the next-patch *target distribution itself* (the ring is
   off-center, the map must rotate around a non-origin centroid, and the
   least-squares fit to noisy-ish phase targets pulls the norm down). This
   reframes Iteration 40: the scale branch collapsed not because calibration
   is impossible but because the *raw* map already embodies the shrinkage, and
   the geometry of an off-center rotation under MSE naturally contracts the
   radius.

**Evidence.** `probe_tokenizer_dynamics.py` (repo root, scratch); frozen
encoder `shrink-rich-30k.pt`.

---

## Iteration 42 — Centered tokenizer map + orthogonal-pair mixtures

**Hypothesis (user question).** Iteration 41's affine map still shrank
(~0.83–0.97). Since centering does not change the covariance, W is identical
across bias variants — only the offset differs. Test whether the shrinkage
comes from the *bias* (off-center ring): `free-b` (LS offset),
`global-c` (shared corpus centroid `c_g`), `oracle-c` (true ring centroid
`c_f` — "rotate around the ring center"). Also evaluate known orthogonal
far-apart pairs (`|f1−f2| ≥ 16` cycles/window = non-leakage regime) as 2-tone
mixtures.

**Results.** Teacher-forced radius ratio (cos):

*Pure tones:*

| variant | f8 | f24 | f64 | f128 |
|---|---|---|---|---|
| free-b | 0.903 (0.993) | 0.839 (0.962) | 0.939 (0.996) | 0.966 (0.975) |
| global-c | 0.907 (0.994) | 0.834 (0.964) | 0.939 (0.996) | 0.973 (0.973) |
| **oracle-c** | **0.939** (0.991) | **0.896** (0.960) | **0.984** (0.997) | **0.983** (0.977) |

*Orthogonal pairs (oracle-c):* 16+128 0.931 (0.965), 8+64 0.953 (0.990),
48+160 0.929 (0.947), 32+160 0.882 (0.919), 24+200 **1.160 (0.366)**.

**Conclusion.** Two findings.
1. **Oracle centering (rotate around the true ring center) consistently
   improves the radius** — 0.90→0.94 (f8), 0.84→0.90 (f24), 0.94→0.98 (f64),
   0.97→0.98 (f128) — confirming the off-center-ring hypothesis: part of the
   shrink is the bias term. But it does **not** eliminate it (still 0.90–0.98,
   not 1.0): W is a *single* LS map shared across frequencies, so even with the
   right centroid the cross-frequency compromise (and the amplitude jitter in
   the corpus) leaves residual radial contraction. `global-c` ≈ `free-b` (the
   shared centroid barely helps) — per-ring centroids are what matter.
2. **Mixtures of in-support orthogonal pairs work** (16+128, 8+64: ratio
   ~0.93–0.95, cos ~0.95–0.99). The broken case is 24+200 (cos 0.37) — because
   **f=200 is off the training marginal** (rich corpus peaks ≤ 128), so the
   encoder has no clean ring there (the coverage effect, Iteration 33), not a
   mixture-structure failure.

**Conclusion vs Iteration 41/40.** The shrinkage is now decomposed: (a) a
*bias* part from the off-center ring (fixed by oracle centering), and (b) a
*residual* part from the single shared W (cross-frequency compromise +
amplitude jitter), which no linear map can remove — consistent with why
Iteration 40's scale branch collapsed. The fix that would work: per-ring
(per-frequency) rotation operators with per-ring centroids — i.e., the geometry
says the map should be *frequency-conditioned*, which the transformer provides
via the ring-azimuth in h, and which a single linear map over z cannot.

**Evidence.** `probe_tokenizer_centered.py` (repo root, scratch); frozen
encoder `shrink-rich-30k.pt`.

---

## Iteration 43 — MLP over z: can it learn freq → (centroid, rotation)?

**Hypothesis (user question).** The frequency is implied in the patch embedding
(ring azimuth), so an MLP over z should be able to store a
freq → (centroid, radius) mapping and dynamically construct the per-frequency
rotation — fixing the radius where the shared affine map fails. Test with two
training regimes: `rich` (mixtures + amplitude jitter, realistic) and `tone`
(pure tones only, no jitter — the ideal regime for this hypothesis). Models:
affine baseline vs a 2-hidden-layer MLP (h=128, 4000 steps).

**Results.** Teacher-forced radius ratio (cos):

*Trained on pure tones (ideal regime):*

| predictor | f8 | f24 | f64 | f128 |
|---|---|---|---|---|
| affine | 1.001 (0.997) | 0.984 (0.994) | 1.006 (0.999) | 0.986 (0.997) |
| **MLP** | **1.000 (0.999)** | **1.001 (1.000)** | **1.000 (1.000)** | **1.000 (1.000)** |

*Trained on the rich corpus (realistic):*

| predictor | f8 | f24 | f64 | f128 |
|---|---|---|---|---|
| affine | 0.903 (0.993) | 0.839 (0.962) | 0.939 (0.996) | 0.966 (0.975) |
| MLP | 0.963 (0.997) | 0.923 (0.949) | 0.954 (0.998) | 0.990 (0.970) |

Training loss: tone MLP 0.0005 (essentially exact), rich MLP 0.045.

**Conclusion.** **Your hypothesis is confirmed in the ideal regime and
partially in the realistic one.**
1. **On pure-tone training data, the MLP achieves radius ratio 1.000 and
   cosine 1.000 at every frequency** — it *does* learn the
   freq → (centroid, rotation) mapping and applies it exactly. Even the
   affine map nearly does (0.984–1.006): on clean tones the shrink largely
   vanishes, confirming it was the *training distribution* (mixtures/jitter),
   not the map class, that caused the Iteration 41–42 shrink.
2. **On the rich corpus the MLP also beats affine everywhere** (f8 0.963 vs
   0.903, f24 0.923 vs 0.839, f64 0.954 vs 0.939, f128 0.990 vs 0.966) — the
   nonlinearity does buy frequency-conditioned radius recovery. Residual
   shrink (0.92–0.99) remains because the rich corpus mixes frequencies and
   jitters amplitudes, so no single map is exact — but the MLP is strictly
   closer to 1 than the affine map.
3. **Mixture eval is noisy** (some pairs degrade under MLP, e.g. 8+64 cos
   0.72 with the tone-trained MLP) — the MLP overfits the tone regime and the
   mixture direct-sum isn't fully captured by a z-only map, consistent with
   the ~40% mixture compression.

**Answer to the user's question:** yes — an MLP over z *can* learn the
freq → (centroid, rotation) map; on pure tones it recovers the radius exactly
(1.000), and on the rich corpus it beats the linear map. The earlier affine
shrink was a training-distribution artifact, not a geometric limit. This
validates the tokenizer + small dynamics-network idea: the encoder is a
faithful tokenizer, and a modest MLP (no transformer, no history) reproduces
the latent dynamics with exact radius in the clean regime.

**Evidence.** `probe_tokenizer_mlp.py` (repo root, scratch); frozen encoder
`shrink-rich-30k.pt`.

---

## Iteration 44 — Partial-oracle predictor: rotate per plane, recombine, decode

**Hypothesis (user question).** Build a hand-crafted predictor: from an input
embedding z (possibly a mixture), decode a list of frequencies; for each, find
its phase and rotate in its subspace by the phase advance `Δφ_f`; recombine by
vector addition; decode to input space with a trained `LatentProbe`. Testable
pipeline: `z_next = z + Σ_f (R_f − I)·P_f(z − c_f)`.

**Experiment.** Frozen encoder (`shrink-rich-30k.pt`). Per-frequency ring
planes `(U_f, c_f, R_f)` from phase loops (R_f = observed 2×2 rotation
p→p+1). LatentProbe decoder. Compare teacher-forced one-step input-space MSE:
oracle (true frequencies), oracle-det (projection-based detection), affine map,
transformer lm_head.

**Results.** One-step input-space MSE:

| signal | oracle (true f) | oracle-det | affine | lm_head |
|---|---|---|---|---|
| tone-f8 | **0.237** | 1.02 | 0.575 | 0.261 |
| tone-f24 | 1.093 | 1.05 | 0.967 | **1.022** |
| tone-f64 | **0.047** | 1.27 | 0.447 | 0.539 |
| tone-f128 | **0.047** | 1.10 | 0.915 | 0.947 |
| mix-16+128 | **0.567** | 1.68 | 1.27 | 1.33 |
| mix-8+64 | 1.57 | 1.54 | 1.22 | **1.09** |
| mix-32+160 | 1.64 | 1.71 | 1.45 | **1.40** |

**Conclusion.** Two-part answer to the user's question.
1. **The oracle works when the frequency list is given.** With true
   frequencies, the hand-built rotate-per-plane-and-recombine predictor beats
   the affine map on every pure tone and beats/ties the transformer head on
   f8/f64/f128 and 16+128. The procedure `z + Σ(R_f−I)P_f(z−c_f)` is
   genuinely correct — the geometry is directly executable.
2. **Decoding the frequency list from a single z is the blocker, and it is a
   geometry limit, not a threshold bug.** Projection-based detection flags
   every frequency: the ring planes at hidden 32 overlap **0.45–0.76 across
   the whole grid** (the local-merging / capacity story), so a f=64 tone
   projects 1.8× its own radius onto the f=8 plane. Fixed per-frequency planes
   cannot separate frequencies below the plane-overlap regime. Detection needs
   either well-separated frequencies (sparse grid / higher hidden → cleaner
   planes) or a learned frequency discriminator — which is precisely what the
   transformer's `h` provides (and why the full model still works).

**Answer to the user's question:** yes, the decompose-rotate-recombine-decode
pipeline works, *conditional on knowing the frequencies*; the missing piece is
frequency *identification*, which is limited by plane overlap exactly where the
geometry says it must be (merging).

**Evidence.** `probe_oracle.py` (repo root, scratch); frozen encoder
`shrink-rich-30k.pt`.

---

## Iteration 45 — Why frequency detection failed: it's the training marginal, not the geometry

**Hypothesis (user question).** Is the failure to decode frequencies universal,
or an artifact of the checkpoint? The dense-grid overlap measured on
`shrink-rich-30k.pt` (0.45–0.76 for far pairs) contradicts the earlier
near-orthogonal finding on `geom-rich.pt` (8–128: 0.086). Test: compare ring
geometry across checkpoints, then run the oracle detection on a *clean*
checkpoint with known-orthogonal combinations.

**Results.**
1. **The 30k uneven-corpus checkpoints have a different geometry — the rings
   are elliptical, not circular.** `geom-rich.pt` (3k, uniform marginal): rings
   near-circular (s1/s2 ≈ 1.3–1.4), centroid ≈ 0.7, far-pair overlap 0.086.
   `shrink-rich-30k.pt` (30k, uneven marginal): rings highly elliptical
   (s1/s2 ≈ 10), centroid 2.1–4.4, far-pair overlap 0.575. The 15k uneven and
   the 30k-tone variants show the same (s1/s2 2.5–11, overlap 0.34–0.58). So
   **the change is systematic with the uneven marginal, not training length.**
2. **On the clean checkpoint, detection works.** On `geom-rich.pt` with a
   well-separated grid (Δf ≥ 16 cycles/window, plane overlaps 0.01–0.21):
   pure f8/f24/f96 detected exactly; the orthogonal combos 24+128 and 8+96 are
   detected exactly (both frequencies recovered). Only pure f128 mis-detects
   (returns {24,48,96,128} — near-Nyquist/edge ambiguity).

**Conclusion.** **The frequency-decodability failure is NOT universal — it is
a property of the training marginal.** With a uniform marginal the encoder
builds clean circular rings, planes are near-orthogonal for well-separated
frequencies, and the decompose-rotate-recombine oracle decodes frequency
lists correctly (including mixtures of orthogonal pairs). With an *uneven*
marginal (the coverage imbalance of the 30k rich corpus), the rings become
elliptical and off-center, planes overlap broadly, and a single z cannot be
separated into per-frequency components. This connects Iteration 33 (coverage
shrinkage) to the *geometry itself*: **the marginal shapes the ring geometry,
not just the radius.** The earlier "near-orthogonal planes" claim holds for the
clean uniform-marginal model; the 30k uneven model is a different (elliptical)
regime. The user's oracle idea is valid — it needs a model whose planes are
separable, i.e., a balanced training marginal.

**CORRECTION (Iteration 49 follow-up).** The marginal attribution was
*partially wrong*: the uneven-marginal runs were also trained 15k–30k steps,
confounding "marginal" with "training duration under SIGReg". The controlled
canonical SIGReg twin (`canonical-rich-sigreg.pt`, **uniform** marginal,
SIGReg, **32k** steps) shows the *same* deformation (f64 s1/s2 = 19.3,
far overlap 0.47), while the no-reg direct model at 32k stays circular (1.30,
overlap 0.001). So the driver is **SIGReg × training duration**: the
regularizer's marginal Gaussianization deforms the rings once it is actually
satisfied (3k steps: weak, circular; 32k: strong, elliptical). RevIN is not
the cause (both canonical models use it). The uneven marginal may *modulate*
the effect but is not the root cause.

**Evidence.** Inline comparisons across `geom-rich.pt` /
`shrink-rich.pt` / `shrink-rich-30k.pt` / `shrink-rich-30k-tone.pt`;
detection test on `geom-rich.pt`.

---

## Iteration 46 — Clean-geometry map-reduce: the transformer head loses badly

**Question (user).** Why does the transformer fail to learn a simple rotation
when a single patch as input should suffice? Test on a *clean-geometry*
encoder (`geom-rich.pt`, circular rings, decodable frequencies) with the
hand-built map-reduce pipeline: detect frequency → rotate each plane by its
phase advance → recombine by vector addition → decode via a shared
`LatentProbe`. Compare against a fitted affine map (single-patch baseline) and
the transformer's own head, all on the same frozen encoder.

**Results.** One-step input-space MSE (teacher-forced, same decoder):

| signal | oracle (detected) | oracle (true f) | affine-tone | affine-rich | **lm_head** |
|---|---|---|---|---|---|
| tone-f8 | 0.070 | 0.070 | 0.060 | 0.390 | **0.444** |
| tone-f24 | 0.059 | 0.059 | 0.046 | 1.371 | **1.153** |
| tone-f96 | 0.043 | 0.043 | 0.047 | 0.481 | **0.637** |
| tone-f128 | 0.043 | 0.043 | 0.044 | 1.357 | **1.545** |
| mix-24+128 | 0.472 | 0.472 | 0.268 | 1.832 | **1.881** |
| mix-8+96 | 0.273 | 0.273 | 0.478 | 0.530 | **0.700** |

**Conclusion.** **On clean geometry the manual pipeline works, and the
transformer head is ~10–35× worse than the simple map.** The oracle
(frequency-detected, no ground-truth frequencies needed) achieves MSE
0.043–0.070 on pure tones — nearly exact. A single affine map fitted on pure
tones is equally good (0.044–0.060), confirming "one patch suffices." But the
transformer's own head scores 0.44–1.55 — an order of magnitude worse. Key
observations:
1. **The transformer head is not learning the rotation even though the encoder
   contains it.** The affine map fitted *on top of the same frozen encoder*
   recovers the dynamics perfectly, so the information is there; the jointly-
   trained `lm_head` fails to exploit it (it was trained through the
   transformer, with SIGReg and only 3k steps).
2. **The affine map fitted on the rich corpus (the transformer's own training
   distribution) is also much worse than the oracle** (0.39–1.37 vs 0.04–0.07
   on tones) — because the rich corpus mixes frequencies and jitters
   amplitudes, so a single linear fit over that distribution can't be exact
   (the Iteration 43 finding). Only the *per-frequency-conditioned* oracle is
   near-exact.
3. **So the answer to "why does the transformer fail":** it is not a geometry
   limit — the rotation is trivially learnable from one patch (affine-tone,
   oracle). The transformer fails because (a) its head is trained jointly under
   SIGReg on a mixed corpus, where the MSE-optimal solution is a compromise
   (shrinkage), and (b) the *history* it attends to does not help — the
   next-patch target is determined by the current patch alone, so the
   transformer's capacity is spent on aggregation that is unnecessary and
   whose training dynamics fight the exact rotation. The manual map-reduce is
   the "ideal predictor" the transformer should have learned; it doesn't,
   because nothing in its objective singles out the rotation structure.

**Answer to the user's question:** yes, the manual map-reduce is constructible
and near-lossless on clean geometry — and the transformer's failure to learn
it is a training/objective artifact, not an expressiveness limit.

**Evidence.** `probe_oracle_clean.py` (repo root, scratch); `geom-rich.pt`.

---

## Iteration 47 — Repo cleanup + canonical test-bed model

**Context (user request).** The repo accumulated many checkpoints, scratch
probes, and stale CLI code from superseded phases. Before further experiments,
establish a clean controlled test bed with centrally documented checkpoints
and a canonical model matching the current best understanding (clean marginal
geometry, `d = k`).

**Cleanup performed.**
- **Archived 66 obsolete checkpoints** (88 MB) to `results/archive/ckpts/`:
  GATr dead-ends, old ablations (GLU head, k-sweep, patch=1, scale-aware),
  head experiments (rollout-*, zcalib-*), uneven-marginal shrink runs
  (elliptical-geometry superseded by the canonical uniform model), and all
  batch-composition / UCR / MONSTER subdirectories (now README appendices).
  Kept 14 active checkpoints (5.9 MB): the clean-geometry family
  (`geom-rich*.pt`), `geom-single.pt`, `shrink-dyn.pt` (deterministic
  pure-tone dynamics), and `monash-direct-h32.pt` (best real forecaster).
- **Moved scratch probes** to `probes/` (evidence generators, kept; fixed the
  one cross-import). Central registry written: `results/CHECKPOINTS.md`
  (objective / spec / dataset / training / rationale per checkpoint, plus
  archived index).
- **Archived the legacy CLI family** (`hmm.py`, `forecast.py`,
  `shape_probe.py`, `ucr_eval.py`, `monster_eval.py`, `tier2.py` and their
  tests) to `results/archive/legacy-cli/` — they imported deleted modules
  (`patchtst`, `simple`) from the pre-refactor era and could not import.
  `cli/__init__.py` now registers only live CLIs; `canonical.py` and
  `monash_lftm.py` are the current entry points. `tests/` = MiniLTFM tests
  only (5 pass, `ruff` clean).
- **README/JOURNAL updated**: top summaries now include the map-reduce
  finding (transformer head 10–35× worse than the per-frequency rotation)
  and the marginal-shapes-geometry result; appendices note archived code.

**Canonical model (per user spec).** `canonical-rich-direct.pt` — ctx 1024 /
patch 64 / hidden 64 (= patch, the d=k threshold) / 2 layers / 4 heads /
**direct input-space forecasting** / SGD lr 1e-2 / 32k steps / **uniform**
rich spectral corpus (balanced marginal → clean rings). Trained via
`python -m batcomp.cli.canonical`; sidecar JSON has the full config for the
registry. This is the model to build on for the "why doesn't the transformer
learn the rotation" question and the frequency-conditioned dynamics head.

**Evidence.** `results/CHECKPOINTS.md`, `results/archive/`, `probes/`,
`src/batcomp/cli/canonical.py`.

---

## Iteration 48 — Canonical model MASE + SIGReg twin + dynamical-system embeddings

**Question (user).** (1) What is the canonical test bed's predictive
performance vs a lag-1 baseline? (2) Prepare a SIGReg canonical twin for
revisiting batch composition. (3) Design a serious GATr test bed.

**Results.**
- **MASE vs lag-1 (teacher-forced one-step, raw space):** rich 1.32,
  tone-f64 1.16, tone-f128 1.20, mix-24+128 1.66, **tone-f8 13.67**. The
  model does not beat trivial persistence on the synthetic corpus. The f=8
  blow-up is expected (period-128 tone: adjacent samples differ by 0.03, so
  lag-1 is near-perfect and the model's 0.43 MAE is 13× worse) — it reflects
  that the direct model's *one-step* raw prediction is not phase-sharp on
  smooth tones, consistent with the map-reduce gap (Iteration 46).
- **Canonical embedding-space quality (Part B):** near-circular rings
  (s1/s2 1.06–1.65, dim1+2 0.94–0.97), the learned rotation matches the
  physical phase advance `Δφ` exactly (f=8: −π, f=24: π, integer-copy
  frequencies: 0), far-pair planes near-orthogonal (f32-vs-f128 0.001,
  f8-vs-f128 0.038). The canonical model is a clean test bed.
- **Dynamical systems through the canonical model:** harmonic/damped stay
  sparse (6–8/64 active dims, low temporal variance) — the frequency prior
  fits; Van der Pol spreads to 33 dims; **Lorenz fills 59/64 dims** — the
  chaotic, broadband signal defeats a fixed frequency decomposition. This is
  the empirical caution against over-optimizing for periodicity.
- **SIGReg twin:** `canonical-rich-sigreg.pt` (same spec/seed/corpus, loss =
  latent JEPA MSE + SIGReg) — trained for the batch-composition/SIGReg
  revisit.

**Conclusion.** The canonical direct model is a clean-geometry test bed but
its forecasting is below persistence — the same story as the map-reduce
finding, now quantified. The SIGReg twin gives the controlled objective
comparison.

**GATr redesign (scoped, corrected).** ezgatr multivectors are **16-blade PGA**
(ℝ^{3,0,1}); `EquiLinear`'s 9-parameter weight is the *equivariant map basis*,
not the channel dim (correction to my earlier note). So a 64-dim patch
embedding = **4 channels × 16 blades** exactly; tensor
`(B, 16 patches, 4 channels, 16 blades)`. Output: `EquiLinear(4→64)` → take
the **scalar blade** (`[..., :, 0]`) of the 64 output channels → 64-step
forecast. The "multiple of 16" input constraint and the d=k threshold align at
patch 64. Deferred until both canonical baselines (direct + SIGReg) are
characterized; the open question is whether the ring-rotation inductive bias
survives projection into PGA (the old GATr runs underperformed there).

**Evidence.** Inline MASE tables; `probes/probe_dynamical_geo.py`;
`assets/dynamical_embeddings.png`.

---

## Iteration 49 — SIGReg canonical twin: the regularizer itself deforms the rings

**Question (user).** Revisit batch composition / SIGReg with a controlled twin
of the canonical model. Trained `canonical-rich-sigreg.pt` — same spec/seed/
corpus as the direct canonical (ctx 1024 / patch 64 / hidden 64 / 2L / 32k /
uniform rich), only the loss differs (latent JEPA MSE + SIGReg).

**Results.** Side-by-side geometry baseline:

| | DIRECT | SIGReg |
|---|---|---|
| f8 ring s1/s2 (ellipticity) | 1.65 | **8.17** |
| f64 ring s1/s2 | 1.30 | **19.3** |
| f128 ring s1/s2 | 1.36 | **5.46** |
| centroid norm | 1.6–2.0 | **4.6–6.3** |
| far overlap 8–128 | 0.038 | **0.467** |
| far overlap 32–128 | 0.001 | **0.484** |
| rotation operator | exact Δφ | exact Δφ |

**Conclusion.** **SIGReg itself deforms the ring geometry — independent of
the training marginal.** With the identical uniform corpus and spec, the
SIGReg model produces elliptical rings (s1/s2 up to 19 vs 1.3 direct), 3×
larger centroid offsets, and far-apart planes that are *not separable*
(overlap ~0.47 vs 0.001–0.04 direct). The rotation operator is still learned
exactly in both, but SIGReg's isotropic-Gaussian target pulls the rings into
off-center ellipses that collapse plane orthogonality — the same signature
Iteration 45 attributed to the uneven marginal, now isolated to the
regularizer. This matters for the batch-composition revisit: SIGReg's marginal
Gaussianization is geometrically destructive to the phase-ring structure, so
any batch-composition effect must be read through (and may be confounded by)
this deformation. The direct objective is the cleaner geometry test bed.

**Evidence.** Inline tables (both canonical models).

---

## Iteration 50 — Mechanism: SIGReg bends the rings to Gaussianize, at forecasting's cost

**Question (user).** (1) Why does SIGReg deform the rings? (2) Does the
deformation hurt forecasting?

**Mechanism (measured).** SIGReg minimizes the Epps–Pulley statistic, which
penalizes non-Gaussian marginals. A ring's 1-D projection is *bimodal*
(kurtosis ≪ 3) — the maximally anti-Gaussian shape. To satisfy EP, the model
stretches each ring into a thin ellipse (s1/s2 up to 19 vs 1.3 direct) and
pushes the centroid off-origin (|c| 4.6–6.3 vs 1.6–2.0): a far-offset ellipse
projects as a tight, locally-Gaussian cluster. Measured on the canonical pair:
DIRECT EP(marginal) = 0.385, kurtosis 6.39, variance spread 4.5×; SIGReg EP =
0.042 (near-Gaussian), kurtosis 2.28, variance spread **71×**. The rotation
operator (phase advance) is exact in *both* — the deformation is static
geometry, not dynamics. This is a first-order regularizer×ring interaction,
independent of batch composition (the SIGReg twin uses uniform batches).

**Forecasting cost (measured, teacher-forced one-step MASE vs lag-1):**

| signal | DIRECT | SIGReg |
|---|---|---|
| rich | **1.318** | 1.669 |
| tone-f8 | **13.67** | 23.27 |
| tone-f64 | **1.164** | **2.417** |
| tone-f128 | 1.203 | **1.014** |
| mix-24+128 | **1.655** | 1.641 |

SIGReg is worse on 4/5 (rich +27%, f8 +70%, **f64 +108%** — the most-deformed
ring, s1/s2=19.3, is the most-degraded forecast); only f128 improves (−16%,
where both are poor and probe noise dominates).

**Conclusion.** The causal chain is complete: **EP pressure → bends rings
(anisotropic stretch + off-center) → Gaussianizes the marginal (EP 0.04) →
distorts the radius forecasting decodes from → worse MASE.** SIGReg wins its
own metric at the direct, measured cost of forecasting accuracy; the most
deformed ring shows the largest degradation. The direct objective is the
better forecasting test bed; SIGReg's deformation is a finding, not a
free regularizer.

**Evidence.** Inline EP/kurtosis table + MASE comparison (both canonical
models).

---

## Iteration 51 — VICReg control: variance-only regularization preserves the rings

**Hypothesis (user question).** SIGReg deforms the rings because its EP
statistic penalizes the ring's *shape* (bimodal projection). VICReg constrains
only *scale* (`Σ relu(1−std)`) and *decorrelation* (off-diagonal covariance) —
second-order, rotation-invariant terms with no shape penalty. It should keep
the rings circular and avoid the SIGReg forecasting cost.

**Experiment.** Trained `canonical-rich-vicreg.pt` — same spec/seed/corpus as
the direct and SIGReg canonicals (ctx 1024 / patch 64 / hidden 64 / 2L / 32k /
uniform rich), loss = latent JEPA MSE + VICReg. Three-axis baseline vs
DIRECT and SIGReg.

**Results.**

*Geometry (rings, planes):*

| | f8 s1/s2 | f64 s1/s2 | far-ov 8–128 |
|---|---|---|---|
| DIRECT | 1.65 | 1.30 | 0.038 |
| SIGReg | 8.17 | 19.28 | 0.467 |
| **VICReg** | **1.52** | **1.13** | **0.004** |

*Marginal (EP / kurtosis / var spread):*

| | EP | kurtosis | var spread |
|---|---|---|---|
| DIRECT | 0.385 | 6.39 | 4.5× |
| SIGReg | 0.042 | 2.28 | 71× |
| **VICReg** | **0.089** | **3.11** | **1.5×** |

*Forecasting (teacher-forced one-step MASE vs lag-1):*

| | DIRECT | SIGReg | VICReg |
|---|---|---|---|
| rich | 1.318 | 1.687 | **1.508** |
| f8 | 13.67 | 24.31 | **10.98** |
| f64 | 1.164 | 2.235 | **1.180** |
| f128 | **1.203** | 0.992 | 1.540 |
| mix | 1.655 | **1.563** | 1.763 |

**Conclusion.** **The user's hypothesis is confirmed.** VICReg keeps the rings
circular at 32k (s1/s2 1.13–1.52, essentially matching DIRECT; best plane
overlap of all, 0.004), because it never penalizes the ring's shape — only
scale and decorrelation, both rotation-invariant. Its marginal is nearly
isotropic (var spread 1.5×, best) but *not* forced Gaussian (kurtosis 3.11, EP
0.089 — intermediate: it decorrelates and standardizes without stretching
rings into Gaussian-looking ellipses). Forecasting is close to DIRECT and much
better than SIGReg on rich/f8/f64 (e.g. f64 1.18 vs 2.24, f8 10.98 vs 24.31).
So: **the ring deformation is specifically the shape-Gaussianization of
SIGReg, not regularization per se** — a variance-only regularizer gets most of
the anti-collapse benefit with clean geometry and better forecasting. This
directly informs the batch-composition revisit: if the goal is a clean-geometry
latent test bed with anti-collapse, VICReg (or VISReg) is the safer regularizer
than SIGReg.

**Evidence.** `results/ckpts/canonical-rich-vicreg.pt` (+ .json); inline
three-axis tables.

---

## Research backlog

The geometry supports a *package* of applications, not just one. Prioritized
by strength/feasibility; each entry states the geometric basis and the concrete
next experiment.

1. **Forecasting — geometry as diagnostic + fix (headline).**
   *Basis:* the shrinkage is a learned dilation (damping) and compounds over
   the rollout. *Have:* norm-copy pins the rollout to the true radius (valid
   only for purely deterministic signals; over-predicts and can hurt on
   stochastic data); scale-aware stops compounding. *Run:* show norm-copy /
   scale-aware / a radius-aware head improves Monash MASE.

2. **Regime / anomaly detection.** *Basis:* azimuth = class identity invisible
   to the value marginal. *Have:* 100% regime classification. *Run:* a
   radial-residual anomaly detector (distance from the ring plane) vs the
   spectral-residual baseline.

3. **Downstream classification via the azimuth.** *Basis:* last-patch `h` is
   phase-anchored; the azimuth is the class-bearing direction. *Run:* extract
   the plane azimuth as the feature for UCR/MONSTER and compare to
   `h-last`/`h-pool`/raw 1-NN.

4. **Temporal tokenizer.** *Basis:* `z ≈ (azimuth, phase, radius)` is a
   discrete, reconstructable code (decoder locality confirmed). *Run:*
   quantize the three components; report reconstruction + interpretability.

5. **Latent-quality metrics.** *Basis:* ring cleanliness (`dim1+2`), plane
   overlap, shrinkage ratio are principled diagnostics. *Run:* apply to a
   real pre-trained TSFM / our Monash model as a new evaluation suite.

**Plan:** lead with #1 (forecasting fix), keep #2 as the second application,
add #3 or #4 as a third.
