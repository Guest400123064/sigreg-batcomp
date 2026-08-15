"""HMM / AR(1) synthetic batch-composition experiments (SIGReg).

Usage: ``python -m batcomp.cli hmm [options]``

Runs are tracked with trackio (project ``--project``); ``--out`` optionally
also dumps a legacy {args, runs, summary} JSON.
"""

from __future__ import annotations

import copy
import json
import os

import torch
import torch.nn.functional as F
import trackio
from torch.utils.data import DataLoader, Dataset, Sampler

from batcomp.criteria import sigreg, var_diff
from batcomp.logging import get_logger
from batcomp.models.simple import SimpleModel

logger = get_logger(__name__)

STRATEGIES = ["representative", "contiguous", "homogeneous", "roundrobin"]

SUMMARY_KEYS = ["probe", "pred_mse", "sb_z", "sb_h", "sw_z", "sw_h", "r2_z", "r2_h"]


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def make_chain(num_regimes, dwell, pi_alpha=None, rng=None):
    r"""Sticky transition matrix with exact stationary distribution pi.

    ``T = p_self * I + (1 - p_self) * ones @ pi``: stay with probability
    ``p_self = 1 - 1 / dwell``, else jump to a fresh draw from ``pi``. The
    regime autocorrelation decays geometrically at rate ``p_self``, so
    ``dwell`` directly sets the autocorrelation time.
    """
    p_self = 1.0 - 1.0 / dwell
    if pi_alpha is None:
        pi = torch.full((num_regimes,), 1.0 / num_regimes)
    else:
        pi = torch.distributions.Dirichlet(torch.full((num_regimes,), pi_alpha)).sample(
            generator=rng
        )
    return p_self * torch.eye(num_regimes) + (1 - p_self) * pi.expand(num_regimes, -1), pi


def sample_regime_chain(num_seqs, seq_len, trans, rng):
    r"""Sample ``num_seqs`` parallel chains from transition matrix ``trans``."""
    num_regimes = trans.size(0)
    s = torch.empty(num_seqs, seq_len, dtype=torch.long)
    s[:, 0] = torch.multinomial(
        torch.full((num_regimes,), 1.0 / num_regimes).expand(num_seqs, -1), 1, generator=rng
    ).squeeze(1)
    for t in range(1, seq_len):
        cum = trans[s[:, t - 1]].cumsum(-1)
        s[:, t] = (cum > torch.rand(num_seqs, 1, generator=rng)).float().argmax(-1)
    return s


def regime_means(num_regimes, sep, d_obs, rng=None):
    r"""Regime means at distance ``sep`` from the origin.

    On a 1-D "sphere" random directions collapse to +-sep with random signs,
    so regimes can coincide; use equally spaced means instead.
    """
    if d_obs == 1:
        return torch.linspace(-sep, sep, num_regimes).unsqueeze(-1)
    m = torch.randn(num_regimes, d_obs, generator=rng)
    return m / m.norm(dim=-1, keepdim=True) * sep


def make_hmm(
    *,
    num_regimes=2,
    dwell=100.0,
    sep=3.0,
    sigma=1.0,
    pi_alpha=None,
    num_seqs=64,
    seq_len=8192,
    d_obs=1,
    seed=0,
):
    r"""Sticky Gaussian HMM: latent regime chain -> Gaussian emission -> labels.

    Regime means are equally spaced at distance ``sep`` from the origin (in 1-D)
    or random directions at distance ``sep`` (in >1-D), so the grand mean is ~0
    and ``tr(Sigma_B) ~ sep^2``; emissions are isotropic ``N(m_r, sigma^2 I)``,
    so separability is governed by ``sep / sigma``.

    Returns
    -------
    x : torch.Tensor, shape (num_seqs, seq_len, d_obs)
    s : torch.Tensor, shape (num_seqs, seq_len), ground-truth regime labels
    means : torch.Tensor, shape (num_regimes, d_obs)
    """
    rng = torch.Generator().manual_seed(seed)
    trans, _ = make_chain(num_regimes, dwell, pi_alpha, rng)
    means = regime_means(num_regimes, sep, d_obs, rng)
    s = sample_regime_chain(num_seqs, seq_len, trans, rng)
    x = means[s] + sigma * torch.randn(num_seqs, seq_len, d_obs, generator=rng)
    return x, s, means


def make_ar1(
    *,
    num_regimes=2,
    dwell=100.0,
    phi=(0.9, -0.9),
    sep=3.0,
    ar_sigma=1.0,
    obs_noise=0.0,
    num_seqs=64,
    seq_len=8192,
    seed=0,
):
    r"""Regime-switching AR(1): a smooth, autocorrelated latent process.

    Within regime ``r`` the deviation ``e_t`` follows an AR(1) around the regime
    level ``mu_r``: ``x_t = mu_r + e_t``, ``e_t = phi_r e_{t-1} + sigma_a eps``.
    With ``sep > 0`` regimes differ in level; with ``sep = 0`` only in
    persistence ``phi`` (per-regime z-marginals then differ in variance,
    ``sigma_a^2 / (1 - phi_r^2)``).

    Returns
    -------
    x : torch.Tensor, shape (num_seqs, seq_len, 1), observations
    s : torch.Tensor, shape (num_seqs, seq_len), ground-truth regime labels
    phi : torch.Tensor, shape (num_regimes,), persistence per regime
    """
    if isinstance(phi, (int, float)):
        phi = (float(phi),) * num_regimes
    assert len(phi) == num_regimes, f"phi length {len(phi)} != num_regimes {num_regimes}"
    phi = torch.as_tensor(phi, dtype=torch.float32)
    assert bool((phi.abs() < 1).all()), "|phi| < 1 required for stationarity"

    rng = torch.Generator().manual_seed(seed)
    trans, _ = make_chain(num_regimes, dwell, None, rng)
    if sep != 0:
        mu = torch.linspace(-sep, sep, num_regimes).unsqueeze(-1)
    else:
        mu = torch.zeros(num_regimes, 1)
    s = sample_regime_chain(num_seqs, seq_len, trans, rng)

    e = torch.empty(num_seqs, seq_len, 1)
    # Warm start each chain from the stationary marginal of its first regime.
    var0 = ar_sigma**2 / (1 - phi[s[:, 0]] ** 2)
    e[:, 0, 0] = var0.sqrt() * torch.randn(num_seqs, generator=rng)
    for t in range(1, seq_len):
        e[:, t] = phi[s[:, t]].unsqueeze(-1) * e[:, t - 1] + ar_sigma * torch.randn(
            num_seqs, 1, generator=rng
        )

    x = mu[s] + e
    if obs_noise > 0:
        x = x + obs_noise * torch.randn(num_seqs, seq_len, 1, generator=rng)
    return x, s, phi


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


class Windows(Dataset):
    r"""Non-overlapping length-``ctx`` windows over parallel chains.

    Each item is ``(x, s)`` with shapes ``(ctx, d_obs)`` and ``(ctx,)``.
    """

    def __init__(self, x, s, ctx):
        self.x, self.s, self.ctx = x, s, ctx
        self.nw = x.size(1) // ctx

    def __len__(self):
        return self.x.size(0) * self.nw

    def __getitem__(self, i):
        n, w = divmod(i, self.nw)
        sl = slice(w * self.ctx, (w + 1) * self.ctx)
        return self.x[n, sl], self.s[n, sl]


class BatchSampler(Sampler):
    r"""Yield ``steps`` batches of window indices under a composition strategy.

    - ``representative``: uniform random windows; the batch marginal matches
      the corpus mixture in expectation.
    - ``contiguous``: each batch is a contiguous run of windows within one
      chain; realistic temporal batching, homogeneous only when
      ``dwell >> batch_size * ctx``.
    - ``homogeneous``: oracle single-regime batches (pure windows of one
      regime, using labels); the theoretical object of the bias analysis.
    - ``roundrobin``: oracle single-regime batches cycling through regimes;
      each batch is still homogeneous (tests bias persistence).
    """

    def __init__(self, ds, batch_size, steps, strategy, seed=0):
        if strategy == "contiguous" and ds.nw < batch_size:
            raise ValueError(f"Need windows-per-chain >= batch_size, got {ds.nw} < {batch_size}")
        self.ds, self.bs, self.steps, self.strategy = ds, batch_size, steps, strategy
        self.rng = torch.Generator().manual_seed(seed)
        # Pure-window indices grouped by regime label.
        sw = ds.s.view(-1, ds.ctx)
        pure = (sw == sw[:, :1]).all(-1)
        idx = torch.arange(len(ds))[pure]
        self.pure = [idx[sw[pure, 0] == r] for r in range(ds.s.max().item() + 1)]

    def __len__(self):
        return self.steps

    def __iter__(self):
        ds, bs, rng = self.ds, self.bs, self.rng
        for step in range(self.steps):
            match self.strategy:
                case "representative":
                    yield torch.randint(len(ds), (bs,), generator=rng).tolist()
                case "contiguous":
                    n = torch.randint(ds.x.size(0), (1,), generator=rng).item()
                    w = torch.randint(ds.nw - bs + 1, (1,), generator=rng).item()
                    yield [n * ds.nw + w + i for i in range(bs)]
                case "homogeneous" | "roundrobin":
                    r = torch.randint(len(self.pure), (1,), generator=rng).item()
                    if self.strategy == "roundrobin":
                        r = step % len(self.pure)
                    pool = self.pure[r]
                    yield pool[torch.randint(len(pool), (bs,), generator=rng)].tolist()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def between_scatter(feats, labels, num_groups):
    r"""Between-group scatter ``tr(Sigma_B) = sum_r pi_r ||mu_r - bar_mu||^2``.

    The continuous-latent analog is ``tr(Cov(E[z | latent]))``: for groups that
    are quantile bins of the latent, this measures how much of the latent
    structure ``z`` retains.
    """
    cnt = torch.bincount(labels, minlength=num_groups)
    bar = feats.mean(0)
    mus = torch.stack([feats[labels == r].mean(0) for r in range(num_groups)])
    return (cnt / cnt.sum() * ((mus - bar) ** 2).sum(-1)).sum().item()


def within_scatter(feats, labels, num_groups):
    r"""Within-group scatter ``tr(E[Cov(z | group)]) = sum_r pi_r tr(C_r)``.

    The residual embedding variance after conditioning on the group; together
    with between_scatter it sums to ``tr(Cov(z))``.
    """
    mus = torch.stack([feats[labels == r].mean(0) for r in range(num_groups)])
    dev = feats - mus[labels]
    return (dev.square().mean(0).sum()).item()


def probe_r2(feats, labels, num_groups):
    r"""R^2 of a closed-form linear probe from ``feats`` to one-hot group labels.

    Uses the SVD-based ``gelsd`` driver (robust to rank-deficient design
    matrices) and returns NaN for non-finite features: at extreme regularization
    a run can diverge to NaN embeddings, and ``lstsq`` raises on NaN inputs
    instead of returning NaN.
    """
    if not torch.isfinite(feats).all():
        return float("nan")
    y = F.one_hot(labels, num_groups).float()
    A = torch.cat([feats, torch.ones(len(feats), 1)], 1)
    sol = torch.linalg.lstsq(A, y, driver="gelsd")
    resid = y - A @ sol.solution
    return (1 - resid.square().sum() / (y - y.mean(0)).square().sum()).item()


def regime_entropy(s, num_groups):
    r"""Shannon entropy (nats) of the batch-level group composition."""
    p = torch.bincount(s.flatten(), minlength=num_groups).float()
    p = p / p.sum()
    return -(p * p.clamp_min(1e-12).log()).sum().item()


def bin_labels(x, num_bins):
    r"""Quantile-bin the pooled values of ``x`` into ``num_bins`` integer labels.

    Binning turns the continuous latent into a finite set of analysis groups,
    so the discrete-regime scatter/probe machinery applies unchanged; the
    binned ``tr(Sigma_B)`` estimates ``tr(Cov(E[z | latent]))``.
    """
    xf = x.reshape(-1)
    qs = torch.quantile(xf, torch.linspace(0, 1, num_bins + 1, device=x.device))
    return torch.bucketize(xf, qs[1:-1]).reshape(x.shape)


@torch.no_grad()
def evaluate(model, x, labels, ctx, num_groups, device):
    r"""Teacher-forced forecasting MSE, scatter, and probe R^2 on held-out windows.

    ``labels`` carries the analysis grouping (regime ids for the HMM, quantile
    bins for the AR(1) DGP) with shape matching ``x``.
    """
    M, T, d = x.shape
    x = x.view(M * (T // ctx), ctx, d)
    labels = labels.view(M * (T // ctx), ctx)
    pred, z, h = model(x.to(device))
    mse = F.mse_loss(pred[:, :-1], z[:, 1:]).item()
    zf, hf, sf = z.flatten(0, 1).cpu(), h.flatten(0, 1).cpu(), labels.flatten()
    return {
        "pred_mse": mse,
        "sb_z": between_scatter(zf, sf, num_groups),
        "sb_h": between_scatter(hf, sf, num_groups),
        "sw_z": within_scatter(zf, sf, num_groups),
        "sw_h": within_scatter(hf, sf, num_groups),
        "r2_z": probe_r2(zf, sf, num_groups),
        "r2_h": probe_r2(hf, sf, num_groups),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_model(args, device):
    return SimpleModel(
        hidden_size=args.hidden, num_hidden_layers=args.layers,
        num_attention_heads=args.heads, num_input_channels=args.d_obs,
        context_size=args.ctx,
    ).to(device)


@torch.no_grad()
def init_probe(model, ds, sampler, ref_size, device):
    r"""Training-free variance-discrepancy probe at initialization.

    Averages per-batch variance over the sampler's batches and compares
    against the variance of a large shuffled reference set of latents.
    """
    ref_idx = torch.randint(len(ds), (ref_size,), generator=sampler.rng)
    ref = torch.stack([ds[i][0] for i in ref_idx.tolist()]).to(device)
    ref = model.get_state_latents(ref).flatten(0, 1)
    vals = []
    for idx in sampler:
        xb = torch.stack([ds[i][0] for i in idx]).to(device)
        vals.append(var_diff(model.get_state_latents(xb), ref).item())
    return sum(vals) / len(vals)


def train_strategy(args, strategy, data, init_state, device):
    r"""Train one copy of the shared init under one batch strategy.

    The whole run (config, per-eval-step metrics) is logged to a trackio run
    named ``<strategy>-seed<seed>`` grouped by strategy; the in-memory ``log``
    is returned for the cross-seed summary printed by the CLI.
    """
    (xtr, str_), (xev, sev) = data
    ds = Windows(xtr, str_, args.ctx)
    # Sampler RNG differs per strategy (reproducibly) so batch streams are
    # independent of each other but fixed across seeds.
    sseed = args.seed * 1000 + STRATEGIES.index(strategy)
    loader = DataLoader(
        ds, batch_sampler=BatchSampler(ds, args.batch_size, args.steps, strategy, seed=sseed)
    )

    model = build_model(args, device)
    model.load_state_dict(copy.deepcopy(init_state))
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    probe = init_probe(
        model, ds,
        BatchSampler(ds, args.batch_size, args.probe_batches, strategy, seed=sseed + 1),
        4096, device,
    )

    def ev():
        model.eval()
        m = evaluate(model, xev, sev, args.ctx, args.num_groups, device)
        model.train()
        return m

    run = trackio.init(
        project=args.project,
        name=f"{strategy}-seed{args.seed}",
        group=strategy,
        config={k: v for k, v in vars(args).items() if k not in ("func", "cmd")}
        | {"strategy": strategy, "seed": args.seed},
    )
    log = []
    try:
        log.append({"step": 0, "probe": probe, **ev()})
        run.log({k: v for k, v in log[-1].items() if k != "step"}, step=0)
        for step, (xb, sb) in enumerate(loader, 1):
            xb = xb.to(device)
            pred, z, h = model(xb)
            loss_pred = F.mse_loss(pred[:, :-1], z[:, 1:].detach())
            loss_reg = sigreg(z.flatten(0, 1))
            (loss_pred + args.lam * loss_reg).backward()
            opt.step()
            opt.zero_grad()

            if step % args.eval_every == 0 or step == args.steps:
                entry = {
                    "step": step, "batch_entropy": regime_entropy(sb, args.num_regimes),
                    "loss_pred": loss_pred.item(), "loss_reg": loss_reg.item(), **ev(),
                }
                log.append(entry)
                run.log({k: v for k, v in entry.items() if k != "step"}, step=step)
                logger.info(f"    step={step} {entry}")
        return log
    finally:
        run.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_cmd(subparsers):
    p = subparsers.add_parser(
        "hmm",
        help="synthetic HMM / AR(1) batch-composition experiments",
        description="SIGReg batch-composition experiments on synthetic DGPs.",
    )
    p.add_argument("--project", type=str, default="batcomp-stage1", help="trackio project name")
    p.add_argument("--dgp", choices=["hmm", "ar1"], default="hmm")
    p.add_argument("--strategies", nargs="*", choices=STRATEGIES, default=STRATEGIES)
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--num-regimes", type=int, default=2)
    p.add_argument("--dwell", type=float, default=100.0)
    p.add_argument("--sep", type=float, default=3.0)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--pi-alpha", type=float, default=None)
    p.add_argument("--d-obs", type=int, default=1)
    p.add_argument("--num-seqs", type=int, default=64)
    p.add_argument("--eval-seqs", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--probe-batches", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--out", type=str, default=None,
                   help="optional: also dump {args, runs, summary} JSON (legacy; "
                        "trackio remains the primary log)")

    # AR(1) DGP options.
    p.add_argument("--phi", type=str, default="0.9,-0.9",
                   help="comma-separated persistence per regime (|phi| < 1); "
                        "a single value is broadcast to all regimes")
    p.add_argument("--ar-sigma", type=float, default=1.0, help="AR(1) innovation scale")
    p.add_argument("--obs-noise", type=float, default=0.0,
                   help="i.i.d. observation noise added on top of the latent")
    p.add_argument("--bins", type=int, default=8,
                   help="quantile bins of the latent used as analysis groups (ar1 only)")
    p.set_defaults(func=run)
    return p


def run(args) -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    runs = []
    for seed in args.seeds:
        args.seed = seed
        data = build_data(args)

        x, s = data[0]
        cnt = torch.bincount(s.flatten(), minlength=args.num_regimes)
        runlen = s.numel() / (s[:, 1:] != s[:, :-1]).sum().item()
        logger.info(f"[seed={seed}] realized pi={[round(v, 3) for v in (cnt / cnt.sum()).tolist()]} mean_runlen={runlen:.1f}")

        # One random init per seed; every strategy trains an identical copy,
        # so batch composition is the only thing that changes.
        torch.manual_seed(seed)
        init_state = build_model(args, device).state_dict()

        for strategy in args.strategies:
            logger.info(f"  [{strategy}]")
            runs.append({"seed": seed, "strategy": strategy,
                         "log": train_strategy(args, strategy, data, init_state, device)})

    summary = {}
    for st in args.strategies:
        finals = [r["log"][-1] for r in runs if r["strategy"] == st]
        probes = [r["log"][0]["probe"] for r in runs if r["strategy"] == st]
        summary[st] = {
            k: {"mean": sum(f[k] for f in finals) / len(finals),
                "std": torch.tensor([f[k] for f in finals]).std().item()}
            for k in SUMMARY_KEYS if k != "probe"
        }
        summary[st]["probe"] = {"mean": sum(probes) / len(probes),
                                "std": torch.tensor(probes).std().item()}

    logger.info("\n[summary] final-step metrics across seeds (mean +/- std)")
    for st, row in summary.items():
        logger.info(f"  {st:>15}: " + "  ".join(f"{k}={v['mean']:.4f}+-{v['std']:.4f}" for k, v in row.items()))

    if args.out:
        with open(args.out, "w") as fs:
            json.dump({"args": vars(args), "runs": runs, "summary": summary}, fs, indent=2)
        logger.info(f"[saved] {args.out}")

    # Point the user at where trackio stored the runs.
    trackio_dir = os.environ.get("TRACKIO_DIR") or os.path.expanduser("~/.cache/huggingface/trackio")
    logger.info(f"[trackio] project '{args.project}' -> {trackio_dir} "
                f"(dashboard: `trackio show --project {args.project}`)")
    return 0


def build_data(args):
    if args.dgp == "hmm":
        x, s, _ = make_hmm(
            num_regimes=args.num_regimes, dwell=args.dwell, sep=args.sep, sigma=args.sigma,
            pi_alpha=args.pi_alpha, num_seqs=args.num_seqs, seq_len=args.seq_len,
            d_obs=args.d_obs, seed=args.seed,
        )
        args.num_groups = args.num_regimes
        s_analysis = s
    else:
        phi = tuple(float(v) for v in args.phi.split(","))
        if len(phi) == 1:
            phi = phi * args.num_regimes
        if len(phi) != args.num_regimes:
            raise ValueError(f"--phi has {len(phi)} values but --num-regimes={args.num_regimes}")
        x, s, _ = make_ar1(
            num_regimes=args.num_regimes, dwell=args.dwell, phi=phi, sep=args.sep,
            ar_sigma=args.ar_sigma, obs_noise=args.obs_noise,
            num_seqs=args.num_seqs, seq_len=args.seq_len, seed=args.seed,
        )
        args.num_groups = args.bins
        # The analysis grouping is the continuous latent binned into quantiles;
        # regime ids remain available for the sampler (pure-window batching).
        s_analysis = bin_labels(x.squeeze(-1), args.bins)
    return (x[: -args.eval_seqs], s[: -args.eval_seqs]), (x[-args.eval_seqs :], s_analysis[-args.eval_seqs :])
