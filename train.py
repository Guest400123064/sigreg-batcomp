from __future__ import annotations

import argparse
import copy
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from batcomp.criteria import sigreg, var_diff
from batcomp.models.simple import SimpleModel

STRATEGIES = ["representative", "contiguous", "homogeneous", "roundrobin"]


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

    Regime means are random directions at distance ``sep`` from the origin, so
    the grand mean is ~0 and ``tr(Sigma_B) ~ sep^2``; emissions are isotropic
    ``N(m_r, sigma^2 I)``, so separability is governed by ``sep / sigma``.

    Returns
    -------
    x : torch.Tensor, shape (num_seqs, seq_len, d_obs)
    s : torch.Tensor, shape (num_seqs, seq_len), ground-truth regime labels
    means : torch.Tensor, shape (num_regimes, d_obs)
    """
    rng = torch.Generator().manual_seed(seed)
    trans, _ = make_chain(num_regimes, dwell, pi_alpha, rng)
    if d_obs == 1:
        # On a 1-D "sphere" random directions collapse to +-sep with random
        # signs, so regimes can coincide; use equally spaced means instead.
        means = torch.linspace(-sep, sep, num_regimes).unsqueeze(-1)
    else:
        means = torch.randn(num_regimes, d_obs, generator=rng)
        means = means / means.norm(dim=-1, keepdim=True) * sep

    s = torch.empty(num_seqs, seq_len, dtype=torch.long)
    s[:, 0] = torch.multinomial(
        torch.full((num_regimes,), 1.0 / num_regimes).expand(num_seqs, -1), 1, generator=rng
    ).squeeze(1)
    for t in range(1, seq_len):
        cum = trans[s[:, t - 1]].cumsum(-1)
        s[:, t] = (cum > torch.rand(num_seqs, 1, generator=rng)).float().argmax(-1)

    x = means[s] + sigma * torch.randn(num_seqs, seq_len, d_obs, generator=rng)
    return x, s, means


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


def between_scatter(feats, labels, num_regimes):
    r"""Between-regime scatter ``tr(Sigma_B) = sum_r pi_r ||mu_r - bar_mu||^2``."""
    cnt = torch.bincount(labels, minlength=num_regimes)
    bar = feats.mean(0)
    mus = torch.stack([feats[labels == r].mean(0) for r in range(num_regimes)])
    return (cnt / cnt.sum() * ((mus - bar) ** 2).sum(-1)).sum().item()


def probe_r2(feats, labels, num_regimes):
    r"""R^2 of a closed-form linear probe from ``feats`` to one-hot regime labels."""
    y = F.one_hot(labels, num_regimes).float()
    A = torch.cat([feats, torch.ones(len(feats), 1)], 1)
    sol = torch.linalg.lstsq(A, y)
    resid = y - A @ sol.solution
    return (1 - resid.square().sum() / (y - y.mean(0)).square().sum()).item()


def regime_entropy(s, num_regimes):
    r"""Shannon entropy (nats) of the batch-level regime composition."""
    p = torch.bincount(s.flatten(), minlength=num_regimes).float()
    p = p / p.sum()
    return -(p * p.clamp_min(1e-12).log()).sum().item()


@torch.no_grad()
def evaluate(model, x, s, ctx, num_regimes, device):
    r"""Teacher-forced forecasting MSE, scatter, and probe R^2 on held-out windows."""
    M, T, d = x.shape
    x = x.view(M * (T // ctx), ctx, d)
    s = s.view(M * (T // ctx), ctx)
    pred, z, h = model(x.to(device))
    mse = F.mse_loss(pred[:, :-1], z[:, 1:]).item()
    zf, hf, sf = z.flatten(0, 1).cpu(), h.flatten(0, 1).cpu(), s.flatten()
    return {
        "pred_mse": mse,
        "sb_z": between_scatter(zf, sf, num_regimes),
        "sb_h": between_scatter(hf, sf, num_regimes),
        "r2_z": probe_r2(zf, sf, num_regimes),
        "r2_h": probe_r2(hf, sf, num_regimes),
    }


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


def run(args, strategy, data, init_state, device):
    r"""Train one copy of the shared init under one batch strategy."""
    (xtr, str_), (xev, sev) = data
    ds = Windows(xtr, str_, args.ctx)
    # Sampler RNG differs per strategy (reproducibly) so batch streams are
    # independent of each other but fixed across seeds.
    sseed = args.seed * 1000 + STRATEGIES.index(strategy)
    loader = DataLoader(
        ds, batch_sampler=BatchSampler(ds, args.batch_size, args.steps, strategy, seed=sseed)
    )

    model = SimpleModel(
        hidden_size=args.hidden, num_hidden_layers=args.layers,
        num_attention_heads=args.heads, num_input_channels=args.d_obs,
        context_size=args.ctx,
    ).to(device)
    model.load_state_dict(copy.deepcopy(init_state))
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    probe = init_probe(
        model, ds,
        BatchSampler(ds, args.batch_size, args.probe_batches, strategy, seed=sseed + 1),
        4096, device,
    )

    def ev():
        model.eval()
        m = evaluate(model, xev, sev, args.ctx, args.num_regimes, device)
        model.train()
        return m

    log = [{"step": 0, "probe": probe, **ev()}]
    for step, (xb, sb) in enumerate(loader, 1):
        xb = xb.to(device)
        pred, z, h = model(xb)
        loss_pred = F.mse_loss(pred[:, :-1], z[:, 1:].detach())
        loss_reg = sigreg(z.flatten(0, 1))
        (loss_pred + args.lam * loss_reg).backward()
        opt.step()
        opt.zero_grad()

        if step % args.eval_every == 0 or step == args.steps:
            log.append({
                "step": step, "batch_entropy": regime_entropy(sb, args.num_regimes),
                "loss_pred": loss_pred.item(), "loss_reg": loss_reg.item(), **ev(),
            })
            print(f"    step={step} {log[-1]}")
    return log


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
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
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    runs = []
    for seed in args.seeds:
        args.seed = seed
        x, s, _ = make_hmm(
            num_regimes=args.num_regimes, dwell=args.dwell, sep=args.sep, sigma=args.sigma,
            pi_alpha=args.pi_alpha, num_seqs=args.num_seqs, seq_len=args.seq_len,
            d_obs=args.d_obs, seed=seed,
        )
        data = (x[: -args.eval_seqs], s[: -args.eval_seqs]), (x[-args.eval_seqs :], s[-args.eval_seqs :])

        cnt = torch.bincount(s.flatten(), minlength=args.num_regimes)
        runlen = s.numel() / (s[:, 1:] != s[:, :-1]).sum().item()
        print(f"[seed={seed}] realized pi={[round(v, 3) for v in (cnt / cnt.sum()).tolist()]} mean_runlen={runlen:.1f}")

        # One random init per seed; every strategy trains an identical copy,
        # so batch composition is the only thing that changes.
        torch.manual_seed(seed)
        init_state = SimpleModel(
            hidden_size=args.hidden, num_hidden_layers=args.layers,
            num_attention_heads=args.heads, num_input_channels=args.d_obs,
            context_size=args.ctx,
        ).state_dict()

        for strategy in args.strategies:
            print(f"  [{strategy}]")
            runs.append({"seed": seed, "strategy": strategy,
                         "log": run(args, strategy, data, init_state, device)})

    keys = ["probe", "pred_mse", "sb_z", "sb_h", "r2_z", "r2_h"]
    summary = {}
    for st in args.strategies:
        finals = [r["log"][-1] for r in runs if r["strategy"] == st]
        probes = [r["log"][0]["probe"] for r in runs if r["strategy"] == st]
        summary[st] = {
            k: {"mean": sum(f[k] for f in finals) / len(finals),
                "std": torch.tensor([f[k] for f in finals]).std().item()}
            for k in keys if k != "probe"
        }
        summary[st]["probe"] = {"mean": sum(probes) / len(probes),
                                "std": torch.tensor(probes).std().item()}

    print("\n[summary] final-step metrics across seeds (mean +/- std)")
    for st, row in summary.items():
        print(f"  {st:>15}: " + "  ".join(f"{k}={v['mean']:.4f}+-{v['std']:.4f}" for k, v in row.items()))

    out = args.out or f"results_dwell{args.dwell:g}_lam{args.lam:g}.json"
    with open(out, "w") as fs:
        json.dump({"args": vars(args), "runs": runs, "summary": summary}, fs, indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
