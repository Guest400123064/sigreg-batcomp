import math

import torch

from batcomp.cli.hmm import (
    BatchSampler,
    Windows,
    between_scatter,
    bin_labels,
    make_ar1,
    make_chain,
    make_hmm,
    probe_r2,
    regime_entropy,
    sample_regime_chain,
    within_scatter,
)

# --- data generation -------------------------------------------------------


def test_make_chain_rows_stochastic():
    T, pi = make_chain(3, dwell=100.0)
    assert T.shape == (3, 3)
    torch.testing.assert_close(T.sum(-1), torch.ones(3), atol=1e-6, rtol=0)
    torch.testing.assert_close(T @ pi, pi, atol=1e-6, rtol=0)


def test_make_chain_run_length():
    # A regime ends only when a jump redraws a *different* regime, so with
    # uniform pi the mean run length is dwell * R / (R - 1) = 2 * dwell here.
    rng = torch.Generator().manual_seed(0)
    T, _ = make_chain(2, dwell=50.0)
    s = sample_regime_chain(4, 40000, T, rng)
    runlen = s.numel() / (s[:, 1:] != s[:, :-1]).sum().item()
    assert abs(runlen - 100.0) < 5.0


def test_make_hmm_shapes_and_seed():
    a = make_hmm(num_regimes=3, dwell=50.0, sep=3.0, num_seqs=4, seq_len=256, seed=7)
    b = make_hmm(num_regimes=3, dwell=50.0, sep=3.0, num_seqs=4, seq_len=256, seed=7)
    x, s, means = a
    assert x.shape == (4, 256, 1) and s.shape == (4, 256)
    torch.testing.assert_close(means, torch.linspace(-3, 3, 3).unsqueeze(-1))
    torch.testing.assert_close(x, b[0])


def test_make_ar1_autocorrelation():
    # Single regime, no switching: lag-1 autocorr ~ phi and stationary
    # variance ~ ar_sigma^2 / (1 - phi^2).
    x, _, phi = make_ar1(num_regimes=1, dwell=1e6, phi=(0.7,), sep=0.0, ar_sigma=1.0,
                         num_seqs=2, seq_len=20000, seed=0)
    assert abs(phi.item() - 0.7) < 1e-6
    x = x.squeeze(-1)
    lag1 = (
        (x[:, 1:] - x[:, :-1].mean(-1, keepdim=True))
        * (x[:, :-1] - x[:, :-1].mean(-1, keepdim=True))
    ).mean()
    var = x.var(-1).mean()
    assert abs(lag1.item() / var.item() - 0.7) < 0.02
    assert abs(var.item() - 1.0 / (1 - 0.7**2)) < 0.1


def test_make_ar1_phi_scalar_broadcast():
    _, _, phi = make_ar1(
        num_regimes=3, dwell=50.0, phi=0.5, num_seqs=2, seq_len=64, seed=0
    )
    torch.testing.assert_close(phi, torch.full((3,), 0.5))


# --- batching --------------------------------------------------------------


def make_ds(num_seqs=2, seq_len=256, ctx=64):
    # Chain 0: windows 0,1 regime 0; windows 2,3 regime 1.
    # Chain 1: windows 0,1 regime 1; windows 2,3 regime 0.
    s0 = torch.tensor([0] * 128 + [1] * 128)
    s1 = torch.tensor([1] * 128 + [0] * 128)
    s = torch.stack([s0, s1])
    return Windows(torch.randn(num_seqs, seq_len, 1), s, ctx), s


def test_windows_and_contiguous_shapes():
    ds, _ = make_ds()
    assert len(ds) == 8
    assert ds[3][0].shape == (64, 1) and ds[3][1].shape == (64,)
    for batch in BatchSampler(ds, 2, 10, "contiguous", seed=0):
        assert len(set(i // 4 for i in batch)) == 1


def test_homogeneous_pure_and_roundrobin_cycle():
    ds, s = make_ds()
    for batch in BatchSampler(ds, 2, 40, "homogeneous", seed=0):
        labels = s.view(-1, ds.ctx)[batch, 0]
        assert labels.min() == labels.max()
    sampler = BatchSampler(ds, 1, 4, "roundrobin", seed=0)
    seen = [s.view(-1, ds.ctx)[batch, 0].item() for batch in sampler]
    assert seen == [0, 1, 0, 1]


# --- metrics ---------------------------------------------------------------


def test_scatter_decomposition_and_probe():
    feats = torch.tensor([[-1.0], [0.0], [1.0], [2.0]])
    labels = torch.tensor([0, 0, 1, 1])
    total = feats.square().mean() - feats.mean().square()
    sb = between_scatter(feats, labels, 2) + within_scatter(feats, labels, 2)
    assert abs(total.item() - sb) < 1e-5
    sep = torch.tensor([[-3.0], [3.0], [-3.0], [3.0]])
    assert abs(probe_r2(sep, torch.tensor([0, 1, 0, 1]), 2) - 1.0) < 1e-4


def test_probe_r2_nan_features_returns_nan():
    feats = torch.tensor([[1.0], [float("nan")], [1.0], [1.0]])
    r2 = probe_r2(feats, torch.tensor([0, 1, 0, 1]), 2)
    assert r2 != r2  # is NaN


def test_regime_entropy_and_bin_labels():
    assert abs(regime_entropy(torch.tensor([0, 1, 0, 1]), 2) - math.log(2)) < 1e-6
    labels = bin_labels(torch.linspace(-5, 5, 1000).unsqueeze(-1), 8)
    assert labels.min().item() >= 0 and labels.max().item() <= 7
