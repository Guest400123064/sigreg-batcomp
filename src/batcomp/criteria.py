from __future__ import annotations

import torch


def var_diff(bs, ref):
    r"""Variance discrepancy probe.

    Compare the per-batch variance against larger IID sample variance,
    and the take the average over multiple batches.
    """
    return torch.abs(bs.var(-2).mean(-1) - ref.var(0).mean()).mean()


def sigreg(z, num_slices=64, num_knots=17, t_max=5):
    t = torch.linspace(-t_max, t_max, num_knots, device=z.device)
    z = torch.atleast_2d(z)

    _, C = torch.atleast_2d(z).shape
    A = torch.randn(C, num_slices, device=z.device)
    A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)

    ref = torch.exp(-0.5 * t**2)
    ecf = torch.exp(1j * (z @ A).unsqueeze(2) * t.view(1, 1, -1)).mean(dim=0)

    return (ecf - ref.unsqueeze(0)).abs().pow(2).mean()
