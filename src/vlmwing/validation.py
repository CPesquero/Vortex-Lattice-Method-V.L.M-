"""Reference data and mesh-convergence utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

from .geometry import Wing
from .solver import solve

__all__ = ["XFLR5_ALPHAS", "load_xflr5", "ConvergencePoint", "mesh_convergence"]

#: Angles of attack of the columns in ``xflr5_cl.txt``.
XFLR5_ALPHAS = (-12.0, -6.0, 0.0, 6.0, 12.0)


def load_xflr5(path: str | Path | None = None) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Read the XFLR5 spanwise lift export.

    The file holds five ``(y, Cl)`` column pairs, one per angle in
    :data:`XFLR5_ALPHAS`, with no header row. Returns ``(y, {alpha: cl})``.
    """
    if path is None:
        text = resources.files("vlmwing.data").joinpath("xflr5_cl.txt").read_text()
        data = np.loadtxt(text.splitlines())
    else:
        data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != 2 * len(XFLR5_ALPHAS):
        raise ValueError(f"unexpected XFLR5 table shape {data.shape}")
    y = data[:, 0]
    for k in range(1, len(XFLR5_ALPHAS)):
        if not np.allclose(data[:, 2 * k], y):
            raise ValueError("spanwise stations differ between columns")
    return y, {a: data[:, 2 * k + 1] for k, a in enumerate(XFLR5_ALPHAS)}


@dataclass(frozen=True)
class ConvergencePoint:
    n_panels: int
    CL: float
    seconds: float


def mesh_convergence(
    wing: Wing, alpha_deg: float = 6.0, panels=(4, 10, 20, 40, 80, 120, 160, 320), repeats: int = 3
) -> list[ConvergencePoint]:
    """CL and wall time (best of ``repeats``) versus number of spanwise panels."""
    out = []
    for n in panels:
        best = np.inf
        for _ in range(repeats):
            t0 = time.perf_counter()
            CL = float(solve(wing, alpha_deg, n).CL[0])
            best = min(best, time.perf_counter() - t0)
        out.append(ConvergencePoint(int(n), CL, best))
    return out
