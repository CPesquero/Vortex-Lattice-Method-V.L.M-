"""Planform search for root-first stall.

Stall onset is taken as the semispan station where the sectional lift
coefficient peaks at a high angle of attack. With a spanwise-uniform
``cl_max`` this is where separation would start. It is a linear,
inviscid proxy: the VLM itself does not model separation.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .geometry import BASELINE, Wing, naca4_camber_slope
from .solver import influence_matrix

__all__ = ["StallCriteria", "Candidate", "stall_onset", "search", "DEFAULT_GRID"]


@dataclass(frozen=True)
class StallCriteria:
    alpha_deg: float = 15.0
    max_onset: float = 0.40  # fraction of semispan
    max_gradient: float = -0.20  # dCl/dy just outboard of the peak [1/m]


@dataclass(frozen=True)
class Candidate:
    wing: Wing
    onset: float  # y/(b/2) of the cl peak
    gradient: float  # dCl/dy outboard of the peak [1/m]
    cost: float
    y_right: np.ndarray = field(repr=False)
    cl_right: np.ndarray = field(repr=False)


def stall_onset(y_mid: np.ndarray, cl: np.ndarray, semispan: float) -> tuple[float, float, int]:
    """Return ``(onset_fraction, gradient, index)`` on the right half-wing."""
    right = y_mid >= 0
    y_r, cl_r = y_mid[right], cl[right]
    k = int(np.argmax(cl_r))
    grad = (cl_r[k + 1] - cl_r[k]) / (y_r[k + 1] - y_r[k]) if k < cl_r.size - 1 else 0.0
    return float(y_r[k] / semispan), float(grad), k


def _cost(wing: Wing, ref: Wing) -> float:
    """Normalized L2 distance from the reference planform."""

    def rel(v: float, v0: float, scale: float) -> float:
        return (v - v0) / (v0 if v0 else scale)

    terms = (
        rel(wing.aspect_ratio, ref.aspect_ratio, 1.0),
        rel(wing.sweep_c4_deg, ref.sweep_c4_deg, 10.0),
        rel(wing.taper, ref.taper, 1.0),
        rel(wing.tip_twist_deg, ref.tip_twist_deg, 10.0),
    )
    return float(np.sqrt(sum(t * t for t in terms)))


#: Search grid. Area is held at the reference value, so aspect ratio is exact.
DEFAULT_GRID: dict[str, Sequence] = {
    "aspect_ratio": np.linspace(15.0, 4.0, 23),
    "sweep_c4_deg": np.linspace(0.0, 24.0, 25),
    "taper": np.linspace(0.10, 1.00, 19),
    "tip_twist_deg": np.arange(0.0, -10.0, -1.0),
    "airfoil": ("0012", "2412", "3217", "4417"),
}


def search(
    reference: Wing = BASELINE,
    grid: dict[str, Iterable] | None = None,
    criteria: StallCriteria | None = None,
    n_panels: int = 20,
) -> list[Candidate]:
    """Evaluate every grid combination and return feasible designs sorted by cost.

    The influence matrix depends only on the planform, so it is built once
    per (aspect ratio, sweep, taper) and reused for all twists and airfoils.
    """
    criteria = criteria or StallCriteria()
    g = {**DEFAULT_GRID, **(grid or {})}
    area = reference.area
    twists = np.asarray(list(g["tip_twist_deg"]), dtype=float)
    airfoils = list(g["airfoil"])
    camber = np.array([naca4_camber_slope(a) for a in airfoils])
    alpha = np.radians(criteria.alpha_deg)

    feasible: list[Candidate] = []
    for ar, sweep, taper in itertools.product(g["aspect_ratio"], g["sweep_c4_deg"], g["taper"]):
        planform = Wing.from_planform(area, ar, taper, sweep, 0.0)
        mesh = planform.mesh(n_panels)
        A = influence_matrix(mesh)
        eta = np.abs(mesh.y_mid) / planform.semispan
        # RHS columns: every (twist, airfoil) pair
        tw = np.radians(np.outer(eta, twists))  # (n, n_twist)
        rhs = -(alpha + tw[:, :, None] - camber[None, None, :]).reshape(mesh.n, -1)
        cl_all = 2.0 * np.linalg.solve(A, rhs) / mesh.chord_mid[:, None]

        for col, (twist, foil) in enumerate(itertools.product(twists, airfoils)):
            onset, grad, _ = stall_onset(mesh.y_mid, cl_all[:, col], planform.semispan)
            if onset <= criteria.max_onset and grad <= criteria.max_gradient:
                wing = planform.with_(tip_twist_deg=float(twist), airfoil=foil, reference_area=area)
                right = mesh.y_mid >= 0
                feasible.append(
                    Candidate(
                        wing=wing,
                        onset=onset,
                        gradient=grad,
                        cost=_cost(wing, reference),
                        y_right=mesh.y_mid[right],
                        cl_right=cl_all[right, col],
                    )
                )
    feasible.sort(key=lambda c: c.cost)
    return feasible
