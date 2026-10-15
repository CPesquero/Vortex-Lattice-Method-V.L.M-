"""Vortex lattice method for swept, tapered, twisted wings."""

from .geometry import BASELINE, Mesh, Wing, naca4_camber_slope
from .optimization import Candidate, StallCriteria, search, stall_onset
from .solver import (
    LiftCurve,
    Solution,
    basic_additional_lift,
    fit_lift_curve,
    influence_matrix,
    solve,
    trefftz_matrix,
)
from .validation import XFLR5_ALPHAS, load_xflr5, mesh_convergence

__version__ = "1.1.0"

__all__ = [
    "BASELINE",
    "Candidate",
    "LiftCurve",
    "Mesh",
    "Solution",
    "StallCriteria",
    "Wing",
    "XFLR5_ALPHAS",
    "basic_additional_lift",
    "fit_lift_curve",
    "influence_matrix",
    "load_xflr5",
    "mesh_convergence",
    "naca4_camber_slope",
    "search",
    "solve",
    "stall_onset",
    "trefftz_matrix",
]
