"""Vortex lattice solver: influence matrix, circulation and aerodynamic coefficients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Mesh, Wing

__all__ = [
    "influence_matrix",
    "trefftz_matrix",
    "solve",
    "Solution",
    "LiftCurve",
    "fit_lift_curve",
    "basic_additional_lift",
]

_EPS = 1e-9


def influence_matrix(mesh: Mesh) -> np.ndarray:
    """Normal velocity at each control point induced by each unit horseshoe vortex.

    Entry ``[i, j]`` is the downwash (per unit circulation) at control point
    ``i`` due to the horseshoe vortex of panel ``j``: bound segment on the
    quarter-chord line plus two trailing legs to +infinity along x.
    Vectorized Biot-Savart law for a planar horseshoe vortex (Bertin & Cummings notation).
    """
    xc = mesh.x_ctrl[:, None]
    yc = mesh.y_mid[:, None]
    x1, y1 = mesh.x_c4_nodes[None, :-1], mesh.y_nodes[None, :-1]
    x2, y2 = mesh.x_c4_nodes[None, 1:], mesh.y_nodes[None, 1:]

    a, b = xc - x1, yc - y1
    c, d = xc - x2, yc - y2
    e, f = np.hypot(a, b), np.hypot(c, d)
    g, h = x2 - x1, y2 - y1

    with np.errstate(divide="ignore", invalid="ignore"):
        den = a * d - c * b
        bound = ((g * a + h * b) / e - (g * c + h * d) / f) / den
        bound = np.where(np.abs(den) > _EPS, bound, 0.0)
        trailing = -(1.0 + a / e) / b + (1.0 + c / f) / d
        trailing = np.where((np.abs(b) > _EPS) & (np.abs(d) > _EPS), trailing, 0.0)
    return (bound + trailing) / (4.0 * np.pi)


def trefftz_matrix(mesh: Mesh) -> np.ndarray:
    """Far-wake (Trefftz-plane) downwash at each station per unit circulation."""
    ym = mesh.y_mid[:, None]
    return ((1.0 / (ym - mesh.y_nodes[None, :-1])) - (1.0 / (ym - mesh.y_nodes[None, 1:]))) / (2.0 * np.pi)


@dataclass(frozen=True)
class Solution:
    """VLM result for one wing at one or more angles of attack.

    Arrays indexed ``[panel, alpha]``. Circulation is normalized by the
    freestream speed (Gamma / U_inf).
    """

    wing: Wing
    mesh: Mesh
    alphas_deg: np.ndarray
    gamma: np.ndarray

    # ---- local (sectional) quantities ----------------------------------
    @property
    def cl(self) -> np.ndarray:
        """Sectional lift coefficient (Kutta-Joukowski)."""
        return 2.0 * self.gamma / self.mesh.chord_mid[:, None]

    @property
    def alpha_induced(self) -> np.ndarray:
        """Induced angle [rad] from the Trefftz-plane downwash (half the far-wake value)."""
        return 0.5 * (trefftz_matrix(self.mesh) @ self.gamma)

    @property
    def cdi(self) -> np.ndarray:
        """Sectional induced-drag coefficient."""
        return self.cl * self.alpha_induced

    # ---- global coefficients ------------------------------------------
    def _integrate(self, sectional: np.ndarray) -> np.ndarray:
        return (sectional * (self.mesh.chord_mid * self.mesh.dy)[:, None]).sum(axis=0) / self.wing.area

    @property
    def CL(self) -> np.ndarray:
        return self._integrate(self.cl)

    @property
    def CDi(self) -> np.ndarray:
        return self._integrate(self.cdi)

    @property
    def CM_origin(self) -> np.ndarray:
        """Pitching moment about the root leading edge (nose-up positive), ref. chord = MAC."""
        return self._integrate(-self.cl * self.mesh.x_c4_mid[:, None]) / self.wing.mac

    @property
    def CM_ac(self) -> np.ndarray:
        """Pitching moment transferred to the wing aerodynamic center."""
        return self.CM_origin + self.CL * self.wing.x_ac / self.wing.mac


def solve(
    wing: Wing,
    alphas_deg: float | np.ndarray,
    n_panels: int = 20,
    *,
    aic: np.ndarray | None = None,
) -> Solution:
    """Solve the flow-tangency condition at every control point.

    The right-hand side includes the angle of attack, the local geometric
    twist and the camber-line slope at the 3/4-chord point.
    """
    alphas = np.atleast_1d(np.asarray(alphas_deg, dtype=float))
    mesh = wing.mesh(n_panels)
    A = influence_matrix(mesh) if aic is None else aic
    rhs = -(np.radians(alphas)[None, :] + wing.twist_rad(mesh.y_mid)[:, None] - wing.camber_slope)
    gamma = np.linalg.solve(A, rhs)
    return Solution(wing, mesh, alphas, gamma)


@dataclass(frozen=True)
class LiftCurve:
    slope_per_deg: float
    cl_at_zero_alpha: float

    @property
    def slope_per_rad(self) -> float:
        return float(np.degrees(self.slope_per_deg))

    @property
    def zero_lift_alpha_deg(self) -> float:
        return -self.cl_at_zero_alpha / self.slope_per_deg


def fit_lift_curve(solution: Solution) -> LiftCurve:
    """Least-squares straight line through CL(alpha)."""
    if solution.alphas_deg.size < 2:
        raise ValueError("need at least two angles of attack")
    slope, intercept = np.polyfit(solution.alphas_deg, solution.CL, 1)
    return LiftCurve(float(slope), float(intercept))


def basic_additional_lift(wing: Wing, n_panels: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the load into basic (CL = 0) and unit additional (CL = 1) distributions.

    Returns ``(y_mid, cl_basic, cl_additional)``.
    """
    mesh = wing.mesh(n_panels)
    A = influence_matrix(mesh)
    # Linearity: cl(alpha) = cl(0) + alpha * dcl/dalpha.
    cl0 = solve(wing, 0.0, n_panels, aic=A)
    gamma_unit = np.linalg.solve(A, -np.ones(mesh.n))
    cl_unit = 2.0 * gamma_unit / mesh.chord_mid
    CL_unit = float((cl_unit * mesh.chord_mid * mesh.dy).sum() / wing.area)
    alpha_l0 = -float(cl0.CL[0]) / CL_unit  # [rad]
    cl_basic = cl0.cl[:, 0] + alpha_l0 * cl_unit
    return mesh.y_mid, cl_basic, cl_unit / CL_unit
