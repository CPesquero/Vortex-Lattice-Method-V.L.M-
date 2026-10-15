"""Figures for the study. Every function returns a Matplotlib ``Figure``."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .geometry import Wing
from .optimization import Candidate, StallCriteria
from .solver import LiftCurve, Solution
from .validation import ConvergencePoint

__all__ = [
    "airfoil",
    "planform",
    "lift_distribution",
    "lift_curve",
    "basic_additional",
    "induced_drag",
    "pitching_moment",
    "xflr5_comparison",
    "mesh_convergence",
    "optimized_lift",
    "planform_comparison",
]

STYLE = {
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "legend.frameon": False,
}
plt.rcParams.update(STYLE)


def _alpha_colors(alphas: Sequence[float]) -> list:
    cmap = plt.get_cmap("viridis")
    lo, hi = min(alphas), max(alphas)
    return [cmap(0.45 if hi == lo else 0.9 * (a - lo) / (hi - lo)) for a in alphas]


def _naca4(naca: str, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    m, p, t = int(naca[0]) / 100, int(naca[1]) / 10, int(naca[2:]) / 100
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    if m == 0:
        yc, dyc = np.zeros_like(x), np.zeros_like(x)
    else:
        fwd = x < p
        yc = np.where(fwd, m / p**2 * (2 * p * x - x**2), m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x**2))
        dyc = np.where(fwd, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    return np.concatenate([xu[::-1], xl[1:]]), np.concatenate([yu[::-1], yl[1:]])


def airfoil(naca: str) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 2.6))
    x, y = _naca4(naca)
    ax.fill(x, y, color="C0", alpha=0.15)
    ax.plot(x, y, color="C0")
    ax.axvline(0.75, color="0.4", ls="--", lw=0.8)
    ax.annotate(
        "control point, x/c = 0.75",
        (0.75, -0.09),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=8,
        color="0.3",
    )
    ax.set_aspect("equal")
    ax.set(xlabel="x/c", ylabel="y/c", title=f"NACA {naca}")
    return fig


def planform(wing: Wing, n_panels: int = 20) -> Figure:
    mesh = wing.mesh(n_panels)
    y = mesh.y_nodes
    x_le, x_te = wing.x_le(y), wing.x_le(y) + wing.chord(y)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.vlines(y, x_le, x_te, color="C0", lw=0.7, label="panel edges")
    ax.plot(y, x_le, "k", lw=1.6)
    ax.plot(y, x_te, "k", lw=1.6)
    ax.plot(mesh.y_nodes, mesh.x_c4_nodes, color="C2", lw=1, ls="--", label="bound vortices (c/4)")
    ax.plot(mesh.y_mid, mesh.x_ctrl, "x", color="C3", ms=4, label="control points (3c/4)")
    for s in (-1, 1):
        x0 = wing.y_ac * wing.tan_sweep_le
        ax.vlines(
            s * wing.y_ac,
            x0,
            x0 + wing.mac,
            color="C1",
            lw=2.2,
            label="mean aerodynamic chord" if s > 0 else None,
        )
        ax.plot(
            s * wing.y_ac, wing.x_ac, "o", color="C1", ms=6, label="aerodynamic center" if s > 0 else None
        )
    ax.set_aspect("equal")
    ax.set_ylim(x_te.max() + 0.75, -0.08)
    ax.set(
        xlabel="y [m]",
        ylabel="x [m]",
        title=f"Planform and lattice: A = {wing.aspect_ratio:.1f}, λ = {wing.taper:.2f}, "
        f"Λc/4 = {wing.sweep_c4_deg:.0f}°",
    )
    ax.legend(loc="lower center", ncol=3, fontsize=8)
    return fig


def lift_distribution(sol: Solution) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for k, (a, c) in enumerate(zip(sol.alphas_deg, _alpha_colors(sol.alphas_deg))):
        ax.plot(sol.mesh.y_mid, sol.cl[:, k], "-o", ms=3, color=c, label=f"α = {a:g}°")
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xlabel="y [m]", ylabel="$c_l$", title="Spanwise sectional lift")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return fig


def lift_curve(sol: Solution, fit: LiftCurve) -> Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    a = np.linspace(sol.alphas_deg.min(), sol.alphas_deg.max(), 50)
    ax.plot(a, fit.cl_at_zero_alpha + fit.slope_per_deg * a, color="C0", lw=1)
    ax.plot(sol.alphas_deg, sol.CL, "o", color="C0", label="VLM")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.plot(
        fit.zero_lift_alpha_deg, 0, "s", color="C3", label=f"$α_{{L=0}}$ = {fit.zero_lift_alpha_deg:.2f}°"
    )
    ax.text(
        0.04,
        0.95,
        f"$C_{{Lα}}$ = {fit.slope_per_deg:.4f} /deg = {fit.slope_per_rad:.3f} /rad\n"
        f"$C_{{L,α=0}}$ = {fit.cl_at_zero_alpha:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    ax.set(xlabel="α [deg]", ylabel="$C_L$", title="Wing lift curve")
    ax.legend(loc="lower right")
    return fig


def basic_additional(y: np.ndarray, cl_b: np.ndarray, cl_a: np.ndarray) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(y, cl_b, "-o", ms=3, color="C3", label="basic, $c_{l,b}$ ($C_L$ = 0)")
    ax.plot(y, cl_a, "-s", ms=3, color="C2", label="additional, $c_{l,a}$ ($C_L$ = 1)")
    ax.plot(y, cl_b + cl_a, "--", color="C0", label="total at $C_L$ = 1")
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xlabel="y [m]", ylabel="$c_l$", title="Basic and additional lift")
    ax.legend()
    return fig


def induced_drag(sol: Solution) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for k, (a, c) in enumerate(zip(sol.alphas_deg, _alpha_colors(sol.alphas_deg))):
        ax.plot(
            sol.mesh.y_mid,
            sol.cdi[:, k],
            "-o",
            ms=3,
            color=c,
            label=f"α = {a:g}°, $C_{{Di}}$ = {sol.CDi[k]:.4f}",
        )
    ax.set(xlabel="y [m]", ylabel="$c_{di}$", title="Spanwise induced drag (Trefftz plane)")
    ax.legend(fontsize=8)
    return fig


def pitching_moment(sol: Solution) -> Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(sol.alphas_deg, sol.CM_origin, "-o", label="about root leading edge, $C_{M,0}$")
    ax.plot(sol.alphas_deg, sol.CM_ac, "-s", label="about aerodynamic center, $C_{M,ac}$")
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xlabel="α [deg]", ylabel="$C_M$", title="Pitching moment")
    ax.legend()
    return fig


def xflr5_comparison(sol: Solution, y_ref: np.ndarray, cl_ref: dict) -> Figure:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for k, (a, c) in enumerate(zip(sol.alphas_deg, _alpha_colors(sol.alphas_deg))):
        ax.plot(sol.mesh.y_mid, sol.cl[:, k], "-", color=c, lw=1.6, label=f"VLM α = {a:g}°")
        ax.plot(y_ref, cl_ref[float(a)], "o", color=c, ms=4, mfc="none")
    ax.plot([], [], "o", color="0.3", mfc="none", label="XFLR5")
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xlabel="y [m]", ylabel="$c_l$", title="Comparison with XFLR5 (lines: this code, circles: XFLR5)")
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return fig


def mesh_convergence(points: Sequence[ConvergencePoint]) -> Figure:
    n = [p.n_panels for p in points]
    fig, ax1 = plt.subplots(figsize=(7, 4.6))
    ax1.plot(n, [p.CL for p in points], "-o", color="C0")
    ax1.set_xscale("log")
    ax1.set_xlabel("spanwise panels, whole wing")
    ax1.set_ylabel("$C_L$ at α = 6°", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(n, [1e3 * p.seconds for p in points], "-s", color="C3")
    ax2.set_yscale("log")
    ax2.set_ylabel("solve time [ms]", color="C3")
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax1.set_title("Mesh convergence")
    return fig


def optimized_lift(best: Candidate, criteria: StallCriteria, baseline: Solution | None = None) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    eta = best.y_right / best.wing.semispan
    w = best.wing
    target = f"target: onset ≤ {criteria.max_onset:.0%}"
    ax.axvspan(0, criteria.max_onset, color="C2", alpha=0.08, label=target)
    if baseline is not None:
        right = baseline.mesh.y_mid >= 0
        ax.plot(
            baseline.mesh.y_mid[right] / baseline.wing.semispan,
            baseline.cl[right, 0],
            "--",
            color="0.5",
            label="baseline",
        )
    ax.plot(
        eta,
        best.cl_right,
        "-o",
        ms=3,
        color="C4",
        label=f"redesign: NACA {w.airfoil}, A = {w.aspect_ratio:.1f}, Λc/4 = {w.sweep_c4_deg:.0f}°, "
        f"λ = {w.taper:.2f}, ε = {w.tip_twist_deg:.0f}°",
    )
    k = int(np.argmax(best.cl_right))
    ax.plot(eta[k], best.cl_right[k], "ko", ms=7, label=f"$c_l$ peak at {best.onset:.0%} of semispan")
    ax.set(
        xlabel="y / (b/2)",
        ylabel="$c_l$",
        xlim=(0, 1),
        title=f"Lift distribution at α = {criteria.alpha_deg:g}°",
    )
    ax.legend(fontsize=8, loc="lower left")
    return fig


def planform_comparison(baseline: Wing, redesign: Wing) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    x_max = 0.0
    for wing, style, label in ((baseline, "-", "baseline"), (redesign, "-", "redesign")):
        y = np.array([-wing.semispan, 0.0, wing.semispan])
        x_le, x_te = wing.x_le(y), wing.x_le(y) + wing.chord(y)
        xs = np.concatenate([x_le, x_te[::-1], x_le[:1]])
        ys = np.concatenate([y, y[::-1], y[:1]])
        ax.fill(ys, xs, alpha=0.15)
        ax.plot(ys, xs, style, lw=1.5, label=f"{label} (A = {wing.aspect_ratio:.1f})")
        x_max = max(x_max, xs.max())
    ax.set_aspect("equal")
    ax.set_ylim(x_max + 0.45, -0.08)
    ax.set(xlabel="y [m]", ylabel="x [m]", title="Planforms at equal area")
    ax.legend(loc="lower center", ncol=2)
    return fig
