"""Reproduce every table and figure of the study.

Usage::

    vlmwing                     # all sections, figures into ./figures
    vlmwing --no-optimize       # skip the planform search
    vlmwing --figures-dir out   # choose the output folder
    vlmwing --no-figures        # tables only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import __version__
from .geometry import BASELINE
from .optimization import StallCriteria, search, stall_onset
from .solver import basic_additional_lift, fit_lift_curve, solve
from .validation import XFLR5_ALPHAS, load_xflr5, mesh_convergence

ALPHAS = np.array(XFLR5_ALPHAS)


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def run(figures_dir: Path | None, optimize: bool, n_panels: int, top: int) -> dict:
    wing = BASELINE
    results: dict = {}
    figs: dict = {}
    if figures_dir is not None:
        from . import plots  # imported lazily: matplotlib only when figures are wanted

    _rule("Geometry")
    geo = {
        "span_m": wing.span,
        "root_chord_m": wing.root_chord,
        "tip_chord_m": wing.tip_chord,
        "taper": wing.taper,
        "aspect_ratio": wing.aspect_ratio,
        "sweep_c4_deg": wing.sweep_c4_deg,
        "sweep_le_deg": wing.sweep_le_deg,
        "mac_m": wing.mac,
        "x_ac_m": wing.x_ac,
        "y_ac_m": wing.y_ac,
        "camber_slope_3c4": wing.camber_slope,
    }
    for k, v in geo.items():
        print(f"{k:<18} {v:9.4f}")
    results["geometry"] = geo

    sol = solve(wing, ALPHAS, n_panels)
    _rule(f"Sectional lift c_l(y), right half-wing, {n_panels} panels")
    right = sol.mesh.y_mid >= 0
    print("y [m]   " + "".join(f"{a:>8g}°" for a in ALPHAS))
    for i in np.flatnonzero(right):
        print(f"{sol.mesh.y_mid[i]:6.3f}  " + "".join(f"{v:9.3f}" for v in sol.cl[i]))

    fit = fit_lift_curve(sol)
    _rule("Global lift")
    print(f"CL_alpha      {fit.slope_per_deg:.4f} /deg   ({fit.slope_per_rad:.3f} /rad)")
    print(f"CL(alpha=0)   {fit.cl_at_zero_alpha:.4f}")
    print(f"alpha_L=0     {fit.zero_lift_alpha_deg:.3f} deg")
    results["lift"] = {
        "CL_alpha_per_deg": fit.slope_per_deg,
        "CL_at_alpha0": fit.cl_at_zero_alpha,
        "alpha_L0_deg": fit.zero_lift_alpha_deg,
    }

    _rule("Induced drag and pitching moment")
    print("alpha    CL       CDi      CM_0     CM_ac")
    for k, a in enumerate(ALPHAS):
        print(f"{a:5g}  {sol.CL[k]:7.3f}  {sol.CDi[k]:7.4f}  {sol.CM_origin[k]:7.3f}  {sol.CM_ac[k]:7.3f}")
    results["polar"] = {
        "alpha_deg": ALPHAS.tolist(),
        "CL": sol.CL.tolist(),
        "CDi": sol.CDi.tolist(),
        "CM_origin": sol.CM_origin.tolist(),
        "CM_ac": sol.CM_ac.tolist(),
    }

    y_ba, cl_b, cl_a = basic_additional_lift(wing, n_panels)

    _rule("Comparison with XFLR5 (same 20 stations)")
    y_x, cl_x = load_xflr5()
    sol20 = solve(wing, ALPHAS, len(y_x))
    offsets = [float(np.mean(sol20.cl[:, k] - cl_x[a])) for k, a in enumerate(ALPHAS)]
    CL_x = [float(np.sum(cl_x[a] * sol20.mesh.chord_mid * sol20.mesh.dy) / wing.area) for a in ALPHAS]
    slope_x = float(np.polyfit(ALPHAS, CL_x, 1)[0])
    print("mean c_l offset (VLM - XFLR5): " + ", ".join(f"{o:+.3f}" for o in offsets))
    print(
        f"lift slope: VLM {fit.slope_per_deg:.4f} /deg, XFLR5 {slope_x:.4f} /deg "
        f"({100 * (fit.slope_per_deg / slope_x - 1):+.1f} %)"
    )
    results["xflr5"] = {"mean_cl_offset": offsets, "CL_alpha_per_deg": slope_x}

    _rule("Mesh convergence, alpha = 6 deg")
    conv = mesh_convergence(wing)
    print("panels   CL        time [ms]")
    for p in conv:
        print(f"{p.n_panels:6d}  {p.CL:.4f}   {1e3 * p.seconds:8.3f}")
    results["mesh_convergence"] = [p.__dict__ for p in conv]

    if figures_dir is not None:
        figs.update(
            airfoil=plots.airfoil(wing.airfoil),
            planform=plots.planform(wing, n_panels),
            lift_distribution=plots.lift_distribution(sol),
            lift_curve=plots.lift_curve(sol, fit),
            basic_additional_lift=plots.basic_additional(y_ba, cl_b, cl_a),
            induced_drag=plots.induced_drag(sol),
            pitching_moment=plots.pitching_moment(sol),
            xflr5_comparison=plots.xflr5_comparison(sol20, y_x, cl_x),
            mesh_convergence=plots.mesh_convergence(conv),
        )

    if optimize:
        criteria = StallCriteria()
        _rule(
            f"Stall redesign: onset <= {criteria.max_onset:.0%} semispan, "
            f"dCl/dy <= {criteria.max_gradient} /m at alpha = {criteria.alpha_deg:g} deg"
        )
        t0 = time.perf_counter()
        cands = search(wing, criteria=criteria, n_panels=20)
        print(f"{len(cands)} feasible designs in {time.perf_counter() - t0:.1f} s")
        print("rank  NACA     A    Lc/4   taper  twist   onset   dCl/dy   cost")
        for r, c in enumerate(cands[:top], 1):
            w = c.wing
            print(
                f"{r:4d}  {w.airfoil}  {w.aspect_ratio:5.1f}  {w.sweep_c4_deg:5.1f}  {w.taper:5.2f}  "
                f"{w.tip_twist_deg:5.1f}  {c.onset:6.0%}  {c.gradient:7.3f}  {c.cost:5.3f}"
            )
        base_sol = solve(wing, criteria.alpha_deg, 20)

        b_onset, b_grad, _ = stall_onset(base_sol.mesh.y_mid, base_sol.cl[:, 0], wing.semispan)
        print(f"baseline: onset {b_onset:.0%}, dCl/dy {b_grad:.3f} /m")
        results["optimization"] = {
            "n_feasible": len(cands),
            "baseline": {"onset": b_onset, "gradient": b_grad},
            "top": [
                {
                    "airfoil": c.wing.airfoil,
                    "aspect_ratio": c.wing.aspect_ratio,
                    "sweep_c4_deg": c.wing.sweep_c4_deg,
                    "taper": c.wing.taper,
                    "tip_twist_deg": c.wing.tip_twist_deg,
                    "onset": c.onset,
                    "gradient": c.gradient,
                    "cost": c.cost,
                }
                for c in cands[:top]
            ],
        }
        if figures_dir is not None and cands:
            figs["optimized_lift"] = plots.optimized_lift(cands[0], criteria, base_sol)
            figs["planform_comparison"] = plots.planform_comparison(wing, cands[0].wing)

    if figures_dir is not None:
        import matplotlib.pyplot as plt

        figures_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(figures_dir / f"{name}.png")
            plt.close(fig)
        print(f"\n{len(figs)} figures written to {figures_dir}/")
    return results


def _round(obj, digits: int = 6):
    """Round floats so the JSON diff only changes when results do."""
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: _round(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, digits) for v in obj]
    return obj


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vlmwing", description=__doc__.split("\n\n")[0])
    p.add_argument("--figures-dir", type=Path, default=Path("figures"))
    p.add_argument("--no-figures", action="store_true", help="print tables only")
    p.add_argument("--no-optimize", action="store_true", help="skip the planform search")
    p.add_argument("--panels", type=int, default=20, help="spanwise panels, whole wing (default 20)")
    p.add_argument("--top", type=int, default=5, help="number of redesigns to list")
    p.add_argument("--json", type=Path, help="also write the numbers to this JSON file")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    a = p.parse_args(argv)
    res = run(None if a.no_figures else a.figures_dir, not a.no_optimize, a.panels, a.top)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(_round(res), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
