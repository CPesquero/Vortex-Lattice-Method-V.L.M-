"""Generate the vector figures used by report.tex (run from the repository root).

python docs/report/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import vlmwing as v  # noqa: E402
from vlmwing import plots  # noqa: E402

OUT = Path(__file__).parent / "figures"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    wing = v.BASELINE
    alphas = np.array(v.XFLR5_ALPHAS)
    sol = v.solve(wing, alphas, 20)
    fit = v.fit_lift_curve(sol)
    y_x, cl_x = v.load_xflr5()
    crit = v.StallCriteria()
    best = v.search(wing, criteria=crit)[0]

    figs = {
        "airfoil": plots.airfoil(wing.airfoil),
        "planform": plots.planform(wing, 20),
        "lift_distribution": plots.lift_distribution(sol),
        "lift_curve": plots.lift_curve(sol, fit),
        "basic_additional_lift": plots.basic_additional(*v.basic_additional_lift(wing)),
        "induced_drag": plots.induced_drag(sol),
        "pitching_moment": plots.pitching_moment(sol),
        "xflr5_comparison": plots.xflr5_comparison(sol, y_x, cl_x),
        "mesh_convergence": plots.mesh_convergence(v.mesh_convergence(wing)),
        "optimized_lift": plots.optimized_lift(best, crit, v.solve(wing, crit.alpha_deg)),
        "planform_comparison": plots.planform_comparison(wing, best.wing),
    }
    for name, fig in figs.items():
        fig.savefig(OUT / f"{name}.pdf")
        plt.close(fig)
    print(f"{len(figs)} figures written to {OUT}")


if __name__ == "__main__":
    main()
