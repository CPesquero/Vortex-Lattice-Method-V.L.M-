import numpy as np
import pytest

import vlmwing as v


def _aic_reference(mesh):
    """Element-by-element version of the original notebook (cell 2)."""
    n = mesh.n
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xc, yc = mesh.x_ctrl[i], mesh.y_mid[i]
            x1, y1 = mesh.x_c4_nodes[j], mesh.y_nodes[j]
            x2, y2 = mesh.x_c4_nodes[j + 1], mesh.y_nodes[j + 1]
            a, b = xc - x1, yc - y1
            c, d = xc - x2, yc - y2
            e, f = np.sqrt(a**2 + b**2), np.sqrt(c**2 + d**2)
            g, h = x2 - x1, y2 - y1
            den = a * d - c * b
            k = ((g * a + h * b) / e - (g * c + h * d) / f) / den if abs(den) > 1e-9 else 0.0
            l = -(1 / b) * (1 + a / e) + (1 / d) * (1 + c / f) if abs(b) > 1e-9 and abs(d) > 1e-9 else 0.0
            A[i, j] = (k + l) / (4 * np.pi)
    return A


@pytest.mark.parametrize("n", [4, 20, 40])
def test_vectorized_aic_matches_reference(n):
    mesh = v.BASELINE.mesh(n)
    np.testing.assert_allclose(v.influence_matrix(mesh), _aic_reference(mesh), rtol=1e-12, atol=1e-12)


def test_report_global_lift(baseline_solution):
    fit = v.fit_lift_curve(baseline_solution)
    assert fit.slope_per_deg == pytest.approx(0.0932, abs=1e-4)
    assert fit.slope_per_rad == pytest.approx(5.342, abs=1e-3)
    assert fit.cl_at_zero_alpha == pytest.approx(0.068, abs=1e-3)
    assert fit.zero_lift_alpha_deg == pytest.approx(-0.733, abs=1e-3)


def test_report_sectional_lift(baseline_solution):
    # right-wing root and tip panels at alpha = 12 deg (report, section 4)
    cl12 = baseline_solution.cl[:, -1]
    assert cl12[10] == pytest.approx(1.277, abs=1e-3)
    assert cl12[-1] == pytest.approx(0.810, abs=1e-3)


def test_report_induced_drag(baseline_solution):
    np.testing.assert_allclose(baseline_solution.CDi, [0.0251, 0.0063, 0.0007, 0.0082, 0.0288], atol=1e-4)


def test_report_pitching_moment(baseline_solution):
    np.testing.assert_allclose(baseline_solution.CM_ac, [0.034, 0.040, 0.045, 0.050, 0.055], atol=1e-3)
    np.testing.assert_allclose(baseline_solution.CM_origin, [1.115, 0.545, -0.026, -0.596, -1.166], atol=1e-3)


def test_symmetry(baseline_solution):
    np.testing.assert_allclose(baseline_solution.cl, baseline_solution.cl[::-1], atol=1e-12)


def test_linearity_in_alpha():
    s = v.solve(v.BASELINE, [0.0, 3.0, 6.0])
    np.testing.assert_allclose(s.cl[:, 2] - s.cl[:, 1], s.cl[:, 1] - s.cl[:, 0], atol=1e-12)


def test_flat_untwisted_wing_has_zero_lift_at_zero_alpha():
    w = v.BASELINE.with_(tip_twist_deg=0.0, airfoil="0012")
    s = v.solve(w, 0.0)
    np.testing.assert_allclose(s.cl, 0.0, atol=1e-14)


def test_high_aspect_ratio_tends_to_thin_airfoil_slope():
    w = v.Wing.from_planform(1.0, 200.0, 1.0, 0.0, 0.0, "0012")
    s = v.solve(w, [0.0, 1.0], 40)
    assert np.degrees(s.CL[1] - s.CL[0]) == pytest.approx(2 * np.pi, rel=0.02)


@pytest.mark.parametrize("ar", [4.0, 8.0, 20.0])
def test_finite_wing_slope_is_bounded_by_helmbold(ar):
    """A rectangular wing loses a little more than the elliptic (Helmbold) estimate."""
    w = v.Wing.from_planform(1.0, ar, 1.0, 0.0, 0.0, "0012")
    s = v.solve(w, [0.0, 1.0], 40)
    slope = np.degrees(s.CL[1] - s.CL[0])
    helmbold = 2 * np.pi * ar / (2 + np.sqrt(ar**2 + 4))
    assert 0.90 * helmbold < slope < helmbold


def test_induced_drag_is_non_negative_and_grows_with_lift(baseline_solution):
    CDi = baseline_solution.CDi
    assert np.all(CDi >= 0)
    assert CDi[4] > CDi[3] > CDi[2]


def test_mesh_convergence_is_monotonic():
    cl = [v.solve(v.BASELINE, 6.0, n).CL[0] for n in (10, 20, 40, 80, 160)]
    assert np.all(np.diff(cl) < 0)
    assert abs(cl[-1] - cl[-2]) / cl[-1] < 0.005  # < 0.5 % from 80 panels on


def test_basic_and_additional_lift_normalization():
    w = v.BASELINE
    y, cl_b, cl_a = v.basic_additional_lift(w)
    m = w.mesh(20)
    assert np.sum(cl_b * m.chord_mid * m.dy) / w.area == pytest.approx(0.0, abs=1e-12)
    assert np.sum(cl_a * m.chord_mid * m.dy) / w.area == pytest.approx(1.0)
    # washout unloads the tips at zero lift
    assert cl_b[10] > 0 > cl_b[-1]


def test_fit_needs_two_angles():
    with pytest.raises(ValueError):
        v.fit_lift_curve(v.solve(v.BASELINE, 0.0))
