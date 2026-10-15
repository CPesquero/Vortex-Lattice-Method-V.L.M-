import numpy as np
import pytest

import vlmwing as v


def test_stall_onset_on_synthetic_distribution():
    y = np.linspace(-0.95, 0.95, 20)
    cl = 1.0 - (np.abs(y) - 0.25) ** 2
    onset, grad, k = v.stall_onset(y, cl, 1.0)
    assert onset == pytest.approx(y[y >= 0][k])
    assert 0.2 <= onset <= 0.3
    assert grad < 0


def test_baseline_fails_gradient_criterion():
    s = v.solve(v.BASELINE, 15.0)
    onset, grad, _ = v.stall_onset(s.mesh.y_mid, s.cl[:, 0], v.BASELINE.semispan)
    assert onset <= 0.40
    assert grad > -0.20


def test_search_returns_sorted_feasible_designs():
    grid = {
        "aspect_ratio": [15.0, 11.5, 8.0],
        "sweep_c4_deg": [3.0, 12.0],
        "taper": [0.65, 1.0],
        "tip_twist_deg": [-5.0, -8.0],
        "airfoil": ["0012", "3217"],
    }
    crit = v.StallCriteria()
    cands = v.search(v.BASELINE, grid=grid, criteria=crit)
    assert cands, "expected at least one feasible design on this grid"
    costs = [c.cost for c in cands]
    assert costs == sorted(costs)
    for c in cands:
        assert c.onset <= crit.max_onset
        assert c.gradient <= crit.max_gradient
        assert c.wing.planform_area == pytest.approx(v.BASELINE.area)
        # re-solve independently and confirm the stored distribution
        s = v.solve(c.wing, crit.alpha_deg)
        np.testing.assert_allclose(s.cl[s.mesh.y_mid >= 0, 0], c.cl_right, atol=1e-10)


def test_baseline_has_zero_cost():
    grid = {
        "aspect_ratio": [15.0],
        "sweep_c4_deg": [12.0],
        "taper": [0.65],
        "tip_twist_deg": [-5.0],
        "airfoil": ["3217"],
    }
    loose = v.StallCriteria(max_onset=1.0, max_gradient=10.0)
    (c,) = v.search(v.Wing.from_planform(1.0, 15.0, 0.65, 12.0, -5.0), grid=grid, criteria=loose)
    assert c.cost == pytest.approx(0.0, abs=1e-12)
