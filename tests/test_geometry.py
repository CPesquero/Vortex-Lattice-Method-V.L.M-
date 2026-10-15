import numpy as np
import pytest

import vlmwing as v


def test_baseline_matches_report():
    w = v.BASELINE
    assert w.aspect_ratio == pytest.approx(15.0, abs=0.02)
    assert w.taper == pytest.approx(0.6486, abs=1e-4)
    assert w.mac == pytest.approx(0.2619, abs=1e-4)
    assert w.y_ac == pytest.approx(0.8994, abs=1e-4)
    assert w.x_ac == pytest.approx(0.2694, abs=1e-4)
    assert w.area == 1.0


def test_from_planform_round_trip():
    w = v.Wing.from_planform(area=1.0, aspect_ratio=15.0, taper=0.65, sweep_c4_deg=12.0, tip_twist_deg=-5.0)
    assert w.planform_area == pytest.approx(1.0)
    assert w.aspect_ratio == pytest.approx(15.0)
    assert w.taper == pytest.approx(0.65)
    assert w.span == pytest.approx(v.BASELINE.span, abs=1e-3)


def test_leading_edge_sweep_exceeds_quarter_chord_sweep_for_tapered_wing():
    w = v.BASELINE
    assert w.sweep_le_deg == pytest.approx(12.776, abs=1e-3)
    rect = w.with_(tip_chord=w.root_chord)
    assert rect.sweep_le_deg == pytest.approx(rect.sweep_c4_deg)


@pytest.mark.parametrize(
    ("naca", "expected"),
    [
        ("0012", 0.0),
        ("3217", 2 * 0.03 / 0.8**2 * (0.2 - 0.75)),
        ("2412", 2 * 0.02 / 0.6**2 * (0.4 - 0.75)),
    ],
)
def test_camber_slope(naca, expected):
    assert v.naca4_camber_slope(naca) == pytest.approx(expected)


def test_camber_slope_forward_of_max_camber():
    assert v.naca4_camber_slope("2412", 0.1) == pytest.approx(2 * 0.02 / 0.4**2 * 0.3)


@pytest.mark.parametrize("bad", ["abc", "123", "12345", "2012"])
def test_invalid_airfoil(bad):
    with pytest.raises(ValueError):
        v.naca4_camber_slope(bad)


def test_mesh_layout():
    m = v.BASELINE.mesh(20)
    assert m.n == 20
    assert m.dy.sum() == pytest.approx(v.BASELINE.span)
    np.testing.assert_allclose(m.y_mid, -m.y_mid[::-1])
    # control points lie half a chord behind the bound vortex
    np.testing.assert_allclose(m.x_ctrl - m.x_c4_mid, 0.5 * m.chord_mid)


@pytest.mark.parametrize("n", [0, 3, 21])
def test_mesh_rejects_odd_or_tiny(n):
    with pytest.raises(ValueError):
        v.BASELINE.mesh(n)
