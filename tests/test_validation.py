import numpy as np
import pytest

import vlmwing as v
from vlmwing.cli import main


def test_xflr5_file_loads_all_rows():
    y, cl = v.load_xflr5()
    assert y.shape == (20,)
    assert set(cl) == set(v.XFLR5_ALPHAS)
    np.testing.assert_allclose(y, -y[::-1])


def test_xflr5_stations_coincide_with_20_panel_mesh():
    y, _ = v.load_xflr5()
    np.testing.assert_allclose(y, v.BASELINE.mesh(20).y_mid, atol=1e-5)


def test_lift_slope_agrees_with_xflr5():
    y, cl = v.load_xflr5()
    m = v.BASELINE.mesh(20)
    alphas = np.array(v.XFLR5_ALPHAS)
    CL_x = [np.sum(cl[a] * m.chord_mid * m.dy) / v.BASELINE.area for a in alphas]
    slope_x = np.polyfit(alphas, CL_x, 1)[0]
    slope = v.fit_lift_curve(v.solve(v.BASELINE, alphas)).slope_per_deg
    assert slope == pytest.approx(slope_x, rel=0.05)


def test_load_xflr5_rejects_wrong_shape(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("0 1 2\n3 4 5\n")
    with pytest.raises(ValueError):
        v.load_xflr5(p)


def test_cli_tables_only(capsys, tmp_path):
    out = tmp_path / "res.json"
    assert main(["--no-figures", "--no-optimize", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "0.0932" in text
    assert out.exists()
