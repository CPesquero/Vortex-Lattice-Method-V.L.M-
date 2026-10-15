import numpy as np
import pytest

import vlmwing as v


@pytest.fixture(scope="session")
def baseline_solution():
    """Baseline wing at the five study angles, 20 spanwise panels (as in the report)."""
    return v.solve(v.BASELINE, np.array([-12.0, -6.0, 0.0, 6.0, 12.0]), 20)
