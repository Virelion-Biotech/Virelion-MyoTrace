import numpy as np

from fusion.agreement import bland_altman, coefficient_of_variation
from myotrace.calibration import fit_force_calibration


def test_force_calibration_recovers_linear_relationship() -> None:
    x = np.arange(1, 11, dtype=float)
    y = 2.5 * x + 3.0
    cal = fit_force_calibration(x, y, units="uN")
    assert abs(cal.slope - 2.5) < 1e-10
    assert abs(cal.intercept - 3.0) < 1e-10
    assert cal.r2 > 0.999999


def test_bland_altman_and_cv() -> None:
    a = np.array([10, 12, 11, 13, 12], dtype=float)
    b = a - 1.0
    ba = bland_altman(a, b)
    assert abs(ba.mean_bias - 1.0) < 1e-12
    assert ba.n == 5
    assert coefficient_of_variation(a) > 0
