import numpy as np
import pytest
from myotrace.physical_units import physical_motion
from myotrace.uncertainty import independent_unit_mean


def test_pixel_interval_motion_becomes_speed_with_acquisition_scale():
    result = physical_motion([1, 2, 3], micrometers_per_pixel=2, fps=10, signal_mode="motion", method="farneback")
    assert np.array_equal(result["motion_speed_um_s"], [20, 40, 60])
    with pytest.raises(ValueError):
        physical_motion([1], micrometers_per_pixel=2, fps=10, signal_mode="motion", method="ensemble")


def test_displacement_and_velocity_use_correct_dimensions():
    result = physical_motion(
        [0, 1, 2], micrometers_per_pixel=2, fps=10, signal_mode="signed_displacement", method="farneback"
    )
    assert np.allclose(result["principal_velocity_um_s"], 20)


def test_nested_observations_do_not_change_biological_n_or_unit_weighting():
    result = independent_unit_mean([1] * 100 + [3], ["culture-a"] * 100 + ["culture-b"], n_boot=100)
    assert result["estimate"] == 2 and result["n_independent"] == 2
    with pytest.raises(ValueError):
        independent_unit_mean([1, 2], ["a", "a"])
