import numpy as np
import pytest

from fabguard.calibration import InsufficientBaseline, calibrated_threshold


def test_noisy_machine_raises_its_threshold():
    scores = np.linspace(0.0, 1.0, 200)
    assert calibrated_threshold(scores, 0.5) == pytest.approx(np.quantile(scores, 0.99))


def test_quiet_machine_keeps_the_fleet_threshold():
    assert calibrated_threshold(np.full(200, 0.1), 0.5) == 0.5


def test_short_or_invalid_baselines_are_refused():
    with pytest.raises(InsufficientBaseline):
        calibrated_threshold([0.1] * 19, 0.5)
    with pytest.raises(InsufficientBaseline):
        calibrated_threshold([0.1] * 199 + [float("nan")], 0.5)
