import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from fabguard.paderborn import (
    BEARINGS,
    HEALTHY,
    bearing_number,
    calibration_check,
    outer_folds,
    parse_recording,
    read_vibration,
)


def test_recording_names_and_bearing_ids_parse():
    meta = parse_recording("N09_M07_F10_KA04_17")
    assert meta == {
        "code": "KA04",
        "rpm": 900,
        "torque_nm": 0.7,
        "radial_force_n": 1000,
        "measurement": 17,
        "condition": "N09_M07_F10",
    }
    assert [bearing_number(c) for c in ("K001", "KA01", "KI21", "KB23")] == [1, 101, 221, 323]
    with pytest.raises(ValueError):
        parse_recording("../K001_1")


def test_outer_folds_hold_out_every_bearing_once_with_both_labels():
    folds = outer_folds()
    held = [b for fold in folds for b in fold]
    assert sorted(held) == sorted(bearing_number(c) for c in BEARINGS)
    healthy = {bearing_number(c) for c in HEALTHY}
    for fold in folds:
        assert len(healthy & set(fold)) == 1 and len(set(fold) - healthy) >= 4


def test_read_vibration_picks_the_vibration_channel(tmp_path):
    channels = np.zeros(2, dtype=[("Name", object), ("Data", object)])
    channels[0] = ("force", np.ones(4))
    channels[1] = ("vibration_1", np.arange(5.0))
    path = tmp_path / "N15_M07_F10_K001_1.mat"
    savemat(path, {"N15_M07_F10_K001_1": {"Y": channels}})
    assert read_vibration(path).tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_calibration_check_compares_fleet_and_machine_thresholds():
    rows = []
    for measurement in range(1, 21):
        for code, label, origin, level in (
            ("K001", 0, "none", 0.6),  # noisy healthy bearing above the fleet threshold
            ("KI04", 1, "real", 0.9),
        ):
            for window in range(13):
                rows.append(
                    {
                        "fold": "bearings-001",
                        "recording_id": f"N15_M07_F10_{code}_{measurement}",
                        "bearing_code": code,
                        "binary_label": label,
                        "damage_origin": origin,
                        "window_index": window,
                        "anomaly_score": level + 0.001 * window,
                        "threshold": 0.5,
                    }
                )
    # 10 baseline measurements x 13 windows = 130, over the 120-window minimum
    result = calibration_check(pd.DataFrame(rows)).iloc[0]
    assert result.fleet_healthy_alarmed == 1.0
    assert result.machine_healthy_alarmed == 0.0
    assert result.machine_real_recall == 1.0



def test_unreadable_recordings_are_excluded_and_reported(tmp_path, monkeypatch):
    from fabguard import paderborn
    from fabguard.config import ExperimentConfig

    files = [
        tmp_path / f"{condition}_KA08_{measurement}.mat"
        for condition in ("N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04")
        for measurement in range(1, 21)
    ]
    broken = {files[3].stem}

    def fake_read(path):
        if path.stem in broken:
            raise TypeError("Expecting matrix here")
        return np.sin(np.arange(2 * paderborn.SAMPLE_RATE_HZ) / 7.0)

    monkeypatch.setattr(paderborn, "read_vibration", fake_read)
    table, excluded = paderborn.bearing_features("KA08", files, ExperimentConfig())
    assert [item["recording_id"] for item in excluded] == list(broken)
    assert "Expecting matrix" in excluded[0]["reason"]
    assert table.recording_id.nunique() == 79

    broken.update(path.stem for path in files[4:6])
    with pytest.raises(ValueError, match="3 unreadable"):
        paderborn.bearing_features("KA08", files, ExperimentConfig())
