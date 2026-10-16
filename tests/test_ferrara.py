from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from fabguard.ferrara import (
    RECORD_MINUTES,
    damage_start_hours,
    record_features,
    summarize_trial,
    verdict,
)
from fabguard.ims import COLUMNS, score_bearing


def _trial(hours: float, *, damage_at: float | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    count = int(hours * 60 / RECORD_MINUTES)
    rows = []
    for index in range(count):
        values = rng.normal(1.0, 0.05, len(COLUMNS))
        if damage_at is not None and index * RECORD_MINUTES / 60 >= damage_at:
            values *= 5.0
        row = {
            "trial": "E9",
            "record": index + 1,
            "timestamp": datetime(2000, 1, 1) + timedelta(minutes=RECORD_MINUTES * index),
        }
        row.update(dict(zip(COLUMNS, values, strict=True)))
        rows.append(row)
    return pd.DataFrame(rows)


def test_damage_start_is_where_rms_stays_doubled():
    frame = _trial(120, damage_at=100)
    # A 12-record median averages its 6th and 7th values, so it crosses on the 6th raised record.
    assert damage_start_hours(frame) == pytest.approx(100 + 5 * RECORD_MINUTES / 60)
    assert np.isnan(damage_start_hours(_trial(120)))


def test_baseline_counts_five_minute_records():
    scored = score_bearing(_trial(120), "healthy_reference", "first_72h", RECORD_MINUTES)
    assert scored.baseline_snapshots.iloc[0] == 72 * 12
    assert scored.hours.min() == pytest.approx(72)


def test_trial_summary_and_verdict():
    frame = _trial(120, damage_at=110)
    onset = damage_start_hours(frame)
    rows = [
        summarize_trial(score_bearing(frame, family, policy, RECORD_MINUTES), onset)
        for family in ("healthy_reference", "isolation_forest")
        for policy in ("first_24h", "first_72h")
    ]
    row = rows[1]
    # The 3-of-5 alarm reacts before the smoothed RMS rule does.
    assert row["alarmed_at_end"] and -1 < row["final_alarm_hours_after_damage_start"] < 1
    assert row["monitored_hours_before_damage"] == pytest.approx(onset - 72)
    result = verdict(pd.DataFrame(rows))
    assert result["eligible_trials"] == ["E9"]
    assert result["trials_with_damage_inside_72h_baseline"] == []
    assert set(result["checks"]) and result["confirmed"] == all(result["checks"].values())


def test_records_must_be_contiguous(tmp_path):
    trial = tmp_path / "E2"
    trial.mkdir()
    for index in (1, 3):
        savemat(trial / f"E2_{index}.mat", {"y": np.zeros((25_600, 1)) + index, "Fs": 25_600})
    with pytest.raises(ValueError, match="contiguous"):
        record_features(trial)
