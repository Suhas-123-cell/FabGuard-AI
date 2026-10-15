import zipfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from fabguard.alerting import AlertRule
from fabguard.ims import (
    COLUMNS,
    TESTS,
    causal_alarm,
    episodes,
    score_bearing,
    snapshot_features,
    snapshot_time,
    summarize_bearing,
    unpack_tests,
)


def test_snapshot_names_parse_and_reject_paths():
    assert snapshot_time("2004.02.12.10.32.39") == datetime(2004, 2, 12, 10, 32, 39)
    with pytest.raises(ValueError):
        snapshot_time("../2004.02.12.10.32.39")


def test_alarm_is_causal_k_of_n_and_counts_episodes():
    abnormal = np.array([1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    alarm = causal_alarm(abnormal, AlertRule(3, 5))
    # Snapshot 3 is the first with three abnormal among the last five; nothing looks ahead.
    assert alarm.tolist() == [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1]
    assert episodes(alarm).tolist() == [3, 10]


def _bearing(hours: int, *, shift_after: float | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    start = datetime(2004, 2, 12)
    rows = []
    for index in range(hours * 6):
        values = rng.normal(1.0, 0.05, len(COLUMNS))
        if shift_after is not None and index / 6 >= shift_after:
            values *= 4.0
        row = {"timestamp": start + timedelta(minutes=10 * index)}
        row.update(dict(zip(COLUMNS, values, strict=True)))
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("family", ["healthy_reference", "isolation_forest"])
def test_failing_bearing_alarms_at_end_and_lead_time_is_measured(family):
    test = TESTS[1]  # bearing 1 failed
    scored = score_bearing(_bearing(72, shift_after=60), family)
    assert scored.hours.min() >= 24 and scored.baseline_snapshots.iloc[0] == 144
    row = summarize_bearing(scored, test, 1)
    assert row["failed"] and row["alarmed_at_end"]
    assert 11 <= row["final_alarm_hours_before_end"] <= 12
    assert np.isnan(row["false_alarms_per_day"])


def test_healthy_bearing_reports_false_alarms_per_day():
    scored = score_bearing(_bearing(72), "healthy_reference")
    row = summarize_bearing(scored, TESTS[1], 2)
    assert not row["failed"]
    assert row["monitored_days"] == pytest.approx(48 / 24 - 1 / 144, rel=1e-3)
    assert row["false_alarms_per_day"] == row["alarm_episodes"] / row["monitored_days"]
    assert np.isnan(row["final_alarm_hours_before_end"])


def test_snapshot_features_check_channel_count(tmp_path):
    rng = np.random.default_rng(0)
    test = TESTS[1]
    np.savetxt(tmp_path / test.first_snapshot, rng.normal(size=(20_480, 4)), delimiter="\t")
    (tmp_path / "README.txt").write_text("not a snapshot")
    table = snapshot_features(tmp_path, test)
    assert len(table) == 4 and set(table.bearing) == {1, 2, 3, 4}
    assert all(np.isfinite(table[column]).all() for column in COLUMNS)
    np.savetxt(tmp_path / test.first_snapshot, rng.normal(size=(20_480, 8)), delimiter="\t")
    with pytest.raises(ValueError, match="channels"):
        snapshot_features(tmp_path, test)


def test_unpack_finds_each_test_inside_nested_archives(tmp_path):
    inner = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner, "w") as archive:
        for index, test in enumerate(TESTS):
            archive.writestr(f"{index}_test/txt/{test.first_snapshot}", "0\t0\n")
    outer = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer, "w") as archive:
        archive.write(inner, "4. Bearings/IMS.zip")
    target = tmp_path / "out"
    target.mkdir()
    found = unpack_tests(outer, target)
    assert {name: path.name for name, path in found.items()} == {
        "test1": "txt",
        "test2": "txt",
        "test3": "txt",
    }


def test_pre_failure_cut_stops_at_failing_bearings_final_alarm():
    from fabguard.ims import add_pre_failure

    hours = np.arange(10.0)
    scored = pd.DataFrame(
        {
            "test": "test2",
            "model_family": "healthy_reference",
            "bearing": [1] * 10 + [2] * 10,
            "hours": np.concatenate([hours, hours]),
            "alarm": [0] * 6 + [1] * 4 + [0, 1, 0, 0, 0, 0, 0, 1, 1, 1],
        }
    ).astype({"alarm": bool})
    per_bearing = pd.DataFrame(
        {
            "test": "test2",
            "model_family": "healthy_reference",
            "bearing": [1, 2],
            "failed": [True, False],
            "final_alarm_hours_before_end": [3.0, np.nan],
        }
    )
    row = add_pre_failure(per_bearing, scored).iloc[1]
    # Bearing 1's final alarm starts at hour 6, so bearing 2's alarm from hour 7 is excluded.
    assert row.pre_failure_episodes == 1 and row.pre_failure_alarmed_fraction == 1 / 6
    assert row.pre_failure_days == 5 / 24
