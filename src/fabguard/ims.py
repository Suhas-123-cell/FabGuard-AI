"""Run-to-failure check on the NASA/IMS bearing data, vibration only, one model per bearing.

This simulates commissioning a new machine: each bearing gets its own healthy-only detector fitted
on its first recorded hours, then every later snapshot is scored causally with the same 3-of-5
alert rule. Nothing is tuned on outcomes; the model settings below are fixed.

``BASELINE_POLICIES`` compares commissioning rules: the first 24 h, the first 72 h, and re-fitting
on the first 24 h after any stop of at least ``STOP_HOURS``. The last two were added after the
first run; re-fitting after a stop risks learning damage that is already present as normal.

Each IMS snapshot is 20,480 samples at 20 kHz taken every 10 minutes (5 minutes for the first 43
files of test 1), so one snapshot gives one one-second window and the alert rule spans
consecutive snapshots, not consecutive seconds. The 10-21 kHz band is above Nyquist and dropped.

There is no labelled degradation onset. Bearings that did not fail give a false-alarm rate per
day; for bearings that failed, the report gives when alarms started relative to the end of the
test, and earlier alarms on them are reported separately because they cannot be called false.
Failures follow the dataset README (Qiu et al. 2006).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .alerting import DEFAULT_ALERT_RULE, AlertRule
from .signals import FEATURE_COLUMNS, extract_features
from .train import Candidate, _build_estimator, anomaly_scores

URL = "https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip"
SAMPLE_RATE_HZ = 20_000
SNAPSHOT_MINUTES = 10  # nominal spacing; recorded time is counted in snapshots, not calendar time
STOP_HOURS = 12.0
MIN_BASELINE_HOURS = 12.0  # a shorter run after a stop keeps the previous model
BASELINE_POLICIES = {  # name -> (baseline hours, re-fit after a stop)
    "first_24h": (24.0, False),
    "first_72h": (72.0, False),
    "restart_24h": (24.0, True),
}
THRESHOLD_QUANTILE = 0.99
SEED = 17
SNAPSHOT = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$")
COLUMNS = [f"vibration__{name}" for name in FEATURE_COLUMNS if name != "band_power_10000_21000_hz"]
MODELS = {
    "healthy_reference": Candidate("healthy_reference", {}, THRESHOLD_QUANTILE),
    "isolation_forest": Candidate(
        "isolation_forest", {"n_estimators": 100, "contamination": 0.01}, THRESHOLD_QUANTILE
    ),
}


@dataclass(frozen=True)
class Test:
    name: str
    first_snapshot: str
    channels: int
    bearing_channel: dict[int, int]  # bearing -> column used (test 1 has two per bearing)
    failed: dict[int, str]


TESTS = (
    Test(
        "test1",
        "2003.10.22.12.06.24",
        8,
        {1: 0, 2: 2, 3: 4, 4: 6},
        {3: "inner_race", 4: "roller"},
    ),
    Test("test2", "2004.02.12.10.32.39", 4, {1: 0, 2: 1, 3: 2, 4: 3}, {1: "outer_race"}),
    Test("test3", "2004.03.04.09.27.46", 4, {1: 0, 2: 1, 3: 2, 4: 3}, {3: "outer_race"}),
)


def snapshot_time(name: str) -> datetime:
    if not SNAPSHOT.match(name):
        raise ValueError(f"{name!r} is not an IMS snapshot name")
    return datetime.strptime(name, "%Y.%m.%d.%H.%M.%S")


def download(destination: Path) -> Path:
    target = destination / "bearings.zip"
    if target.exists():
        return target
    destination.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    request = urllib.request.Request(URL, headers={"User-Agent": "FabGuard-AI/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as sink:
        shutil.copyfileobj(response, sink, 1 << 20)
    partial.replace(target)
    return target


def _unpack(archive: Path, destination: Path) -> None:
    subprocess.run(["bsdtar", "-xf", str(archive), "-C", str(destination)], check=True)


def unpack_tests(archive: Path, destination: Path) -> dict[str, Path]:
    """Unpack nested archives until the three snapshot directories appear."""
    _unpack(archive, destination)
    seen: set[Path] = set()
    while nested := [
        path
        for path in destination.rglob("*")
        if path.suffix.lower() in {".7z", ".rar", ".zip"} and path not in seen
    ]:
        for path in nested:
            seen.add(path)
            _unpack(path, path.parent)
    found: dict[str, Path] = {}
    for test in TESTS:
        matches = [path.parent for path in destination.rglob(test.first_snapshot) if path.is_file()]
        if len(matches) != 1:
            raise ValueError(f"expected one directory for {test.name}, found {len(matches)}")
        found[test.name] = matches[0]
    return found


def snapshot_features(directory: Path, test: Test) -> pd.DataFrame:
    files = sorted(path for path in directory.iterdir() if SNAPSHOT.match(path.name))
    rows = []
    for path in files:
        data = pd.read_csv(path, sep="\t", header=None, dtype=float).to_numpy()
        if data.shape[1] != test.channels:
            raise ValueError(f"{path.name} has {data.shape[1]} channels, expected {test.channels}")
        timestamp = snapshot_time(path.name)
        for bearing, channel in test.bearing_channel.items():
            features = extract_features(data[:, channel], sample_rate_hz=SAMPLE_RATE_HZ)
            row: dict[str, Any] = {
                "test": test.name,
                "bearing": bearing,
                "snapshot": path.name,
                "timestamp": timestamp,
            }
            row.update({column: float(features[column.split("__", 1)[1]]) for column in COLUMNS})
            rows.append(row)
    return pd.DataFrame(rows)


def build_feature_table(root: Path) -> pd.DataFrame:
    cache = root / "features"
    cache.mkdir(parents=True, exist_ok=True)
    if not all((cache / f"{test.name}.parquet").exists() for test in TESTS):
        archive = download(root)
        with tempfile.TemporaryDirectory(dir=root) as scratch:
            directories = unpack_tests(archive, Path(scratch))
            for test in TESTS:
                table = snapshot_features(directories[test.name], test)
                table.to_parquet(cache / f"{test.name}.parquet", index=False)
                print(f"{test.name}: {table.snapshot.nunique()} snapshots")
    return pd.concat(
        [pd.read_parquet(cache / f"{test.name}.parquet") for test in TESTS], ignore_index=True
    )


def causal_alarm(abnormal: np.ndarray, rule: AlertRule = DEFAULT_ALERT_RULE) -> np.ndarray:
    """Alarm at snapshot i when k of the last n snapshots (up to i) were abnormal."""
    counts = np.convolve(abnormal.astype(int), np.ones(rule.window_count, dtype=int))
    return counts[: abnormal.size] >= rule.min_abnormal


def episodes(alarm: np.ndarray) -> np.ndarray:
    """Indices where the alarm switches on."""
    previous = np.concatenate([[False], alarm[:-1]])
    return np.flatnonzero(alarm & ~previous)


def _snapshots(hours: float) -> int:
    return round(hours * 60 / SNAPSHOT_MINUTES)


def score_bearing(frame: pd.DataFrame, family: str, policy: str = "first_24h") -> pd.DataFrame:
    """Fit on this bearing's own baseline snapshots, then score every later snapshot causally."""
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    hours = ((ordered.timestamp - ordered.timestamp.iloc[0]).dt.total_seconds() / 3600).to_numpy()
    baseline_hours, refit_after_stop = BASELINE_POLICIES[policy]
    starts = [0]
    if refit_after_stop:
        starts += [int(i) for i in np.flatnonzero(np.diff(hours) >= STOP_HOURS) + 1]
    x = ordered[COLUMNS].to_numpy()
    candidate = MODELS[family]
    model, threshold, fits, first_baseline = None, np.nan, 0, 0
    parts = []
    for begin, end in zip(starts, [*starts[1:], len(hours)], strict=True):
        segment = np.arange(begin, end)
        baseline = segment[: _snapshots(baseline_hours)]
        if model is None or baseline.size >= _snapshots(MIN_BASELINE_HOURS):
            model = _build_estimator(candidate, SEED).fit(x[baseline])
            fitted = anomaly_scores(model, family, x[baseline])
            threshold = float(np.quantile(fitted, candidate.threshold_quantile))
            first_baseline = first_baseline or int(baseline.size)
            fits += 1
            segment = segment[baseline.size :]
        if not segment.size:
            continue
        scores = anomaly_scores(model, family, x[segment])
        part = ordered.loc[segment].copy()
        part["hours"] = hours[segment]
        part["anomaly_score"] = scores
        part["threshold"] = threshold
        part["alarm"] = causal_alarm(scores >= threshold)  # the k-of-n count restarts after a stop
        parts.append(part)
    monitored = pd.concat(parts)
    monitored["model_family"] = family
    monitored["baseline_policy"] = policy
    monitored["baseline_snapshots"] = first_baseline
    monitored["baseline_fits"] = fits
    return monitored


def summarize_bearing(scored: pd.DataFrame, test: Test, bearing: int) -> dict[str, Any]:
    alarm = scored.alarm.to_numpy()
    hours = scored.hours.to_numpy()
    end = float(hours[-1])
    starts = episodes(alarm)
    failed = bearing in test.failed
    row: dict[str, Any] = {
        "test": test.name,
        "bearing": bearing,
        "model_family": scored.model_family.iloc[0],
        "baseline_policy": scored.baseline_policy.iloc[0],
        "baseline_fits": int(scored.baseline_fits.iloc[0]),
        "failed": failed,
        "failure_mode": test.failed.get(bearing, "none"),
        "baseline_snapshots": int(scored.baseline_snapshots.iloc[0]),
        "monitored_snapshots": len(scored),
        "monitored_days": len(scored) * SNAPSHOT_MINUTES / 1440,
        "alarm_episodes": len(starts),
        "alarmed_fraction": float(alarm.mean()),
        "alarmed_at_end": bool(alarm[-1]),
        "false_alarms_per_day": np.nan,
        "first_alarm_hours_before_end": np.nan,
        "final_alarm_hours_before_end": np.nan,
        "earlier_episodes": np.nan,
    }
    if not failed:
        row["false_alarms_per_day"] = len(starts) / row["monitored_days"]
    elif len(starts):
        # The episode still running at the end counts as detection; earlier ones are unattributable.
        row["first_alarm_hours_before_end"] = end - float(hours[starts[0]])
        if alarm[-1]:
            row["final_alarm_hours_before_end"] = end - float(hours[starts[-1]])
        row["earlier_episodes"] = len(starts) - int(alarm[-1])
    return row


def add_pre_failure(per_bearing: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    """Healthy-bearing alarms before a failing bearing on the same shaft starts its final alarm.

    All four bearings share one shaft, so once one fails the others see its vibration; alarms
    after that point are not cleanly false. This cut is a reporting choice made after the first
    run, not a model change.
    """
    frame = per_bearing.copy()
    for column in ("pre_failure_days", "pre_failure_episodes", "pre_failure_alarmed_fraction"):
        frame[column] = np.nan
    keys = ["test", "model_family", "baseline_policy"]
    for (test, family, policy), group in frame.groupby(keys):
        failed = group[group.failed & group.final_alarm_hours_before_end.notna()]
        same = scored[
            (scored.test == test)
            & (scored.model_family == family)
            & (scored.baseline_policy == policy)
        ]
        end = float(same.hours.max())
        onset = float((end - failed.final_alarm_hours_before_end).min()) if len(failed) else end
        for index, row in group[~group.failed].iterrows():
            before = same[(same.bearing == row.bearing) & (same.hours < onset)]
            if len(before) < 2:
                continue
            frame.loc[index, "pre_failure_days"] = len(before) * SNAPSHOT_MINUTES / 1440
            frame.loc[index, "pre_failure_episodes"] = len(episodes(before.alarm.to_numpy()))
            frame.loc[index, "pre_failure_alarmed_fraction"] = float(before.alarm.mean())
    return frame


def _plot(scored: pd.DataFrame, path: Path) -> None:
    # Notebooks export an inline MPLBACKEND that this environment may not have.
    os.environ["MPLBACKEND"] = "Agg"
    from matplotlib import pyplot as plt

    failed = [(test, bearing) for test in TESTS for bearing in test.failed]
    figure, axes = plt.subplots(len(failed), 1, figsize=(10, 2.4 * len(failed)))
    for axis, (test, bearing) in zip(axes, failed, strict=True):
        part = scored[
            (scored.test == test.name)
            & (scored.bearing == bearing)
            & (scored.model_family == "isolation_forest")
            & (scored.baseline_policy == "first_24h")
        ]
        axis.plot(part.hours, part.anomaly_score, linewidth=0.6)
        axis.axhline(part.threshold.iloc[0], color="black", linestyle="--", linewidth=0.8)
        alarmed = part[part.alarm]
        axis.scatter(alarmed.hours, alarmed.anomaly_score, s=4, color="tab:red")
        axis.set(title=f"{test.name} bearing {bearing} ({test.failed[bearing]})", ylabel="score")
    axes[-1].set_xlabel("hours since start of test")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def run(root: Path, output: Path) -> dict[str, Any]:
    table = build_feature_table(root)
    output.mkdir(parents=True, exist_ok=True)
    scored_parts, rows = [], []
    for test in TESTS:
        for bearing in test.bearing_channel:
            frame = table[(table.test == test.name) & (table.bearing == bearing)]
            for family in MODELS:
                for policy in BASELINE_POLICIES:
                    scored = score_bearing(frame, family, policy)
                    scored_parts.append(scored)
                    rows.append(summarize_bearing(scored, test, bearing))
    scored = pd.concat(scored_parts, ignore_index=True)
    per_bearing = add_pre_failure(pd.DataFrame(rows), scored)
    per_bearing.to_csv(output / "per_bearing.csv", index=False)
    keep = ["test", "bearing", "model_family", "baseline_policy", "snapshot", "hours"]
    scored[[*keep, "anomaly_score", "threshold", "alarm"]].to_csv(
        output / "scores.csv.gz", index=False
    )
    _plot(scored, output / "failed_bearings.png")
    summary = []
    for (policy, family), group in per_bearing.groupby(["baseline_policy", "model_family"]):
        healthy, failed = group[~group.failed], group[group.failed]
        summary.append(
            {
                "baseline_policy": policy,
                "model_family": family,
                "healthy_bearings": len(healthy),
                "healthy_monitored_days": float(healthy.monitored_days.sum()),
                "healthy_alarm_episodes": int(healthy.alarm_episodes.sum()),
                "false_alarms_per_day": float(
                    healthy.alarm_episodes.sum() / healthy.monitored_days.sum()
                ),
                "healthy_alarmed_fraction": float(healthy.alarmed_fraction.mean()),
                "pre_failure_false_alarms_per_day": float(
                    healthy.pre_failure_episodes.sum() / healthy.pre_failure_days.sum()
                ),
                "pre_failure_alarmed_fraction": float(healthy.pre_failure_alarmed_fraction.mean()),
                "failed_bearings": len(failed),
                "failed_alarmed_at_end": int(failed.alarmed_at_end.sum()),
                "median_final_alarm_hours_before_end": float(
                    failed.final_alarm_hours_before_end.median()
                ),
            }
        )
    result = {
        "dataset": "NASA/IMS bearing run-to-failure data, one vibration channel per bearing",
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "baseline_policies": BASELINE_POLICIES,
        "stop_hours": STOP_HOURS,
        "threshold_quantile": THRESHOLD_QUANTILE,
        "alert_rule": DEFAULT_ALERT_RULE.to_dict(),
        "features": COLUMNS,
        "summary": summary,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/ims"))
    parser.add_argument("--output", type=Path, default=Path("reports/ims-evaluation"))
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.output)["summary"], indent=2))


if __name__ == "__main__":
    main()
