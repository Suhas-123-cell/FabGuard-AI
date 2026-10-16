"""Check the IMS 72-hour commissioning rule on the University of Ferrara run-to-failure trials.

The protocol in ``reports/ferrara-evaluation/PROTOCOL.md`` was written before the data was opened;
this module implements it and nothing else. Each trial is one test bearing on one channel (the
support-bearing sensors are not published), so false alarms are counted on the test bearing before
its damage starts, defined from RMS alone so that it does not depend on the detector.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .ims import COLUMNS, MODELS, _unpack, episodes, score_bearing
from .signals import extract_features

RECORD_MINUTES = 5
POLICIES = ("first_24h", "first_72h")
ONSET_LEVEL_HOURS = 24.0
ONSET_FACTOR = 2.0
ONSET_SMOOTHING = 12  # records, one hour
ELIGIBLE_MONITORED_HOURS = 24.0
MAX_FALSE_ALARMS_PER_DAY = 0.5
RECORD = re.compile(r"^(E\d)_(\d+)\.mat$")
START = datetime(2000, 1, 1)  # the files carry no clock time; only the cadence is known


def record_features(directory: Path) -> pd.DataFrame:
    files = {}
    for path in directory.glob("*.mat"):
        if match := RECORD.match(path.name):
            files[int(match.group(2))] = path
    trial = directory.name
    if trial == "E4":
        raise NotImplementedError("E4's first 453 records are hourly; mixed cadence is not handled")
    if sorted(files) != list(range(1, len(files) + 1)):
        raise ValueError(f"{trial}: record numbers are not contiguous from 1")
    rows = []
    for index in sorted(files):
        record = loadmat(files[index])
        signal = np.asarray(record["y"], dtype=float).ravel()
        features = extract_features(signal, sample_rate_hz=float(np.ravel(record["Fs"])[0]))
        row: dict[str, Any] = {
            "trial": trial,
            "record": index,
            "timestamp": START + timedelta(minutes=RECORD_MINUTES * (index - 1)),
        }
        row.update({column: float(features[column.split("__", 1)[1]]) for column in COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows)


def damage_start_hours(frame: pd.DataFrame) -> float:
    """When the trailing one-hour median RMS last rises to twice its first-day level, for good."""
    hours = (frame.timestamp - frame.timestamp.iloc[0]).dt.total_seconds().to_numpy() / 3600
    rms = frame["vibration__rms"].to_numpy()
    level = float(np.median(rms[hours < ONSET_LEVEL_HOURS]))
    smoothed = pd.Series(rms).rolling(ONSET_SMOOTHING, min_periods=1).median().to_numpy()
    below = np.flatnonzero(smoothed < ONSET_FACTOR * level)
    start = 0 if not below.size else int(below[-1]) + 1
    return float(hours[start]) if start < len(hours) else float("nan")


def summarize_trial(scored: pd.DataFrame, onset: float) -> dict[str, Any]:
    hours = scored.hours.to_numpy()
    alarm = scored.alarm.to_numpy()
    starts = episodes(alarm)
    end = float(hours[-1])
    before = hours < onset
    final = float(hours[starts[-1]]) if alarm[-1] else np.nan
    return {
        "trial": scored.trial.iloc[0],
        "model_family": scored.model_family.iloc[0],
        "baseline_policy": scored.baseline_policy.iloc[0],
        "end_hours": end,
        "damage_start_hours": onset,
        "monitored_hours_before_damage": float(before.sum() * RECORD_MINUTES / 60),
        "false_alarms": int(before[starts].sum()),
        "false_alarm_days": float(before.sum() * RECORD_MINUTES / 1440),
        "time_in_alarm_before_damage": float(alarm[before].mean()) if before.any() else np.nan,
        "alarmed_at_end": bool(alarm[-1]),
        "final_alarm_hours_before_end": end - final,
        "final_alarm_hours_after_damage_start": final - onset,
    }


def verdict(per_trial: pd.DataFrame) -> dict[str, Any]:
    reference = per_trial[per_trial.model_family == "healthy_reference"]
    long = reference[(reference.baseline_policy == "first_72h") & (reference.end_hours > 96)]
    damaged_baseline = sorted(long[long.damage_start_hours < 72].trial)
    eligible = sorted(
        long[long.monitored_hours_before_damage >= ELIGIBLE_MONITORED_HOURS].trial.unique()
    )
    rates = {}
    for policy in POLICIES:
        rows = reference[(reference.baseline_policy == policy) & reference.trial.isin(eligible)]
        days = rows.false_alarm_days.sum()
        rates[policy] = float(rows.false_alarms.sum() / days) if days else float("nan")
    final = long[long.trial.isin(eligible)]
    checks = {
        "pooled_false_alarms_at_most_0.5_per_day": rates["first_72h"] <= MAX_FALSE_ALARMS_PER_DAY,
        "every_eligible_bearing_in_alarm_at_end": bool(final.alarmed_at_end.all()),
        "fewer_false_alarms_than_first_24h": rates["first_72h"] < rates["first_24h"],
        "no_damage_inside_72h_baseline": not damaged_baseline,
    }
    return {
        "eligible_trials": eligible,
        "trials_with_damage_inside_72h_baseline": damaged_baseline,
        "false_alarms_per_day": rates,
        "checks": checks,
        "confirmed": bool(eligible) and all(checks.values()),
    }


def run(root: Path, output: Path) -> dict[str, Any]:
    cache = root / "features"
    cache.mkdir(parents=True, exist_ok=True)
    for archive in sorted((root / "archives").glob("E*.7z")):
        trial = archive.stem
        if not (cache / f"{trial}.parquet").exists():
            if not (root / trial).is_dir():
                _unpack(archive, root)
            table = record_features(root / trial)
            table.to_parquet(cache / f"{trial}.parquet", index=False)
            print(f"{trial}: {len(table)} records")
    table = pd.concat([pd.read_parquet(path) for path in sorted(cache.glob("E*.parquet"))])
    rows, scored_parts = [], []
    for _, frame in table.groupby("trial"):
        frame = frame.sort_values("record").reset_index(drop=True)
        onset = damage_start_hours(frame)
        end = (frame.timestamp.iloc[-1] - frame.timestamp.iloc[0]).total_seconds() / 3600
        for family in MODELS:
            for policy in POLICIES:
                if policy == "first_72h" and end <= 72:
                    continue
                scored = score_bearing(frame, family, policy, snapshot_minutes=RECORD_MINUTES)
                scored_parts.append(scored)
                rows.append(summarize_trial(scored, onset))
    per_trial = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    per_trial.to_csv(output / "per_trial.csv", index=False)
    scored = pd.concat(scored_parts, ignore_index=True)
    keep = ["trial", "record", "model_family", "baseline_policy", "hours", "vibration__rms"]
    scored[[*keep, "anomaly_score", "threshold", "alarm"]].to_csv(
        output / "scores.csv.gz", index=False
    )
    result = {
        "dataset": "University of Ferrara run-to-failure trials, Mendeley Data htk59pp5wx",
        "protocol": "reports/ferrara-evaluation/PROTOCOL.md",
        "trials": sorted(table.trial.unique()),
        "record_minutes": RECORD_MINUTES,
        "features": COLUMNS,
        "verdict": verdict(per_trial),
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("data/ferrara"))
    parser.add_argument("--output", type=Path, default=Path("reports/ferrara-evaluation"))
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.output)["verdict"], indent=2))


if __name__ == "__main__":
    main()
