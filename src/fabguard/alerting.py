"""Recording-level alert rule: k abnormal windows within any n consecutive windows.

The score is the k-th largest window score inside the best span, so
``score >= threshold`` is exactly the k-of-n rule and decision/score stay consistent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlertRule:
    min_abnormal: int = 3
    window_count: int = 5

    def __post_init__(self) -> None:
        if not 1 <= self.min_abnormal <= self.window_count:
            raise ValueError("require 1 <= min_abnormal <= window_count")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


DEFAULT_ALERT_RULE = AlertRule()


def persistent_score(scores: Sequence[float], rule: AlertRule) -> tuple[float, int]:
    """Return (score, window index) for the strongest k-of-n span."""
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        raise ValueError("no window scores")
    span = min(rule.window_count, values.size)
    k = min(rule.min_abnormal, span)
    best_score, best_index = -np.inf, 0
    for start in range(values.size - span + 1):
        order = np.argsort(values[start : start + span])[::-1]
        index = start + int(order[k - 1])
        if values[index] > best_score:
            best_score, best_index = float(values[index]), index
    return best_score, best_index


def rule_sensitivity(predictions: pd.DataFrame, rules: Sequence[AlertRule]) -> pd.DataFrame:
    """Share of recordings alarmed per health state, from out-of-fold window predictions."""
    ordered = predictions.sort_values(["recording_id", "window_index"])
    states = ordered.groupby("recording_id").health_state.first()
    rows = []
    for rule in rules:
        alarms = {
            key: persistent_score(group.anomaly_score, rule)[0] >= group.threshold.iloc[0]
            for key, group in ordered.groupby("recording_id")
        }
        row: dict[str, object] = {"rule": f"{rule.min_abnormal}-of-{rule.window_count}"}
        for state in ("healthy", "developing", "faulty"):
            ids = states.index[states == state]
            hits = [alarms[i] for i in ids]
            row[f"{state}_alarmed"] = float(np.mean(hits)) if hits else np.nan
            row[f"{state}_recordings"] = len(ids)
        rows.append(row)
    return pd.DataFrame(rows)
