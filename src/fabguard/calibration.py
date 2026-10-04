"""Per-machine threshold calibration from that machine's own healthy windows.

The threshold can only move up from the fleet threshold, so calibration trades recall for fewer
false alarms and never makes the detector more sensitive. This is not validated on UORED: each
bearing has only 19 overlapping healthy windows, far below ``MIN_BASELINE_WINDOWS``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

MIN_BASELINE_WINDOWS = 120  # about one minute of one-second windows at a half-second hop
BASELINE_QUANTILE = 0.99


class InsufficientBaseline(ValueError):
    """Raised instead of guessing a threshold from too little healthy data."""


def calibrated_threshold(
    baseline_scores: Sequence[float],
    fleet_threshold: float,
    *,
    min_windows: int = MIN_BASELINE_WINDOWS,
    quantile: float = BASELINE_QUANTILE,
) -> float:
    scores = np.asarray(baseline_scores, dtype=float)
    if scores.size < min_windows:
        raise InsufficientBaseline(f"need {min_windows} healthy windows, got {scores.size}")
    if not np.all(np.isfinite(scores)):
        raise InsufficientBaseline("baseline scores must be finite")
    return float(max(fleet_threshold, np.quantile(scores, quantile)))
