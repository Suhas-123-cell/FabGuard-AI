import numpy as np
import pandas as pd

from fabguard.signals import FEATURE_COLUMNS
from fabguard.train import Candidate, HealthyReferenceDetector, anomaly_scores, evaluate_lobo


def test_healthy_reference_scores_far_samples_higher():
    detector = HealthyReferenceDetector().fit(np.array([[0.0, 0.0], [0.1, -0.1], [-0.1, 0.1]]))
    scores = detector.score_samples(np.array([[0.0, 0.0], [4.0, 4.0]]))
    assert scores[1] > scores[0]


def test_lobo_keeps_bearing_out_and_returns_all_predictions():
    rows = []
    rng = np.random.default_rng(3)
    for bearing in range(1, 6):
        for state, label, shift in (
            ("healthy", 0, 0.0),
            ("developing", 1, 2.0),
            ("faulty", 1, 4.0),
        ):
            for window in range(3):
                row = {
                    "recording_id": f"r-{bearing}-{state}",
                    "bearing_id": bearing,
                    "health_state": state,
                    "fault_family": "test",
                    "manufacturer": "test",
                    "cohort": "primary",
                    "binary_label": label,
                    "window_index": window,
                    "start_sample": window * 10,
                    "end_sample": window * 10 + 10,
                    "load_mean": 400.0,
                    "rpm_mean": 1750.0,
                }
                for name in FEATURE_COLUMNS:
                    row[f"vibration__{name}"] = shift + rng.normal(0, 0.05)
                    row[f"audio__{name}"] = shift + rng.normal(0, 0.05)
                rows.append(row)
    table = pd.DataFrame(rows)
    predictions, folds = evaluate_lobo(
        table, policies=("vibration",), candidates=(Candidate("healthy_reference", {}, 0.95),)
    )
    assert len(folds) == 5
    assert len(predictions) == len(table)
    assert (
        predictions.loc[predictions.binary_label == 1, "anomaly_score"].mean()
        > predictions.loc[predictions.binary_label == 0, "anomaly_score"].mean()
    )


def test_anomaly_scores_invert_isolation_forest_decision():
    class Stub:
        def decision_function(self, values):
            return np.array([0.5, -0.5])

    scores = anomaly_scores(Stub(), "isolation_forest", np.zeros((2, 1)))
    assert scores.tolist() == [-0.5, 0.5]
