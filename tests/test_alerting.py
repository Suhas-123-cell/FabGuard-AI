import pandas as pd
import pytest

from fabguard.alerting import AlertRule, persistent_score, rule_sensitivity


def test_single_spike_does_not_alarm_under_three_of_five():
    score, index = persistent_score([0.1, 0.9, 0.1, 0.1, 0.1, 0.1], AlertRule(3, 5))
    assert score == pytest.approx(0.1)
    assert score < 0.5 and index != 1


def test_three_in_a_span_alarms_and_picks_the_kth_window():
    scores = [0.1, 0.8, 0.7, 0.6, 0.1, 0.1]
    score, index = persistent_score(scores, AlertRule(3, 5))
    assert (score, index) == (0.6, 3)


def test_one_of_one_matches_the_maximum():
    assert persistent_score([0.2, 0.9, 0.4], AlertRule(1, 1)) == (0.9, 1)


def test_short_recordings_shrink_the_span_and_bad_rules_are_rejected():
    assert persistent_score([0.3, 0.2], AlertRule(3, 5))[0] == 0.2
    with pytest.raises(ValueError):
        AlertRule(4, 3)
    with pytest.raises(ValueError):
        persistent_score([], AlertRule())


def test_rule_sensitivity_counts_alarmed_recordings():
    rows = []
    for recording, state, scores in (
        ("h", "healthy", [0.1, 0.9, 0.1, 0.1]),
        ("f", "faulty", [0.9, 0.9, 0.9, 0.1]),
    ):
        for index, score in enumerate(scores):
            rows.append((recording, state, index, score, 0.5))
    frame = pd.DataFrame(
        rows, columns=["recording_id", "health_state", "window_index", "anomaly_score", "threshold"]
    )
    result = rule_sensitivity(frame, [AlertRule(1, 1), AlertRule(3, 5)]).set_index("rule")
    assert result.loc["1-of-1", "healthy_alarmed"] == 1.0
    assert result.loc["3-of-5", "healthy_alarmed"] == 0.0
    assert result.loc["3-of-5", "faulty_alarmed"] == 1.0
