import numpy as np
import pytest

from fabguard.signals import (
    FEATURE_COLUMNS,
    QualityThresholds,
    assess_pair_alignment,
    assess_quality,
    extract_features,
    extract_recording_features,
    ordered_feature_vector,
    window_signal,
)


def test_ten_second_recording_produces_nineteen_declared_windows():
    data = np.arange(420_000, dtype=float)

    windows = window_signal(data)

    assert windows.shape == (19, 42_000)
    assert windows[1, 0] == 21_000
    assert np.shares_memory(data, windows)


def test_short_recording_has_no_complete_window():
    windows = window_signal(np.zeros(99), sample_rate_hz=100, window_seconds=1)
    assert windows.shape == (0, 100)


def test_features_remove_mean_but_preserve_amplitude_and_frequency_band():
    sample_rate = 42_000
    time = np.arange(sample_rate) / sample_rate
    unit = 7.5 + np.sin(2 * np.pi * 1_000 * time)
    doubled = 7.5 + 2 * np.sin(2 * np.pi * 1_000 * time)

    unit_features = extract_features(unit, sample_rate_hz=sample_rate)
    doubled_features = extract_features(doubled, sample_rate_hz=sample_rate)

    assert tuple(unit_features) == FEATURE_COLUMNS
    assert unit_features["rms"] == pytest.approx(1 / np.sqrt(2), rel=1e-3)
    assert doubled_features["rms"] == pytest.approx(2 * unit_features["rms"], rel=1e-6)
    assert doubled_features["spectral_power"] == pytest.approx(
        4 * unit_features["spectral_power"], rel=1e-6
    )
    assert unit_features["band_power_500_2000_hz"] > 100 * unit_features["band_power_0_500_hz"]
    assert np.allclose(ordered_feature_vector(unit_features), list(unit_features.values()))


def test_recording_features_include_reproducible_offsets():
    values = np.random.default_rng(7).normal(size=1_000)
    rows = extract_recording_features(
        values,
        sample_rate_hz=100,
        window_seconds=1,
        hop_seconds=0.5,
    )
    assert len(rows) == 19
    assert rows[0]["start_sample"] == 0
    assert rows[1]["start_sample"] == 50
    assert rows[-1]["end_sample"] == 1_000


def test_quality_separates_constant_silence_and_probable_clipping():
    silent = assess_quality(np.zeros(100), sample_rate_hz=10)
    clipped_values = np.concatenate([np.linspace(-1, 0.8, 80), np.ones(20)])
    clipped = assess_quality(
        clipped_values,
        thresholds=QualityThresholds(clipping_rail_fraction=0.1),
    )

    assert silent.is_constant and silent.is_silent and not silent.is_usable
    assert "constant_signal" in silent.issues
    assert clipped.is_clipped and "probable_clipping" in clipped.issues


def test_periodic_extrema_without_plateaus_are_not_called_clipping():
    phase = np.arange(1_000) / 100
    sinusoid = np.sin(2 * np.pi * 4 * phase)
    assert not assess_quality(sinusoid).is_clipped


def test_non_finite_signal_fails_quality_and_feature_extraction():
    values = np.array([0.0, 1.0, np.nan, 2.0])
    quality = assess_quality(values)
    assert not quality.is_usable
    assert "non_finite_samples" in quality.issues
    with pytest.raises(ValueError, match="finite"):
        extract_features(values)


def test_pair_alignment_reports_length_without_treating_correlation_as_gate():
    left = np.arange(100, dtype=float)
    right = np.arange(99, dtype=float)
    result = assess_pair_alignment(left, right, sample_rate_hz=10, estimate_lag=True)
    assert not result.aligned_by_length
    assert result.sample_count_difference == 1
    assert result.duration_difference_seconds == pytest.approx(0.1)
    assert result.correlation == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate_hz": 0},
        {"window_seconds": 0},
        {"hop_seconds": -1},
    ],
)
def test_windowing_rejects_nonpositive_configuration(kwargs):
    with pytest.raises(ValueError, match="positive"):
        window_signal(np.ones(100), **kwargs)
