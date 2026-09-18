"""Signal validation, windowing, and deterministic feature extraction.

The functions in this module deliberately know nothing about fault labels.  They
operate on one-dimensional arrays and keep amplitude information intact; model
training is responsible for fitting any subsequent scaler on training data.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal as scipy_signal
from scipy import stats

DEFAULT_SAMPLE_RATE_HZ = 42_000.0
DEFAULT_WINDOW_SECONDS = 1.0
DEFAULT_HOP_SECONDS = 0.5

# Fixed before model comparison. The final band stops at the data's Nyquist
# frequency. These are descriptive bands, not claimed fault bands.
DEFAULT_FREQUENCY_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (0.0, 500.0),
    (500.0, 2_000.0),
    (2_000.0, 5_000.0),
    (5_000.0, 10_000.0),
    (10_000.0, 21_000.0),
)


def _band_name(low_hz: float, high_hz: float) -> str:
    def compact(value: float) -> str:
        return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")

    return f"band_power_{compact(low_hz)}_{compact(high_hz)}_hz"


FEATURE_COLUMNS: tuple[str, ...] = (
    "rms",
    "crest_factor",
    "kurtosis",
    "peak_to_peak",
    "spectral_power",
    *(_band_name(low, high) for low, high in DEFAULT_FREQUENCY_BANDS_HZ),
)


@dataclass(frozen=True)
class QualityThresholds:
    """Conservative technical-usability thresholds.

    ``silence_rms`` and ``constant_tolerance`` are expressed in source units.
    They flag zero-like data but intentionally do not judge low-amplitude valid
    recordings. Clipping is inferred only when many samples repeat either
    observed rail, because CSV files do not preserve ADC dtype/range metadata.
    """

    minimum_finite_fraction: float = 0.999
    silence_rms: float = 1e-12
    constant_tolerance: float = 1e-12
    clipping_rail_fraction: float = 0.01
    clipping_tolerance_fraction: float = 1e-9
    minimum_samples: int = 2


@dataclass(frozen=True)
class SignalQuality:
    sample_count: int
    finite_count: int
    finite_fraction: float
    duration_seconds: float | None
    mean: float | None
    standard_deviation: float | None
    rms: float | None
    minimum: float | None
    maximum: float | None
    peak_to_peak: float | None
    zero_fraction: float | None
    rail_fraction: float | None
    is_constant: bool
    is_silent: bool
    is_clipped: bool
    is_usable: bool
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PairAlignment:
    left_samples: int
    right_samples: int
    sample_count_difference: int
    duration_difference_seconds: float | None
    aligned_by_length: bool
    correlation: float | None
    lag_samples: int | None
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _one_dimensional_floats(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"expected one-dimensional signal, got shape {array.shape}")
    return array


def assess_quality(
    values: Sequence[float] | np.ndarray,
    *,
    sample_rate_hz: float | None = None,
    thresholds: QualityThresholds | None = None,
) -> SignalQuality:
    """Assess only technical signal usability, not predictive usefulness."""

    data = _one_dimensional_floats(values)
    limits = thresholds or QualityThresholds()
    if sample_rate_hz is not None and sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")

    sample_count = int(data.size)
    finite_mask = np.isfinite(data)
    finite = data[finite_mask]
    finite_count = int(finite.size)
    finite_fraction = finite_count / sample_count if sample_count else 0.0
    duration = sample_count / sample_rate_hz if sample_rate_hz else None
    issues: list[str] = []

    if sample_count < limits.minimum_samples:
        issues.append("too_few_samples")
    if finite_fraction < limits.minimum_finite_fraction:
        issues.append("non_finite_samples")

    if finite_count == 0:
        issues.append("no_finite_samples")
        return SignalQuality(
            sample_count=sample_count,
            finite_count=0,
            finite_fraction=finite_fraction,
            duration_seconds=duration,
            mean=None,
            standard_deviation=None,
            rms=None,
            minimum=None,
            maximum=None,
            peak_to_peak=None,
            zero_fraction=None,
            rail_fraction=None,
            is_constant=True,
            is_silent=True,
            is_clipped=False,
            is_usable=False,
            issues=tuple(issues),
        )

    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    peak_to_peak = maximum - minimum
    mean = float(np.mean(finite))
    standard_deviation = float(np.std(finite))
    rms = float(np.sqrt(np.mean(np.square(finite))))
    zero_fraction = float(np.mean(finite == 0.0))

    scale = max(abs(minimum), abs(maximum), 1.0)
    rail_tolerance = limits.clipping_tolerance_fraction * scale
    at_low_rail = np.isclose(finite, minimum, rtol=0.0, atol=rail_tolerance)
    at_high_rail = np.isclose(finite, maximum, rtol=0.0, atol=rail_tolerance)
    rail_fraction = float(max(np.mean(at_low_rail), np.mean(at_high_rail)))

    def longest_run(mask: np.ndarray) -> int:
        if not np.any(mask):
            return 0
        padded = np.pad(mask.astype(np.int8), 1)
        edges = np.flatnonzero(np.diff(padded))
        return int(np.max(edges[1::2] - edges[::2]))

    is_constant = peak_to_peak <= limits.constant_tolerance
    is_silent = rms <= limits.silence_rms
    # A constant signal repeats its rails by definition; report it as constant,
    # not additionally as clipped.
    has_rail_plateau = max(longest_run(at_low_rail), longest_run(at_high_rail)) >= 3
    is_clipped = (
        not is_constant and rail_fraction >= limits.clipping_rail_fraction and has_rail_plateau
    )
    if is_constant:
        issues.append("constant_signal")
    if is_silent:
        issues.append("silent_signal")
    if is_clipped:
        issues.append("probable_clipping")

    return SignalQuality(
        sample_count=sample_count,
        finite_count=finite_count,
        finite_fraction=finite_fraction,
        duration_seconds=duration,
        mean=mean,
        standard_deviation=standard_deviation,
        rms=rms,
        minimum=minimum,
        maximum=maximum,
        peak_to_peak=peak_to_peak,
        zero_fraction=zero_fraction,
        rail_fraction=rail_fraction,
        is_constant=is_constant,
        is_silent=is_silent,
        is_clipped=is_clipped,
        is_usable=not issues,
        issues=tuple(issues),
    )


def assess_pair_alignment(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
    *,
    sample_rate_hz: float | None = None,
    estimate_lag: bool = False,
    maximum_correlation_samples: int = 200_000,
) -> PairAlignment:
    """Check paired-channel length and, optionally, descriptive correlation.

    Vibration and sound need not be strongly correlated, so correlation never
    decides usability. It is exposed as an audit observation only.
    """

    left_array = _one_dimensional_floats(left)
    right_array = _one_dimensional_floats(right)
    if sample_rate_hz is not None and sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    difference = int(left_array.size - right_array.size)
    issues: list[str] = []
    if difference:
        issues.append("sample_count_mismatch")

    correlation: float | None = None
    lag: int | None = None
    usable = min(left_array.size, right_array.size, maximum_correlation_samples)
    if estimate_lag and usable >= 2:
        x = left_array[:usable]
        y = right_array[:usable]
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size >= 2 and np.std(x) > 0 and np.std(y) > 0:
            x = x - np.mean(x)
            y = y - np.mean(y)
            correlation = float(np.corrcoef(x, y)[0, 1])
            cross = scipy_signal.correlate(x, y, mode="full", method="fft")
            lags = scipy_signal.correlation_lags(x.size, y.size, mode="full")
            lag = int(lags[np.argmax(cross)])

    duration_difference = difference / sample_rate_hz if sample_rate_hz else None
    return PairAlignment(
        left_samples=int(left_array.size),
        right_samples=int(right_array.size),
        sample_count_difference=difference,
        duration_difference_seconds=duration_difference,
        aligned_by_length=difference == 0,
        correlation=correlation,
        lag_samples=lag,
        issues=tuple(issues),
    )


def window_signal(
    values: Sequence[float] | np.ndarray,
    *,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
) -> np.ndarray:
    """Return a window view containing complete windows only."""

    data = _one_dimensional_floats(values)
    if sample_rate_hz <= 0 or window_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("sample rate, window length, and hop must be positive")
    window_samples = int(round(window_seconds * sample_rate_hz))
    hop_samples = int(round(hop_seconds * sample_rate_hz))
    if window_samples < 1 or hop_samples < 1:
        raise ValueError("window and hop must each contain at least one sample")
    if data.size < window_samples:
        return np.empty((0, window_samples), dtype=np.float64)
    return np.lib.stride_tricks.sliding_window_view(data, window_samples)[::hop_samples]


def iter_windows(
    values: Sequence[float] | np.ndarray,
    **kwargs: float,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(start_sample, window)`` without copying window data."""

    sample_rate_hz = float(kwargs.get("sample_rate_hz", DEFAULT_SAMPLE_RATE_HZ))
    hop_seconds = float(kwargs.get("hop_seconds", DEFAULT_HOP_SECONDS))
    hop_samples = int(round(sample_rate_hz * hop_seconds))
    for index, window in enumerate(window_signal(values, **kwargs)):
        yield index * hop_samples, window


def extract_features(
    values: Sequence[float] | np.ndarray,
    *,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    frequency_bands_hz: Iterable[tuple[float, float]] = DEFAULT_FREQUENCY_BANDS_HZ,
) -> dict[str, float]:
    """Extract amplitude-preserving time and Hann-periodogram features.

    The mean is removed once before every feature, as declared in the design.
    No per-recording standardization or amplitude normalization is performed.
    """

    data = _one_dimensional_floats(values)
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if data.size < 2:
        raise ValueError("at least two samples are required")
    if not np.all(np.isfinite(data)):
        raise ValueError("feature extraction requires finite samples")

    centered = data - np.mean(data)
    rms = float(np.sqrt(np.mean(np.square(centered))))
    peak = float(np.max(np.abs(centered)))
    crest_factor = peak / rms if rms > 0 else 0.0
    peak_to_peak = float(np.ptp(centered))
    kurtosis = float(stats.kurtosis(centered, fisher=False, bias=False)) if rms > 0 else 0.0
    if not np.isfinite(kurtosis):
        kurtosis = 0.0

    frequencies, density = scipy_signal.periodogram(
        centered,
        fs=sample_rate_hz,
        window="hann",
        detrend=False,
        scaling="density",
        return_onesided=True,
    )
    df = float(frequencies[1] - frequencies[0]) if frequencies.size > 1 else 0.0
    spectral_power = float(np.sum(density) * df)
    result: dict[str, float] = {
        "rms": rms,
        "crest_factor": crest_factor,
        "kurtosis": kurtosis,
        "peak_to_peak": peak_to_peak,
        "spectral_power": spectral_power,
    }

    bands = tuple(frequency_bands_hz)
    previous_high = -np.inf
    nyquist = sample_rate_hz / 2.0
    for low_hz, high_hz in bands:
        if low_hz < 0 or high_hz <= low_hz:
            raise ValueError(f"invalid frequency band ({low_hz}, {high_hz})")
        if low_hz < previous_high:
            raise ValueError("frequency bands must be ordered and non-overlapping")
        previous_high = high_hz
        # Half-open intervals avoid double counting shared boundaries; include
        # Nyquist in the final reachable band.
        if high_hz >= nyquist:
            mask = (frequencies >= low_hz) & (frequencies <= min(high_hz, nyquist))
        else:
            mask = (frequencies >= low_hz) & (frequencies < high_hz)
        result[_band_name(low_hz, high_hz)] = float(np.sum(density[mask]) * df)
    return result


def extract_recording_features(
    values: Sequence[float] | np.ndarray,
    *,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    frequency_bands_hz: Iterable[tuple[float, float]] = DEFAULT_FREQUENCY_BANDS_HZ,
) -> list[dict[str, float | int]]:
    """Extract one row per complete window, including reproducible offsets."""

    rows: list[dict[str, float | int]] = []
    for start_sample, window in iter_windows(
        values,
        sample_rate_hz=sample_rate_hz,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
    ):
        row: dict[str, float | int] = {
            "window_index": len(rows),
            "start_sample": start_sample,
            "end_sample": start_sample + int(window.size),
        }
        row.update(
            extract_features(
                window,
                sample_rate_hz=sample_rate_hz,
                frequency_bands_hz=frequency_bands_hz,
            )
        )
        rows.append(row)
    return rows


def ordered_feature_vector(features: Mapping[str, float]) -> np.ndarray:
    """Convert a feature mapping to the canonical model input order."""

    missing = [name for name in FEATURE_COLUMNS if name not in features]
    if missing:
        raise KeyError(f"missing feature columns: {', '.join(missing)}")
    return np.asarray([features[name] for name in FEATURE_COLUMNS], dtype=np.float64)
