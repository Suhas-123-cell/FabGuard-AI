"""UORED-VAFCLS discovery, audit, cohort, and grouped-fold utilities.

The dataset publishes the same 60 recordings as CSV, Excel, and MATLAB files.
Discovery groups those representations by recording name so they cannot be
mistaken for independent samples. Filename metadata is validated against the
published convention and discrepancies remain explicit in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

from fabguard.signals import (
    DEFAULT_SAMPLE_RATE_HZ,
    PairAlignment,
    SignalQuality,
    assess_pair_alignment,
    assess_quality,
)

EXPECTED_SAMPLE_COUNT = 420_000
EXPECTED_DURATION_SECONDS = 10.0
SUPPORTED_SUFFIXES = frozenset({".csv", ".xlsx", ".xls", ".mat"})
FORMAT_PREFERENCE = {".csv": 0, ".mat": 1, ".xlsx": 2, ".xls": 3}

CHANNEL_MAPPING: Mapping[str, int] = {
    "vibration": 0,
    "audio": 1,
    "motor_speed": 2,
    "load": 3,
    "temperature": 4,
}

# The paper identifies sensor outputs but does not establish that every released
# table has already been converted to these engineering units. Unknown units are
# intentionally not inferred from magnitudes.
CHANNEL_UNITS: Mapping[str, str] = {
    "vibration": "source_unit_unknown",
    "audio": "source_unit_unknown",
    "motor_speed": "rpm",
    "load": "N",
    "temperature": "degC",
}


class HealthState(StrEnum):
    HEALTHY = "healthy"
    DEVELOPING = "developing"
    FAULTY = "faulty"


class FaultFamily(StrEnum):
    INNER_RACE = "inner_race"
    OUTER_RACE = "outer_race"
    BALL = "ball"
    CAGE = "cage"


class Manufacturer(StrEnum):
    NSK = "NSK 6203ZZ"
    FAFNIR = "FAFNIR 203KD"


class Cohort(StrEnum):
    PRIMARY = "primary"
    QUARANTINED_LOAD = "quarantined_load"
    CONFOUND_AUDIT = "confound_audit"
    EXCLUDED_QUALITY = "excluded_quality"
    UNRESOLVED_METADATA = "unresolved_metadata"


def state_to_binary_label(state: HealthState | str) -> int:
    """Map healthy to 0 and developing/faulty to 1."""

    parsed = state if isinstance(state, HealthState) else HealthState(state)
    return 0 if parsed is HealthState.HEALTHY else 1


# Alias reads naturally in downstream training code.
health_state_to_binary_label = state_to_binary_label


def eventual_fault_family(bearing_id: int) -> FaultFamily:
    if 1 <= bearing_id <= 5:
        return FaultFamily.INNER_RACE
    if 6 <= bearing_id <= 10:
        return FaultFamily.OUTER_RACE
    if 11 <= bearing_id <= 15:
        return FaultFamily.BALL
    if 16 <= bearing_id <= 20:
        return FaultFamily.CAGE
    raise ValueError(f"bearing_id must be in 1..20, got {bearing_id}")


def bearing_manufacturer(bearing_id: int) -> Manufacturer:
    if 1 <= bearing_id <= 5:
        return Manufacturer.NSK
    if 6 <= bearing_id <= 20:
        return Manufacturer.FAFNIR
    raise ValueError(f"bearing_id must be in 1..20, got {bearing_id}")


@dataclass(frozen=True)
class RecordingIdentity:
    source_stem: str
    prefix: str
    bearing_id: int
    state_index: int
    health_state: HealthState
    fault_family: FaultFamily
    manufacturer: Manufacturer
    discrepancies: tuple[str, ...] = ()

    @property
    def recording_key(self) -> str:
        return f"{self.prefix}-{self.bearing_id}-{self.state_index}"

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class DiscoveredRecording:
    source_stem: str
    representations: tuple[Path, ...]
    identity: RecordingIdentity | None
    discovery_issues: tuple[str, ...] = ()

    @property
    def canonical_path(self) -> Path:
        return min(
            self.representations,
            key=lambda path: (FORMAT_PREFERENCE.get(path.suffix.lower(), 99), str(path)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_stem": self.source_stem,
            "representations": [str(path) for path in self.representations],
            "canonical_path": str(self.canonical_path),
            "identity": self.identity.to_dict() if self.identity else None,
            "discovery_issues": list(self.discovery_issues),
        }


@dataclass(frozen=True)
class NumericSummary:
    count: int
    finite_fraction: float
    minimum: float | None
    maximum: float | None
    mean: float | None
    median: float | None
    standard_deviation: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecordingManifestEntry:
    recording_id: str
    source_name: str
    bearing_id: int
    health_state: HealthState
    binary_label: int
    eventual_fault_family: FaultFamily
    manufacturer: Manufacturer
    cohort: Cohort
    representations: tuple[str, ...]
    canonical_path: str
    canonical_format: str
    sha256: str
    content_sha256: str
    annotation_source: str
    sample_rate_hz: float
    sample_count: int
    duration_seconds: float
    channel_mapping: Mapping[str, int]
    channel_units: Mapping[str, str]
    load: NumericSummary
    motor_speed: NumericSummary
    vibration_quality: SignalQuality
    audio_quality: SignalQuality
    alignment: PairAlignment
    technically_usable: bool
    predictive_usefulness: str
    issues: tuple[str, ...] = ()
    naming_discrepancies: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class DatasetAudit:
    root: str
    entries: tuple[RecordingManifestEntry, ...]
    unparsed_files: tuple[str, ...]
    duplicate_content_groups: tuple[tuple[str, ...], ...]
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def write_json(self, destination: str | Path) -> None:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class LOBOFold:
    fold_id: str
    test_bearing_id: int
    train_recording_ids: tuple[str, ...]
    test_recording_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_IDENTITY_PATTERN = re.compile(
    r"^(?P<prefix>[HIOBC])(?P<separator1>[-_])(?P<bearing>\d{1,2})"
    r"(?P<separator2>[-_])(?P<state>[012])$",
    re.I,
)
_STATE_BY_INDEX = {0: HealthState.HEALTHY, 1: HealthState.DEVELOPING, 2: HealthState.FAULTY}
_PREFIX_BY_FAMILY = {
    FaultFamily.INNER_RACE: "I",
    FaultFamily.OUTER_RACE: "O",
    FaultFamily.BALL: "B",
    FaultFamily.CAGE: "C",
}


def parse_recording_identity(stem_or_path: str | Path) -> RecordingIdentity:
    """Parse and validate the published ``Letter-Bearing-State`` convention."""

    stem = Path(stem_or_path).stem
    match = _IDENTITY_PATTERN.fullmatch(stem)
    if not match:
        raise ValueError(f"unrecognized recording name {stem!r}; expected H-1-0 style")
    prefix = match.group("prefix").upper()
    bearing_id = int(match.group("bearing"))
    state_index = int(match.group("state"))
    family = eventual_fault_family(bearing_id)
    state = _STATE_BY_INDEX[state_index]
    expected_prefix = "H" if state is HealthState.HEALTHY else _PREFIX_BY_FAMILY[family]
    discrepancies: list[str] = []
    if match.group("separator1") != match.group("separator2"):
        discrepancies.append("mixed_filename_separators")
    if prefix != expected_prefix:
        discrepancies.append(f"prefix_{prefix}_expected_{expected_prefix}")
    return RecordingIdentity(
        source_stem=stem,
        prefix=prefix,
        bearing_id=bearing_id,
        state_index=state_index,
        health_state=state,
        fault_family=family,
        manufacturer=bearing_manufacturer(bearing_id),
        discrepancies=tuple(discrepancies),
    )


def discover_recordings(root: str | Path) -> tuple[DiscoveredRecording, ...]:
    """Recursively group CSV/MAT/XLS(X) representations by case-insensitive stem."""

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {root_path}")
    groups: dict[str, list[Path]] = {}
    display_stems: dict[str, str] = {}
    for path in root_path.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        key = path.stem.casefold()
        groups.setdefault(key, []).append(path)
        display_stems.setdefault(key, path.stem)

    discovered: list[DiscoveredRecording] = []
    for key in sorted(groups):
        paths = tuple(sorted(groups[key], key=lambda path: str(path)))
        issues: list[str] = []
        suffix_counts: dict[str, int] = {}
        for path in paths:
            suffix_counts[path.suffix.lower()] = suffix_counts.get(path.suffix.lower(), 0) + 1
        for suffix, count in suffix_counts.items():
            if count > 1:
                issues.append(f"multiple_{suffix.removeprefix('.')}_representations")
        try:
            identity = parse_recording_identity(display_stems[key])
        except ValueError as error:
            identity = None
            issues.append(str(error))
        discovered.append(
            DiscoveredRecording(
                source_stem=display_stems[key],
                representations=paths,
                identity=identity,
                discovery_issues=tuple(issues),
            )
        )
    return tuple(discovered)


def sha256_file(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: str | Path) -> str:
    """Prefer repository-relative paths while preserving external paths."""

    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _coerce_numeric(frame: pd.DataFrame) -> np.ndarray:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    # Header and comment rows become fully NaN and are not signal samples.
    numeric = numeric.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return numeric.to_numpy(dtype=np.float64, copy=False)


def iter_numeric_chunks(path: str | Path, *, chunk_rows: int = 65_536) -> Iterator[np.ndarray]:
    """Yield numeric table chunks while avoiding whole-CSV memory residency."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        for frame in pd.read_csv(
            source,
            header=None,
            chunksize=chunk_rows,
            low_memory=False,
            on_bad_lines="error",
        ):
            array = _coerce_numeric(frame)
            if array.size:
                yield array
        return
    if suffix in {".xlsx", ".xls"}:
        try:
            frame = pd.read_excel(source, header=None)
        except ImportError as error:
            raise RuntimeError(
                "Excel audit requires a pandas Excel engine such as openpyxl"
            ) from error
        array = _coerce_numeric(frame)
        for start in range(0, array.shape[0], chunk_rows):
            yield array[start : start + chunk_rows]
        return
    if suffix == ".mat":
        matrix = _matrix_from_mat(source)
        for start in range(0, matrix.shape[0], chunk_rows):
            yield matrix[start : start + chunk_rows]
        return
    raise ValueError(f"unsupported signal table format: {suffix}")


def read_numeric_table(path: str | Path) -> np.ndarray:
    chunks = list(iter_numeric_chunks(path))
    if not chunks:
        raise ValueError(f"no numeric rows found in {path}")
    column_counts = {chunk.shape[1] for chunk in chunks}
    if len(column_counts) != 1:
        raise ValueError(f"inconsistent numeric column counts in {path}: {sorted(column_counts)}")
    return np.concatenate(chunks, axis=0)


def _matrix_from_mat(path: Path) -> np.ndarray:
    try:
        variables = loadmat(path, squeeze_me=True)
    except NotImplementedError as error:
        raise RuntimeError("MATLAB v7.3/HDF5 files require an HDF5 reader") from error
    candidates: list[np.ndarray] = []
    vectors: list[tuple[str, np.ndarray]] = []
    for name, value in variables.items():
        if (
            name.startswith("__")
            or not isinstance(value, np.ndarray)
            or not np.issubdtype(value.dtype, np.number)
        ):
            continue
        squeezed = np.asarray(value).squeeze()
        if squeezed.ndim == 2 and min(squeezed.shape) >= len(CHANNEL_MAPPING):
            matrix = squeezed.T if squeezed.shape[0] == len(CHANNEL_MAPPING) else squeezed
            candidates.append(np.asarray(matrix, dtype=np.float64))
        elif squeezed.ndim == 1:
            vectors.append((name, np.asarray(squeezed, dtype=np.float64)))
    if candidates:
        return max(candidates, key=lambda candidate: candidate.shape[0] * candidate.shape[1])
    if vectors:
        lengths = {vector.size for _, vector in vectors}
        if len(lengths) == 1 and len(vectors) >= len(CHANNEL_MAPPING):
            vectors.sort(key=lambda item: item[0])
            return np.column_stack([vector for _, vector in vectors])
    raise ValueError(f"no numeric signal matrix found in {path}")


def numeric_summary(values: Sequence[float] | np.ndarray) -> NumericSummary:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return NumericSummary(array.size, 0.0, None, None, None, None, None)
    return NumericSummary(
        count=int(array.size),
        finite_fraction=float(finite.size / array.size) if array.size else 0.0,
        minimum=float(np.min(finite)),
        maximum=float(np.max(finite)),
        mean=float(np.mean(finite)),
        median=float(np.median(finite)),
        standard_deviation=float(np.std(finite)),
    )


def metadata_summary(values: Sequence[float] | np.ndarray) -> NumericSummary:
    """Summarize a metadata column, including UORED's scalar-plus-zero encoding.

    The v5 CSV files store speed and load in the first row and pad the rest of
    those columns with zeros.  Treating that representation as a sampled time
    series produces a false near-zero mean.  Synthetic or alternate releases
    with genuinely varying metadata still use the ordinary numeric summary.
    """

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size and np.isfinite(array[0]) and np.all(array[1:] == 0.0):
        scalar = float(array[0])
        return NumericSummary(
            count=1,
            finite_fraction=1.0,
            minimum=scalar,
            maximum=scalar,
            mean=scalar,
            median=scalar,
            standard_deviation=0.0,
        )
    return numeric_summary(array)


def content_sha256(matrix: np.ndarray) -> str:
    """Hash normalized numeric content so format copies can be compared."""

    contiguous = np.ascontiguousarray(np.asarray(matrix, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _nominal_load_matches(
    identity: RecordingIdentity,
    summary: NumericSummary,
    tolerance_n: float,
) -> bool:
    if summary.mean is None:
        return False
    ball_fault = (
        identity.fault_family is FaultFamily.BALL
        and identity.health_state is not HealthState.HEALTHY
    )
    expected = 0.0 if ball_fault else 400.0
    return abs(summary.mean - expected) <= tolerance_n


def audit_recording(
    recording: DiscoveredRecording,
    *,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    expected_sample_count: int = EXPECTED_SAMPLE_COUNT,
    load_tolerance_n: float = 80.0,
) -> RecordingManifestEntry:
    if recording.identity is None:
        raise ValueError(f"cannot audit unparsed recording {recording.source_stem!r}")
    identity = recording.identity
    canonical = recording.canonical_path
    matrix = read_numeric_table(canonical)
    issues = list(recording.discovery_issues)
    if matrix.ndim != 2 or matrix.shape[1] < len(CHANNEL_MAPPING):
        raise ValueError(f"{canonical} has shape {matrix.shape}; expected at least five columns")
    if matrix.shape[1] != len(CHANNEL_MAPPING):
        issues.append(f"unexpected_column_count_{matrix.shape[1]}")

    vibration = matrix[:, CHANNEL_MAPPING["vibration"]]
    audio = matrix[:, CHANNEL_MAPPING["audio"]]
    speed = metadata_summary(matrix[:, CHANNEL_MAPPING["motor_speed"]])
    load = metadata_summary(matrix[:, CHANNEL_MAPPING["load"]])
    vibration_quality = assess_quality(vibration, sample_rate_hz=sample_rate_hz)
    audio_quality = assess_quality(audio, sample_rate_hz=sample_rate_hz)
    alignment = assess_pair_alignment(vibration, audio, sample_rate_hz=sample_rate_hz)
    if matrix.shape[0] != expected_sample_count:
        issues.append(f"sample_count_{matrix.shape[0]}_expected_{expected_sample_count}")
    if not _nominal_load_matches(identity, load, load_tolerance_n):
        issues.append("load_metadata_mismatch")
    if speed.mean is None or speed.mean <= 0:
        issues.append("implausible_motor_speed")
    issues.extend(f"vibration:{issue}" for issue in vibration_quality.issues)
    issues.extend(f"audio:{issue}" for issue in audio_quality.issues)
    issues.extend(f"alignment:{issue}" for issue in alignment.issues)
    # The primary detector remains eligible when vibration is valid even if
    # audio fails its separate modality gate. An audio failure must lead to a
    # vibration-only policy, not the loss of an otherwise usable recording.
    technically_usable = vibration_quality.is_usable

    if not technically_usable:
        cohort = Cohort.EXCLUDED_QUALITY
    elif identity.discrepancies:
        cohort = Cohort.UNRESOLVED_METADATA
    elif identity.fault_family is FaultFamily.BALL:
        cohort = Cohort.QUARANTINED_LOAD
    elif "load_metadata_mismatch" in issues:
        cohort = Cohort.UNRESOLVED_METADATA
    else:
        cohort = Cohort.PRIMARY
    predictive_usefulness = (
        "confounded_operating_condition"
        if identity.fault_family is FaultFamily.BALL
        else "not_assessed"
    )

    stable_id = hashlib.sha256(identity.recording_key.encode("ascii")).hexdigest()[:16]
    return RecordingManifestEntry(
        recording_id=f"uored-{stable_id}",
        source_name=recording.source_stem,
        bearing_id=identity.bearing_id,
        health_state=identity.health_state,
        binary_label=state_to_binary_label(identity.health_state),
        eventual_fault_family=identity.fault_family,
        manufacturer=identity.manufacturer,
        cohort=cohort,
        representations=tuple(portable_path(path) for path in recording.representations),
        canonical_path=portable_path(canonical),
        canonical_format=canonical.suffix.lower().removeprefix("."),
        sha256=sha256_file(canonical),
        content_sha256=content_sha256(matrix[:, : len(CHANNEL_MAPPING)]),
        annotation_source="published filename convention plus measured raw channels",
        sample_rate_hz=float(sample_rate_hz),
        sample_count=int(matrix.shape[0]),
        duration_seconds=float(matrix.shape[0] / sample_rate_hz),
        channel_mapping=dict(CHANNEL_MAPPING),
        channel_units=dict(CHANNEL_UNITS),
        load=load,
        motor_speed=speed,
        vibration_quality=vibration_quality,
        audio_quality=audio_quality,
        alignment=alignment,
        technically_usable=technically_usable,
        predictive_usefulness=predictive_usefulness,
        issues=tuple(dict.fromkeys(issues)),
        naming_discrepancies=identity.discrepancies,
    )


def audit_dataset(
    root: str | Path,
    *,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    expected_sample_count: int = EXPECTED_SAMPLE_COUNT,
    load_tolerance_n: float = 80.0,
) -> DatasetAudit:
    """Audit all parseable recordings and report duplicates and omissions."""

    discovered = discover_recordings(root)
    entries: list[RecordingManifestEntry] = []
    unparsed: list[str] = []
    top_level_issues: list[str] = []
    for recording in discovered:
        if recording.identity is None:
            unparsed.extend(str(path) for path in recording.representations)
            continue
        try:
            entries.append(
                audit_recording(
                    recording,
                    sample_rate_hz=sample_rate_hz,
                    expected_sample_count=expected_sample_count,
                    load_tolerance_n=load_tolerance_n,
                )
            )
        except (OSError, ValueError, RuntimeError) as error:
            top_level_issues.append(f"{recording.source_stem}: {error}")

    content_groups: dict[str, list[str]] = {}
    for entry in entries:
        content_groups.setdefault(entry.content_sha256, []).append(entry.recording_id)
    duplicate_groups = tuple(tuple(sorted(ids)) for ids in content_groups.values() if len(ids) > 1)
    if len(entries) != 60:
        top_level_issues.append(f"recording_count_{len(entries)}_expected_60")
    expected_keys = {(bearing, state) for bearing in range(1, 21) for state in range(3)}
    actual_keys = {(entry.bearing_id, _state_index(entry.health_state)) for entry in entries}
    for bearing, state in sorted(expected_keys - actual_keys):
        top_level_issues.append(f"missing_bearing_{bearing}_state_{state}")

    return DatasetAudit(
        root=portable_path(Path(root).expanduser()),
        entries=tuple(
            sorted(entries, key=lambda entry: (entry.bearing_id, _state_index(entry.health_state)))
        ),
        unparsed_files=tuple(sorted(unparsed)),
        duplicate_content_groups=duplicate_groups,
        issues=tuple(top_level_issues),
    )


def freeze_cohorts(
    entries: Iterable[RecordingManifestEntry],
    *,
    require_complete_states: bool = True,
) -> tuple[RecordingManifestEntry, ...]:
    """Return entries with deterministic cohort assignments frozen.

    Quality failures and unresolved naming/load findings are excluded. Bearings
    11--15 always remain in the all-data confound audit but never the primary
    cohort. If requested, a bearing missing any state is excluded as a unit so
    every primary LOBO test fold contains both binary classes.
    """

    records = list(entries)
    states_by_bearing: dict[int, set[HealthState]] = {}
    for entry in records:
        states_by_bearing.setdefault(entry.bearing_id, set()).add(entry.health_state)
    complete = set(HealthState)
    frozen: list[RecordingManifestEntry] = []
    for entry in records:
        issues = list(entry.issues)
        if not entry.technically_usable:
            cohort = Cohort.EXCLUDED_QUALITY
        elif entry.naming_discrepancies or "load_metadata_mismatch" in entry.issues:
            cohort = Cohort.UNRESOLVED_METADATA
        elif 11 <= entry.bearing_id <= 15:
            cohort = Cohort.QUARANTINED_LOAD
        elif require_complete_states and states_by_bearing.get(entry.bearing_id, set()) != complete:
            cohort = Cohort.UNRESOLVED_METADATA
            issues.append("incomplete_bearing_states")
        else:
            cohort = Cohort.PRIMARY
        frozen.append(replace(entry, cohort=cohort, issues=tuple(dict.fromkeys(issues))))
    return tuple(
        sorted(frozen, key=lambda entry: (entry.bearing_id, _state_index(entry.health_state)))
    )


def leave_one_bearing_out(
    entries: Iterable[RecordingManifestEntry],
    *,
    cohort: Cohort | None = Cohort.PRIMARY,
) -> tuple[LOBOFold, ...]:
    """Build grouped outer folds; never split states or modalities by window."""

    records = [entry for entry in entries if cohort is None or entry.cohort is cohort]
    if not records:
        return ()
    ids = [entry.recording_id for entry in records]
    if len(ids) != len(set(ids)):
        raise ValueError("recording_id values must be unique")
    folds: list[LOBOFold] = []
    for bearing_id in sorted({entry.bearing_id for entry in records}):
        test = [entry for entry in records if entry.bearing_id == bearing_id]
        if {entry.binary_label for entry in test} != {0, 1}:
            raise ValueError(f"bearing {bearing_id} does not contain both binary labels")
        folds.append(
            LOBOFold(
                fold_id=f"bearing-{bearing_id:02d}",
                test_bearing_id=bearing_id,
                train_recording_ids=tuple(
                    entry.recording_id for entry in records if entry.bearing_id != bearing_id
                ),
                test_recording_ids=tuple(entry.recording_id for entry in test),
            )
        )
    return tuple(folds)


# Common alternate spelling used in notebooks and training modules.
build_lobo_folds = leave_one_bearing_out


def _state_index(state: HealthState) -> int:
    return {HealthState.HEALTHY: 0, HealthState.DEVELOPING: 1, HealthState.FAULTY: 2}[state]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value
