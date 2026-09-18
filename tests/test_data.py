from dataclasses import replace

import numpy as np
import pytest
from scipy.io import savemat

from fabguard.audit import main as audit_main
from fabguard.data import (
    Cohort,
    FaultFamily,
    HealthState,
    Manufacturer,
    audit_dataset,
    audit_recording,
    discover_recordings,
    freeze_cohorts,
    leave_one_bearing_out,
    metadata_summary,
    parse_recording_identity,
    read_numeric_table,
    state_to_binary_label,
)


def _recording_matrix(seed: int, *, load: float, samples: int = 1_000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.normal(scale=0.2, size=samples),
            rng.normal(scale=0.1, size=samples),
            rng.normal(loc=1_750, scale=1, size=samples),
            rng.normal(loc=load, scale=1, size=samples),
            rng.normal(loc=25, scale=0.1, size=samples),
        ]
    )


def _write_csv(path, matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, delimiter=",")


def test_filename_convention_derives_state_family_and_manufacturer():
    healthy = parse_recording_identity("H_1_0.csv")
    ball = parse_recording_identity("B-11-2")

    assert healthy.health_state is HealthState.HEALTHY
    assert healthy.fault_family is FaultFamily.INNER_RACE
    assert healthy.manufacturer is Manufacturer.NSK
    assert ball.health_state is HealthState.FAULTY
    assert ball.fault_family is FaultFamily.BALL
    assert ball.manufacturer is Manufacturer.FAFNIR
    assert state_to_binary_label(healthy.health_state) == 0
    assert state_to_binary_label(ball.health_state) == 1
    assert healthy.source_stem == "H_1_0"
    assert healthy.recording_key == "H-1-0"


def test_naming_discrepancy_is_preserved_instead_of_silently_corrected():
    identity = parse_recording_identity("H-6-1")
    assert identity.health_state is HealthState.DEVELOPING
    assert identity.discrepancies == ("prefix_H_expected_O",)
    with pytest.raises(ValueError, match="unrecognized"):
        parse_recording_identity("bearing-six")


def test_discovery_groups_format_copies_as_one_recording(tmp_path):
    for folder, suffix in [("csv", ".csv"), ("mat", ".mat"), ("excel", ".xlsx")]:
        path = tmp_path / folder / f"H-1-0{suffix}"
        path.parent.mkdir()
        path.touch()
    unknown = tmp_path / "notes.xlsx"
    unknown.touch()

    records = discover_recordings(tmp_path)

    assert len(records) == 2
    recording = next(record for record in records if record.identity)
    assert len(recording.representations) == 3
    assert recording.canonical_path.suffix == ".csv"
    assert next(record for record in records if record.identity is None).source_stem == "notes"


def test_csv_reader_ignores_header_and_mat_reader_finds_signal_matrix(tmp_path):
    matrix = _recording_matrix(1, load=400, samples=12)
    csv_path = tmp_path / "signal.csv"
    csv_path.write_text("acc,audio,rpm,load,temp\n")
    with csv_path.open("ab") as destination:
        np.savetxt(destination, matrix, delimiter=",")
    mat_path = tmp_path / "signal.mat"
    savemat(mat_path, {"measurement": matrix})

    assert np.allclose(read_numeric_table(csv_path), matrix)
    assert np.allclose(read_numeric_table(mat_path), matrix)


def test_metadata_summary_understands_release_scalar_zero_padding():
    summary = metadata_summary([1819.0, 0.0, 0.0, 0.0])

    assert summary.count == 1
    assert summary.mean == 1819.0
    assert metadata_summary([400.0, 401.0, 399.0]).mean == 400.0


def test_audit_builds_serializable_manifest_and_distinct_quality_decisions(tmp_path):
    path = tmp_path / "csv" / "I-1-1.csv"
    _write_csv(path, _recording_matrix(2, load=400))
    discovered = discover_recordings(tmp_path)[0]

    entry = audit_recording(discovered, sample_rate_hz=100, expected_sample_count=1_000)
    payload = entry.to_dict()

    assert entry.cohort is Cohort.PRIMARY
    assert entry.sample_count == 1_000
    assert entry.duration_seconds == 10
    assert entry.technically_usable
    assert entry.predictive_usefulness == "not_assessed"
    assert payload["health_state"] == "developing"
    assert payload["vibration_quality"]["is_usable"] is True
    assert len(entry.sha256) == 64 and len(entry.content_sha256) == 64


def test_load_changed_bearings_are_quarantined_even_when_technically_usable(tmp_path):
    path = tmp_path / "B-11-2.csv"
    _write_csv(path, _recording_matrix(3, load=0))
    entry = audit_recording(
        discover_recordings(tmp_path)[0],
        sample_rate_hz=100,
        expected_sample_count=1_000,
    )
    assert entry.technically_usable
    assert entry.cohort is Cohort.QUARANTINED_LOAD
    assert entry.predictive_usefulness == "confounded_operating_condition"


def test_unusable_audio_does_not_exclude_valid_vibration_from_primary_cohort(tmp_path):
    matrix = _recording_matrix(4, load=400)
    matrix[:, 1] = 0
    _write_csv(tmp_path / "I_1_1.csv", matrix)
    entry = audit_recording(
        discover_recordings(tmp_path)[0],
        sample_rate_hz=100,
        expected_sample_count=1_000,
    )
    assert entry.vibration_quality.is_usable
    assert not entry.audio_quality.is_usable
    assert entry.technically_usable
    assert entry.cohort is Cohort.PRIMARY


def test_freeze_cohorts_and_lobo_keep_all_states_of_bearing_together(tmp_path):
    names_and_loads = [
        ("H-1-0", 400),
        ("I-1-1", 400),
        ("I-1-2", 400),
        ("H-6-0", 400),
        ("O-6-1", 400),
        ("O-6-2", 400),
        ("H-11-0", 400),
        ("B-11-1", 0),
        ("B-11-2", 0),
    ]
    for seed, (name, load) in enumerate(names_and_loads):
        _write_csv(tmp_path / f"{name}.csv", _recording_matrix(seed + 10, load=load))
    audit = audit_dataset(tmp_path, sample_rate_hz=100, expected_sample_count=1_000)
    frozen = freeze_cohorts(audit.entries)

    assert {entry.bearing_id for entry in frozen if entry.cohort is Cohort.PRIMARY} == {1, 6}
    assert {entry.bearing_id for entry in frozen if entry.cohort is Cohort.QUARANTINED_LOAD} == {11}

    folds = leave_one_bearing_out(frozen)
    assert len(folds) == 2
    for fold in folds:
        test_entries = [entry for entry in frozen if entry.recording_id in fold.test_recording_ids]
        train_entries = [
            entry for entry in frozen if entry.recording_id in fold.train_recording_ids
        ]
        assert {entry.bearing_id for entry in test_entries} == {fold.test_bearing_id}
        assert fold.test_bearing_id not in {entry.bearing_id for entry in train_entries}
        assert {entry.binary_label for entry in test_entries} == {0, 1}


def test_incomplete_bearing_is_excluded_before_fold_construction(tmp_path):
    for seed, name in enumerate(("H-1-0", "I-1-1")):
        _write_csv(tmp_path / f"{name}.csv", _recording_matrix(seed, load=400))
    audit = audit_dataset(tmp_path, sample_rate_hz=100, expected_sample_count=1_000)
    frozen = freeze_cohorts(audit.entries)
    assert all(entry.cohort is Cohort.UNRESOLVED_METADATA for entry in frozen)
    assert all("incomplete_bearing_states" in entry.issues for entry in frozen)
    assert leave_one_bearing_out(frozen) == ()


def test_duplicate_recording_ids_are_rejected_for_lobo(tmp_path):
    for seed, name in enumerate(("H-1-0", "I-1-1", "I-1-2")):
        _write_csv(tmp_path / f"{name}.csv", _recording_matrix(seed, load=400))
    frozen = freeze_cohorts(
        audit_dataset(tmp_path, sample_rate_hz=100, expected_sample_count=1_000).entries
    )
    duplicate = replace(frozen[-1], recording_id=frozen[0].recording_id)
    with pytest.raises(ValueError, match="unique"):
        leave_one_bearing_out((*frozen[:-1], duplicate))


def test_audit_cli_writes_machine_readable_manifest(tmp_path):
    dataset = tmp_path / "raw"
    _write_csv(dataset / "H_1_0.csv", _recording_matrix(41, load=400))
    output = tmp_path / "manifest.json"

    status = audit_main(
        [
            str(dataset),
            "--output",
            str(output),
            "--sample-rate",
            "100",
            "--expected-samples",
            "1000",
        ]
    )

    assert status == 0
    contents = output.read_text()
    assert '"source_name": "H_1_0"' in contents
    assert '"cohort": "unresolved_metadata"' in contents
