import hashlib
import json

import numpy as np
import pytest
from joblib import dump

from fabguard.replay import replay_recording
from fabguard.train import HealthyReferenceDetector


def test_replay_scores_opaque_recording_without_label_leakage(tmp_path):
    sample_rate = 100
    time = np.arange(200) / sample_rate
    matrix = np.column_stack(
        [
            np.sin(2 * np.pi * 8 * time),
            0.5 * np.sin(2 * np.pi * 6 * time),
            np.r_[1750, np.zeros(199)],
            np.r_[400, np.zeros(199)],
            np.full(200, 25.0),
        ]
    )
    source = tmp_path / "label-bearing-name.csv"
    np.savetxt(source, matrix, delimiter=",")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "recording_id": "opaque-recording-1",
                        "source_name": "secret-fault-label",
                        "health_state": "faulty",
                        "canonical_path": str(source),
                        "sha256": source_hash,
                        "technically_usable": True,
                    }
                ]
            }
        )
    )
    detector = HealthyReferenceDetector().fit(np.array([[0.6], [0.7], [0.8]]))
    bundle = tmp_path / "model.joblib"
    dump(
        {
            "bundle_version": 1,
            "model_version": "test-model",
            "model_family": "healthy_reference",
            "feature_policy": "vibration",
            "feature_columns": ["vibration__rms"],
            "threshold": 2.0,
            "sample_rate_hz": sample_rate,
            "window_seconds": 1.0,
            "hop_seconds": 0.5,
            "frequency_bands_hz": ((0.0, 50.0),),
            "estimator": detector,
        },
        bundle,
    )
    bundle.with_suffix(".joblib.sha256").write_text(hashlib.sha256(bundle.read_bytes()).hexdigest())

    result = replay_recording(
        manifest_path=manifest,
        model_path=bundle,
        recording_id="opaque-recording-1",
        output_directory=tmp_path / "artifacts",
    )

    assert len(result["windows"]) == 3
    assert result["recording_id"] == "opaque-recording-1"
    assert "source_name" not in result
    assert "health_state" not in result
    assert result["artifact_id"].startswith("art_")

    with bundle.open("ab") as sink:
        sink.write(b"tampered")
    with pytest.raises(ValueError, match="integrity verification"):
        replay_recording(
            manifest_path=manifest,
            model_path=bundle,
            recording_id="opaque-recording-1",
            output_directory=tmp_path / "tampered-artifacts",
        )
