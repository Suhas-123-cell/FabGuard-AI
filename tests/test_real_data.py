import json
from pathlib import Path

import pytest

from fabguard.calibration import InsufficientBaseline
from fabguard.replay import replay_recording

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/uored_vafcls_v5.json"
MODEL = ROOT / "runs/uored-v5-seed17/model.joblib"

pytestmark = pytest.mark.skipif(
    not (MODEL.exists() and (ROOT / "data/raw/csv/H_1_0.csv").exists()),
    reason="needs the downloaded UORED data and the trained model",
)


def _replay(source_name, tmp_path, **kwargs):
    entries = json.loads(MANIFEST.read_text())["entries"]
    recording_id = next(e["recording_id"] for e in entries if e["source_name"] == source_name)
    return replay_recording(
        manifest_path=MANIFEST,
        model_path=MODEL,
        recording_id=recording_id,
        output_directory=tmp_path,
        **kwargs,
    )


def test_deployed_detector_separates_healthy_and_faulty_bearing_one(tmp_path):
    healthy = _replay("H_1_0", tmp_path)["prediction"]
    faulty = _replay("I_1_1", tmp_path)["prediction"]
    assert healthy["decision"] == "healthy"
    assert faulty["decision"] == "abnormal"
    assert healthy["alert_rule"] == {"min_abnormal": 3, "window_count": 5}


def test_short_baseline_is_refused_and_long_quiet_baseline_keeps_fleet_threshold(tmp_path):
    with pytest.raises(InsufficientBaseline):
        _replay("H_1_0", tmp_path, baseline_scores=[0.0] * 19)
    fleet = _replay("H_1_0", tmp_path)["prediction"]["threshold"]
    calibrated = _replay("H_1_0", tmp_path, baseline_scores=[-9.0] * 200)["prediction"]
    assert calibrated["threshold"] == fleet
    assert calibrated["threshold_source"] == "machine_baseline"
