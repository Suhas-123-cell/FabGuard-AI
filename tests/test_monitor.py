import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile

from fabguard.calibration import InsufficientBaseline
from fabguard.monitor import SAMPLE_RATE_HZ, Monitor, read_pcm, read_wav, run


def _machine(seconds: float, *, fault_after: float | None = None, rate: int = SAMPLE_RATE_HZ):
    """A hum plus noise; after ``fault_after`` seconds, periodic impacts like a damaged bearing."""
    rng = np.random.default_rng(5)
    t = np.arange(int(seconds * rate)) / rate
    signal = 0.2 * np.sin(2 * np.pi * 120 * t) + rng.normal(0, 0.02, t.size)
    if fault_after is not None:
        impacts = (t >= fault_after) & (np.mod(t, 1 / 37) < 0.0004)
        signal[impacts] += 0.6
    return signal


def _events(signal, tmp_path, baseline_seconds=61.0, **kwargs):
    monitor = Monitor(baseline_seconds=baseline_seconds, modality="vibration", source="fan-1")
    chunks = (signal[i : i + 9_999] for i in range(0, signal.size, 9_999))  # odd chunk size
    return list(run(chunks, monitor, output_directory=tmp_path, **kwargs))


def test_commissions_then_stays_quiet_on_the_same_machine(tmp_path):
    events = _events(_machine(100), tmp_path)
    assert events[0]["event"] == "commissioned" and events[0]["baseline_windows"] == 121
    windows = [e for e in events if e["event"] in {"window", "alarm"}]
    assert len(windows) == (100 - 1) * 2 + 1 - 121
    assert not any(e["alarm"] for e in windows)
    assert not list(tmp_path.iterdir())


def test_fault_raises_one_alarm_with_replay_shaped_evidence(tmp_path):
    submitted = []
    events = _events(_machine(100, fault_after=80), tmp_path, submit=submitted.append)
    alarms = [e for e in events if e["event"] == "alarm"]
    assert len(alarms) == 1 and 80 <= alarms[0]["t_seconds"] <= 82
    assert all(e["alarm"] for e in events if e["t_seconds"] > alarms[0]["t_seconds"])
    artifact = json.loads(Path(alarms[0]["artifact_path"]).read_text())
    assert submitted[0]["artifact_id"] == artifact["artifact_id"]
    selected = artifact["windows"][artifact["prediction"]["selected_window_index"]]
    assert selected["anomaly_score"] == artifact["prediction"]["score"] >= selected["threshold"]
    assert artifact["windows"][0]["start_sample"] == 0 and len(artifact["windows"]) == 20
    assert set(selected["vibration_features"]) and selected["audio_features"] == {}
    assert artifact["quality"]["vibration"]["is_usable"]
    assert artifact["quality"]["audio"] == {"is_usable": False, "issues": ["not_recorded"]}


def test_alarm_evidence_runs_through_the_investigation_worker(tmp_path):
    from langgraph.checkpoint.memory import InMemorySaver

    from fabguard.graph import InvestigationGraph, build_langgraph
    from fabguard.llm import LocalTemplateLLM
    from fabguard.retrieval import InMemoryRetriever, Passage
    from fabguard.storage import Database, file_sha256
    from fabguard.worker import InvestigationWorker, RuntimeGraphRunner

    root = tmp_path / "artifacts"
    alarm = next(e for e in _events(_machine(100, fault_after=80), root) if e["event"] == "alarm")
    artifact = json.loads(Path(alarm["artifact_path"]).read_text())
    db = Database(f"sqlite:///{tmp_path / 'runtime.db'}", artifact_root=root)
    db.create_schema_for_tests()
    registered = db.register_artifact(
        artifact_key=f"monitor:{artifact['artifact_id']}",
        path=alarm["artifact_path"],
        sha256=file_sha256(alarm["artifact_path"]),
    )
    prediction = artifact["prediction"]
    incident, _ = db.ingest_event(
        delivery_key=f"monitor:{artifact['artifact_id']}",
        evidence_artifact_id=registered,
        prediction={
            **{key: prediction[key] for key in ("score", "threshold", "decision")},
            "evaluation_scope": artifact["evaluation_scope"],
        },
        evidence_version=artifact["preprocessing_version"],
        model_version=artifact["model_version"],
        graph_version="graph-v1",
        prompt_version="prompt-v1",
    )
    passage = Passage(
        source_id="guide",
        passage_id="guide:p1",
        title="Bearing anomaly inspection guidance",
        text="Inspection should preserve safety controls and compare operating conditions.",
        section="Inspection",
        applicability=("general",),
    )
    graph = build_langgraph(
        InvestigationGraph(InMemoryRetriever([passage]), LocalTemplateLLM(), adaptive=False),
        checkpointer=InMemorySaver(),
    )
    result = InvestigationWorker(db, RuntimeGraphRunner(db, graph), worker_id="w").run_once()
    assert result.outcome == "completed"
    assert db.get_incident(incident["id"])["reports"][0]["status"] == "ready"


def test_short_baseline_is_refused():
    with pytest.raises(InsufficientBaseline):
        Monitor(baseline_seconds=30, modality="audio", source="fan-1")


def test_wav_and_pcm_inputs_are_resampled_to_42_khz(tmp_path):
    signal = _machine(2, rate=48_000)
    path = tmp_path / "fan.wav"
    soundfile.write(path, np.column_stack([signal, signal]), 48_000)
    assert sum(len(chunk) for chunk in read_wav(path, channel=1)) == 2 * SAMPLE_RATE_HZ
    pcm = (signal * 32767).astype("<i2").tobytes()
    chunks = list(read_pcm(io.BytesIO(pcm[:-1]), 48_000))  # a torn final sample is dropped
    assert abs(sum(len(chunk) for chunk in chunks) - 2 * SAMPLE_RATE_HZ) <= len(chunks)
    with pytest.raises(ValueError, match="channel"):
        next(read_wav(path, channel=2))
