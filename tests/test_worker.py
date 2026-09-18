from __future__ import annotations

import hashlib
import json
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from fabguard.graph import InvestigationGraph, build_langgraph
from fabguard.llm import LocalTemplateLLM
from fabguard.retrieval import InMemoryRetriever, Passage
from fabguard.storage import Database
from fabguard.worker import InvestigationWorker, RuntimeGraphRunner


def setup_db(tmp_path: Path) -> tuple[Database, str]:
    db = Database(f"sqlite:///{tmp_path / 'worker.db'}", artifact_root=tmp_path / "artifacts")
    db.artifact_root.mkdir()
    db.create_schema_for_tests()
    path = db.artifact_root / "evidence"
    path.write_bytes(b"x")
    artifact_id = db.register_artifact(
        artifact_key="worker:evidence", path=path, sha256=hashlib.sha256(b"x").hexdigest()
    )
    incident, _ = db.ingest_event(
        delivery_key="worker:event01",
        evidence_artifact_id=artifact_id,
        prediction={"score": 1},
        evidence_version="e1",
        model_version="m1",
        graph_version="g1",
        prompt_version="p1",
    )
    return db, incident["id"]


def test_worker_processes_and_completes_delivery(tmp_path: Path):
    db, incident_id = setup_db(tmp_path)
    calls = []

    def runner(incident, investigation):
        calls.append(investigation["thread_id"])
        db.save_report(
            incident["id"],
            revision=investigation["revision"],
            status="ready",
            content={},
            ticket_draft={"title": "Inspect"},
        )

    result = InvestigationWorker(db, runner, worker_id="worker-1").run_once()
    assert result.outcome == "completed" and result.incident_id == incident_id
    assert calls == [incident_id]
    assert InvestigationWorker(db, runner, worker_id="worker-1").run_once().outcome == "idle"


def test_failed_run_is_reclaimed_and_resumes_same_revision(tmp_path: Path):
    db, incident_id = setup_db(tmp_path)
    calls = []

    def runner(incident, investigation):
        calls.append(investigation["thread_id"])
        if len(calls) == 1:
            raise RuntimeError("provider unavailable")
        db.save_report(
            incident["id"],
            revision=investigation["revision"],
            status="ready",
            content={},
            ticket_draft={"title": "Inspect"},
        )

    worker = InvestigationWorker(db, runner, worker_id="worker-1")
    assert worker.run_once().outcome == "failed"
    assert worker.run_once().outcome == "completed"
    assert calls == [incident_id, incident_id]


def test_each_delivery_targets_its_own_revision(tmp_path: Path):
    db, incident_id = setup_db(tmp_path)
    second, _ = db.request_investigation(incident_id, request_key="worker:revision2")
    calls = []

    def runner(incident, investigation):
        calls.append(investigation["revision"])
        db.save_report(
            incident["id"],
            revision=investigation["revision"],
            status="ready",
            content={},
            ticket_draft={"title": "Inspect"},
        )

    worker = InvestigationWorker(db, runner, worker_id="worker-1")
    assert worker.run_once().outcome == "completed"
    assert worker.run_once().outcome == "completed"
    assert calls == [1, second["revision"]]


def test_saved_report_is_not_recomputed_after_delivery_failure(tmp_path: Path):
    db, incident_id = setup_db(tmp_path)
    claimed = db.claim_event("crashed-worker")
    assert claimed is not None
    investigation = db.resumable_investigation(incident_id, 1)
    db.mark_investigation_running(investigation["id"])
    db.save_report(
        incident_id, revision=1, status="ready", content={}, ticket_draft={"title": "Inspect"}
    )
    db.fail_event(claimed.id, "crashed-worker", "connection lost")

    def must_not_run(incident, revision):
        raise AssertionError("completed graph revision was repeated")

    result = InvestigationWorker(db, must_not_run, worker_id="replacement-worker").run_once()
    assert result.outcome == "completed"


def test_runtime_runner_turns_replay_artifact_into_verified_report(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    db = Database(f"sqlite:///{tmp_path / 'runtime.db'}", artifact_root=artifact_root)
    db.create_schema_for_tests()
    replay = {
        "prediction": {
            "score": 0.8,
            "threshold": 0.6,
            "decision": "abnormal",
            "selected_window_index": 0,
        },
        "windows": [
            {
                "start_sample": 0,
                "end_sample": 100,
                "vibration_features": {"rms": 2.5},
                "audio_features": {"rms": 1.1},
            }
        ],
        "quality": {
            "vibration": {"is_usable": True, "issues": []},
            "audio": {"is_usable": True, "issues": []},
        },
        "sample_rate_hz": 100,
        "feature_policy": "fusion",
        "preprocessing_version": "features-v1",
        "model_version": "model-v1",
        "evaluation_scope": "test scope",
    }
    artifact_path = artifact_root / "replay.json"
    artifact_path.write_text(json.dumps(replay))
    artifact_id = db.register_artifact(
        artifact_key="runtime:artifact",
        path=artifact_path,
        sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
    )
    incident, _ = db.ingest_event(
        delivery_key="runtime:event",
        evidence_artifact_id=artifact_id,
        prediction={
            "score": 0.8,
            "threshold": 0.6,
            "decision": "abnormal",
            "evaluation_scope": "test scope",
        },
        evidence_version="features-v1",
        model_version="model-v1",
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
    result = InvestigationWorker(
        db, RuntimeGraphRunner(db, graph), worker_id="runtime-worker"
    ).run_once()

    with db.engine.connect() as connection:
        failure = connection.exec_driver_sql(
            "SELECT last_error FROM event_deliveries WHERE incident_id = ?", (incident["id"],)
        ).scalar_one()
    assert result.outcome == "completed", failure
    saved = db.get_incident(incident["id"])["reports"][0]
    assert saved["status"] == "ready"
    assert saved["content"]["trace"][-2:] == ["verifier", "review"]
