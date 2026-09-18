from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from fabguard.storage import Conflict, Database, StaleRevision


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(f"sqlite:///{tmp_path / 'test.db'}", artifact_root=tmp_path / "artifacts")
    database.artifact_root.mkdir()
    database.create_schema_for_tests()
    return database


def artifact(db: Database) -> str:
    path = db.artifact_root / "recording.bin"
    path.write_bytes(b"signal")
    return db.register_artifact(
        artifact_key="recording:0001", path=path, sha256=hashlib.sha256(b"signal").hexdigest()
    )


def ingest(db: Database, artifact_id: str, key: str = "event:00000001"):
    return db.ingest_event(
        delivery_key=key,
        evidence_artifact_id=artifact_id,
        prediction={"score": 0.8, "threshold": 0.4, "decision": "abnormal"},
        evidence_version="ev1",
        model_version="m1",
        graph_version="g1",
        prompt_version="p1",
    )


def test_duplicate_delivery_returns_one_incident_and_rejects_changed_payload(db: Database):
    artifact_id = artifact(db)
    first, created = ingest(db, artifact_id)
    second, duplicate_created = ingest(db, artifact_id)
    assert created is True and duplicate_created is False
    assert first["id"] == second["id"]
    with pytest.raises(Conflict):
        db.ingest_event(
            delivery_key="event:00000001",
            evidence_artifact_id=artifact_id,
            prediction={"score": 0.1},
            evidence_version="ev1",
            model_version="m1",
            graph_version="g1",
            prompt_version="p1",
        )


def test_report_revision_and_approval_are_immutable_and_idempotent(db: Database):
    incident, _ = ingest(db, artifact(db))
    db.save_report(
        incident["id"],
        revision=1,
        status="ready",
        content={"observations": [], "guidance": []},
        ticket_draft={"title": "Inspect"},
    )
    ticket, created = db.approve_ticket(
        incident["id"],
        report_revision=1,
        reviewer_id="operator-1",
        action="create_internal_ticket",
        idempotency_key="approval:0001",
    )
    duplicate, duplicate_created = db.approve_ticket(
        incident["id"],
        report_revision=1,
        reviewer_id="operator-1",
        action="create_internal_ticket",
        idempotency_key="approval:0001",
    )
    assert created is True and duplicate_created is False and duplicate["id"] == ticket["id"]
    db.request_investigation(incident["id"], request_key="revision:0002")
    late_retry, late_retry_created = db.approve_ticket(
        incident["id"],
        report_revision=1,
        reviewer_id="operator-1",
        action="create_internal_ticket",
        idempotency_key="approval:0001",
    )
    assert late_retry_created is False and late_retry["id"] == ticket["id"]
    with pytest.raises(StaleRevision):
        db.approve_ticket(
            incident["id"],
            report_revision=1,
            reviewer_id="operator-2",
            action="create_internal_ticket",
            idempotency_key="approval:0002",
        )
    db.save_report(
        incident["id"],
        revision=2,
        status="ready",
        content={"observations": ["new"]},
        ticket_draft={"title": "Inspect again"},
    )
    with pytest.raises(StaleRevision):
        db.approve_ticket(
            incident["id"],
            report_revision=1,
            reviewer_id="operator-2",
            action="create_internal_ticket",
            idempotency_key="approval:0003",
        )


def test_artifact_paths_cannot_escape_root(db: Database, tmp_path: Path):
    outside = tmp_path / "outside"
    outside.write_text("private")
    with pytest.raises(ValueError, match="escapes"):
        db.register_artifact(
            artifact_key="artifact:escape",
            path=outside,
            sha256=hashlib.sha256(b"private").hexdigest(),
        )


def test_registered_artifact_is_content_addressed_and_verified(db: Database):
    source = db.artifact_root / "mutable.bin"
    source.write_bytes(b"original")
    artifact_id = db.register_artifact(
        artifact_key="artifact:immutable",
        path=source,
        sha256=hashlib.sha256(b"original").hexdigest(),
    )
    source.write_bytes(b"tampered")

    resolved = db.resolve_artifact(artifact_id)

    assert resolved.read_bytes() == b"original"
    assert resolved != source
