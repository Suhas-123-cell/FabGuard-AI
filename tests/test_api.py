from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fabguard.api import Principal, create_app
from fabguard.storage import Database


@pytest.fixture
def api(tmp_path: Path):
    db = Database(f"sqlite:///{tmp_path / 'api.db'}", artifact_root=tmp_path / "artifacts")
    db.artifact_root.mkdir()
    db.create_schema_for_tests()
    evidence = db.artifact_root / "evidence.bin"
    evidence.write_bytes(b"evidence")
    artifact_id = db.register_artifact(
        artifact_key="evidence:0001", path=evidence, sha256=hashlib.sha256(b"evidence").hexdigest()
    )
    app = create_app(
        db,
        api_keys={
            "producer-token": Principal("replay", frozenset({"producer"})),
            "analyst-token": Principal("analyst", frozenset({"analyst"})),
            "reviewer-token": Principal("reviewer-1", frozenset({"reviewer"})),
        },
    )
    return TestClient(app), db, artifact_id


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def create_incident(client: TestClient, artifact_id: str) -> str:
    body = {
        "evidence_artifact_id": artifact_id,
        "prediction": {
            "score": 0.9,
            "threshold": 0.5,
            "decision": "abnormal",
            "evaluation_scope": "held-out bearing",
        },
        "evidence_version": "ev1",
        "model_version": "m1",
        "graph_version": "g1",
        "prompt_version": "p1",
    }
    response = client.post(
        "/v1/events",
        json=body,
        headers={**auth("producer-token"), "Idempotency-Key": "event:api-0001"},
    )
    assert response.status_code == 201, response.text
    return response.json()["incident"]["id"]


def test_auth_and_event_idempotency(api):
    client, _, artifact_id = api
    assert client.post("/v1/events", json={}).status_code == 401
    incident_id = create_incident(client, artifact_id)
    body = {
        "evidence_artifact_id": artifact_id,
        "prediction": {
            "score": 0.9,
            "threshold": 0.5,
            "decision": "abnormal",
            "evaluation_scope": "held-out bearing",
        },
        "evidence_version": "ev1",
        "model_version": "m1",
        "graph_version": "g1",
        "prompt_version": "p1",
    }
    duplicate = client.post(
        "/v1/events",
        json=body,
        headers={**auth("producer-token"), "Idempotency-Key": "event:api-0001"},
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["created"] is False
    assert duplicate.json()["incident"]["id"] == incident_id
    assert (
        client.get(f"/v1/incidents/{incident_id}", headers=auth("producer-token")).status_code
        == 403
    )
    assert (
        client.get(f"/v1/incidents/{incident_id}", headers=auth("analyst-token")).status_code == 200
    )


def test_unknown_artifact_and_reviewer_impersonation_are_rejected(api):
    client, db, artifact_id = api
    body = {
        "evidence_artifact_id": "art_" + "0" * 32,
        "prediction": {
            "score": 1,
            "threshold": 0.5,
            "decision": "abnormal",
            "evaluation_scope": "test",
        },
        "evidence_version": "ev1",
        "model_version": "m1",
        "graph_version": "g1",
        "prompt_version": "p1",
    }
    response = client.post(
        "/v1/events",
        json=body,
        headers={**auth("producer-token"), "Idempotency-Key": "event:unknown1"},
    )
    assert response.status_code == 404
    incident_id = create_incident(client, artifact_id)
    db.save_report(
        incident_id, revision=1, status="ready", content={}, ticket_draft={"title": "Inspect"}
    )
    response = client.post(
        f"/v1/incidents/{incident_id}/approve",
        json={
            "report_revision": 1,
            "action": "create_internal_ticket",
            "reviewer_id": "someone-else",
        },
        headers={**auth("reviewer-token"), "Idempotency-Key": "approval:api1"},
    )
    assert response.status_code == 403


def test_approval_is_bound_to_authenticated_reviewer(api):
    client, db, artifact_id = api
    incident_id = create_incident(client, artifact_id)
    db.save_report(
        incident_id, revision=1, status="ready", content={}, ticket_draft={"title": "Inspect"}
    )
    headers = {**auth("reviewer-token"), "Idempotency-Key": "approval:api2"}
    first = client.post(
        f"/v1/incidents/{incident_id}/approve",
        json={"report_revision": 1, "action": "create_internal_ticket"},
        headers=headers,
    )
    second = client.post(
        f"/v1/incidents/{incident_id}/approve",
        json={"report_revision": 1, "action": "create_internal_ticket"},
        headers=headers,
    )
    assert first.status_code == 200 and first.json()["created"] is True
    assert second.status_code == 200 and second.json()["created"] is False
