from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from sqlalchemy import text

from fabguard.graph import InvestigationGraph, build_langgraph, postgres_saver_factory
from fabguard.llm import LocalTemplateLLM
from fabguard.replay import register_and_submit
from fabguard.retrieval import postgres_retriever
from fabguard.worker import InvestigationWorker, RuntimeGraphRunner


def submit(replay, settings, url):
    return register_and_submit(
        replay,
        database_url=settings.database_url,
        api_url=url,
        api_token=settings.producer_token,
        artifact_root=settings.artifact_root,
    )


@pytest.mark.parametrize("adaptive", [False, True])
def test_real_http_replay_to_review_and_idempotent_ticket(live_api, replay_fixture, adaptive):
    url, settings, db = live_api
    first = submit(replay_fixture, settings, url)
    duplicate = submit(replay_fixture, settings, url)
    incident_id = first["incident"]["id"]
    assert first["created"] and not duplicate["created"]
    assert duplicate["incident"]["id"] == incident_id
    llm = LocalTemplateLLM()
    with postgres_saver_factory(settings.checkpoint_url) as saver:
        graph = build_langgraph(
            InvestigationGraph(postgres_retriever(settings.database_url), llm, adaptive=adaptive),
            checkpointer=saver,
        )
        worker = InvestigationWorker(db, RuntimeGraphRunner(db, graph))
        assert worker.run_once().outcome == "completed"
        assert worker.run_once().outcome == "idle"
        reviewer = {"Authorization": f"Bearer {settings.reviewer_token}"}
        producer = {"Authorization": f"Bearer {settings.producer_token}"}
        endpoint = f"{url}/v1/incidents/{incident_id}"
        assert httpx.get(endpoint, headers=producer).status_code == 403
        report = httpx.get(endpoint, headers=reviewer).json()["reports"][0]
        assert report["status"] == "ready"
        assert report["content"]["verification"]["passed"]
        assert report["content"]["initial_passages"]
        body = {"report_revision": 1, "action": "create_internal_ticket"}
        assert httpx.post(endpoint + "/approve", headers=producer, json=body).status_code == 403
        headers = {**reviewer, "Idempotency-Key": "approval:e2e-fixture"}
        approval = httpx.post(endpoint + "/approve", headers=headers, json=body)
        repeated = httpx.post(endpoint + "/approve", headers=headers, json=body)
        assert approval.status_code == repeated.status_code == 200
        assert approval.json()["created"] and not repeated.json()["created"]
        assert approval.json()["ticket"]["id"] == repeated.json()["ticket"]["id"]
        analyst = {"Authorization": f"Bearer {settings.analyst_token}"}
        revision = httpx.post(
            endpoint + "/investigations",
            headers={
                **analyst,
                "Idempotency-Key": "revision:e2e-fixture",
            },
        )
        assert revision.status_code == 202
        stale = httpx.post(
            endpoint + "/approve",
            headers={
                **reviewer,
                "Idempotency-Key": "approval:e2e-stale",
            },
            json=body,
        )
        assert stale.status_code == 409 and stale.json()["code"] == "stale_revision"
        assert worker.run_once().outcome == "completed"
        refreshed = httpx.get(endpoint, headers=reviewer).json()
        assert [item["revision"] for item in refreshed["reports"]] == [1, 2]
        assert len(refreshed["tickets"]) == 1
        assert llm.usage.logical_calls == 2


@pytest.mark.parametrize("crash_point", ["initial_retrieval", "planner", "inflight"])
def test_process_crash_resumes_postgres_worker(live_api, replay_fixture, crash_point, tmp_path):
    url, settings, db = live_api
    incident_id = submit(replay_fixture, settings, url)["incident"]["id"]
    log = tmp_path / "calls.jsonl"
    environment = {
        **os.environ,
        "FABGUARD_DATABASE_URL": settings.database_url,
        "FABGUARD_CHECKPOINT_DATABASE_URL": settings.checkpoint_url,
        "FABGUARD_ARTIFACT_ROOT": str(settings.artifact_root),
        "FABGUARD_CRASH_POINT": crash_point,
        "FABGUARD_TEST_CALL_LOG": str(log),
    }
    helper = Path(__file__).parent / "support/crash_worker.py"
    crashed = subprocess.run(
        [sys.executable, str(helper)], env=environment, timeout=30, capture_output=True, text=True
    )
    assert crashed.returncode == 17, crashed.stderr
    assert not db.get_incident(incident_id)["reports"]
    with db.engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE event_deliveries SET lease_expires_at = now() - interval '1 second' "
                "WHERE incident_id = :incident"
            ),
            {"incident": incident_id},
        )
    environment["FABGUARD_CRASH_POINT"] = ""
    resumed = subprocess.run(
        [sys.executable, str(helper)], env=environment, timeout=30, capture_output=True, text=True
    )
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(resumed.stdout)["outcome"] == "completed"
    calls = [json.loads(line)["call"] for line in log.read_text().splitlines()]
    assert calls.count("retrieval") == 1
    # A provider-accepted call that never reached a checkpoint can repeat.
    assert calls.count("llm") == (2 if crash_point == "inflight" else 1)
    report = db.get_incident(incident_id)["reports"][0]
    assert report["status"] == "ready"
    assert report["content"]["retrieval_count"] == report["content"]["llm_call_count"] == 1
