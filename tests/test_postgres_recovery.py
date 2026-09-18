from __future__ import annotations

import os
from uuid import uuid4

import pytest

from fabguard.graph import (
    InvestigationGraph,
    InvestigationInput,
    QualityStatus,
    TelemetryEvidence,
    build_langgraph,
    postgres_saver_factory,
)
from fabguard.llm import LocalTemplateLLM
from fabguard.retrieval import InMemoryRetriever, Passage


@pytest.mark.skipif(
    not os.getenv("FABGUARD_TEST_POSTGRES_URL"),
    reason="set FABGUARD_TEST_POSTGRES_URL to run PostgreSQL checkpoint recovery coverage",
)
def test_postgres_checkpoint_resumes_after_saved_retrieval():
    evidence = InvestigationInput(
        incident_id=uuid4(),
        quality=QualityStatus(valid=True),
        telemetry=TelemetryEvidence(
            artifact_id="artifact.recovery",
            window_start_seconds=0,
            window_end_seconds=1,
            features={"rms": 1.5},
            preprocessing_version="features-v1",
            model_version="model-v1",
            anomaly_score=0.8,
            threshold=0.6,
            abnormal=True,
            evaluation_scope="workflow recovery test",
        ),
        initial_query="bearing vibration inspection guidance",
    )
    passage = Passage(
        source_id="guide",
        passage_id="guide:p1",
        title="Bearing inspection guidance",
        section="Inspection",
        text="Inspect bearing mounting and lubrication after elevated vibration.",
        applicability=("general",),
    )
    configuration = {
        "configurable": {"thread_id": str(evidence.incident_id), "checkpoint_ns": "recovery"},
        "recursion_limit": 20,
    }
    with postgres_saver_factory(os.environ["FABGUARD_TEST_POSTGRES_URL"]) as saver:
        saver.setup()
        graph = build_langgraph(
            InvestigationGraph(InMemoryRetriever([passage]), LocalTemplateLLM()),
            checkpointer=saver,
            interrupt_after=("initial_retrieval",),
        )
        paused = graph.invoke(
            {"request": evidence.model_dump(mode="json")}, config=configuration, durability="sync"
        )
        assert paused["trace"] == ["quality", "telemetry", "audio", "initial_retrieval"]

        completed = graph.invoke(None, config=configuration, durability="sync")

    assert completed["trace"].count("initial_retrieval") == 1
    assert completed["trace"][-2:] == ["verifier", "review"]
    assert completed["result"]["retrieval_count"] == 1
    assert completed["result"]["llm_call_count"] == 1
