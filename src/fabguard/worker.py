"""Single-process worker with database leases and per-incident locks."""

from __future__ import annotations

import argparse
import json
import socket
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from .config import RuntimeSettings
from .graph import (
    AudioEvidence,
    GraphRunContext,
    InvestigationGraph,
    InvestigationInput,
    InvestigationResult,
    QualityStatus,
    TelemetryEvidence,
    build_langgraph,
    postgres_saver_factory,
)
from .llm import LocalTemplateLLM, OpenAICompatibleLLM
from .retrieval import (
    InMemoryRetriever,
    ingest_documents,
    load_reference_corpus,
    postgres_retriever,
)
from .storage import Database, LeaseLost


class InvestigationRunner(Protocol):
    def __call__(
        self,
        incident: Mapping[str, Any],
        investigation: Mapping[str, Any],
        *,
        ensure_owned: Callable[[], None],
    ) -> None: ...


@dataclass(frozen=True)
class WorkerResult:
    outcome: str
    incident_id: str | None = None
    event_id: str | None = None


class InvestigationWorker:
    """Claim one delivery and resume its graph revision synchronously.

    The runner must use the persisted ``thread_id`` and
    ``checkpoint_namespace`` supplied in investigation state. It is
    intentionally not retried inside this class; the event lease is released
    and the next claim resumes through the graph checkpointer.
    """

    def __init__(
        self,
        database: Database,
        runner: InvestigationRunner,
        *,
        worker_id: str | None = None,
        lease_seconds: int = 120,
        max_attempts: int = 3,
    ):
        self.database = database
        self.runner = runner
        self.worker_id = worker_id or f"{socket.gethostname()}:{uuid.uuid4().hex[:12]}"
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts

    def run_once(self) -> WorkerResult:
        event = self.database.claim_event(self.worker_id, lease_seconds=self.lease_seconds)
        if event is None:
            return WorkerResult("idle")
        investigation: Mapping[str, Any] | None = None
        try:
            with self.database.incident_lock(event.incident_id) as lock_connection:
                if lock_connection is None:
                    self.database.fail_event(
                        event.id,
                        self.worker_id,
                        "incident is locked",
                        max_attempts=self.max_attempts,
                    )
                    return WorkerResult("contended", event.incident_id, event.id)

                def ensure_owned() -> None:
                    try:
                        lock_connection.execute(text("SELECT 1")).scalar_one()
                    except SQLAlchemyError as exc:
                        raise LeaseLost("incident lock connection was lost") from exc
                    if not self.database.lease_is_owned(event.id, self.worker_id):
                        raise LeaseLost("event lease is no longer owned")

                ensure_owned()
                incident = self.database.get_incident(event.incident_id)
                investigation = self.database.get_investigation_revision(
                    event.incident_id, event.investigation_revision
                )
                if investigation["status"] == "completed":
                    self.database.finish_event(event.id, self.worker_id)
                    return WorkerResult("completed", event.incident_id, event.id)
                if investigation["status"] == "failed":
                    raise RuntimeError("investigation revision is terminally failed")
                self.database.mark_investigation_running(investigation["id"])
                self.runner(incident, investigation, ensure_owned=ensure_owned)
                completed = self.database.get_investigation_revision(
                    event.incident_id, event.investigation_revision
                )
                if completed["status"] != "completed":
                    raise RuntimeError("runner returned without saving a terminal report")
                # A session-level advisory lock disappears with its connection.
                # Validate that connection before publishing completion.
                ensure_owned()
                self.database.finish_event(event.id, self.worker_id)
            return WorkerResult("completed", event.incident_id, event.id)
        except LeaseLost:
            return WorkerResult("lease_lost", event.incident_id, event.id)
        except Exception as exc:
            try:
                status = self.database.fail_event(
                    event.id, self.worker_id, str(exc), max_attempts=self.max_attempts
                )
                if status == "dead" and investigation is not None:
                    self.database.mark_investigation_failed(investigation["id"])
            except LeaseLost:
                return WorkerResult("lease_lost", event.incident_id, event.id)
            return WorkerResult("failed", event.incident_id, event.id)


class RuntimeGraphRunner:
    """Translate a replay artifact into checkpointed graph state and save its report."""

    def __init__(
        self,
        database: Database,
        graph: object,
        *,
        graph_version: str | None = None,
        prompt_version: str | None = None,
        audio_enabled: bool = True,
    ):
        self.database = database
        self.graph = graph
        self.graph_version = graph_version
        self.prompt_version = prompt_version
        self.audio_enabled = audio_enabled

    def __call__(
        self,
        incident: Mapping[str, Any],
        investigation: Mapping[str, Any],
        *,
        ensure_owned: Callable[[], None] | None = None,
    ) -> None:
        artifact_path = self.database.resolve_artifact(
            incident["evidence_artifact_id"], incident_id=incident["id"]
        )
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        prediction = artifact["prediction"]
        if artifact.get("model_version") != incident["model_version"]:
            raise ValueError("artifact model version does not match the incident envelope")
        if artifact.get("preprocessing_version") != incident["evidence_version"]:
            raise ValueError("artifact evidence version does not match the incident envelope")
        if self.graph_version and investigation["graph_version"] != self.graph_version:
            raise ValueError("pinned graph version is not available in this worker")
        if self.prompt_version and investigation["prompt_version"] != self.prompt_version:
            raise ValueError("pinned prompt version is not available in this worker")
        for field in ("score", "threshold", "decision"):
            if prediction.get(field) != incident["prediction"].get(field):
                raise ValueError(
                    f"artifact prediction {field} does not match the incident envelope"
                )
        if artifact.get("evaluation_scope") != incident["prediction"].get("evaluation_scope"):
            raise ValueError("artifact evaluation scope does not match the incident envelope")
        selected_index = prediction["selected_window_index"]
        if not isinstance(selected_index, int) or not 0 <= selected_index < len(
            artifact["windows"]
        ):
            raise ValueError("selected window index is outside the replay evidence")
        selected = artifact["windows"][selected_index]
        if (
            selected.get("anomaly_score", prediction["score"]) != prediction["score"]
            or selected.get("threshold", prediction["threshold"]) != prediction["threshold"]
        ):
            raise ValueError("selected window does not match the incident prediction")
        vibration_quality = artifact["quality"]["vibration"]
        audio_quality = artifact["quality"]["audio"]
        evidence = InvestigationInput(
            incident_id=incident["id"],
            quality=QualityStatus(
                valid=bool(vibration_quality["is_usable"]),
                reasons=tuple(vibration_quality["issues"]),
            ),
            telemetry=TelemetryEvidence(
                artifact_id=incident["evidence_artifact_id"],
                window_start_seconds=selected["start_sample"] / artifact["sample_rate_hz"],
                window_end_seconds=selected["end_sample"] / artifact["sample_rate_hz"],
                features=selected["vibration_features"],
                preprocessing_version=artifact["preprocessing_version"],
                model_version=artifact["model_version"],
                anomaly_score=prediction["score"],
                threshold=prediction["threshold"],
                abnormal=prediction["decision"] == "abnormal",
                evaluation_scope=artifact["evaluation_scope"],
            ),
            audio=AudioEvidence(
                artifact_id=f"{incident['evidence_artifact_id']}.audio",
                usable=bool(audio_quality["is_usable"]),
                quality_reasons=tuple(audio_quality["issues"]),
                findings=selected["audio_features"],
                preprocessing_version=artifact["preprocessing_version"],
            ),
            audio_policy_enabled=(
                self.audio_enabled and artifact["feature_policy"] in {"audio", "fusion"}
            ),
            applicability=("general", "uored"),
            evidence_version=incident["evidence_version"],
            prompt_version=investigation["prompt_version"],
        )
        configuration = {
            "configurable": {"thread_id": investigation["thread_id"]},
            "recursion_limit": 20,
        }
        saved = self.graph.get_state(configuration)
        revision = int(investigation["revision"])
        if saved.values.get("investigation_revision") == revision:
            if saved.values.get("request") != evidence.model_dump(mode="json"):
                raise ValueError("checkpoint evidence does not match the pinned revision")
            # A saved terminal result may still need its report transaction.
            state = (
                saved.values
                if not saved.next
                else self.graph.invoke(
                    None,
                    config=configuration,
                    context=GraphRunContext(ensure_owned),
                    durability="sync",
                )
            )
        else:
            if saved.values.get("investigation_revision", 0) > revision:
                raise ValueError("checkpoint has advanced beyond this investigation revision")
            # Root checkpoint_ns is reserved for LangGraph subgraphs and is
            # normalized to ''. Keep the stable incident thread, and reset all
            # node-owned outputs only for an explicit fresh revision.
            state = self.graph.invoke(
                {
                    "request": evidence.model_dump(mode="json"),
                    "investigation_revision": revision,
                    "started_at_epoch": time.time(),
                    "initial_passages": [],
                    "refined_passages": [],
                    "decision": {},
                    "report": {},
                    "failure": "",
                    "retrieval_count": 0,
                    "llm_call_count": 0,
                    "trace": [],
                    "result": {},
                },
                config=configuration,
                context=GraphRunContext(ensure_owned),
                durability="sync",
            )
        if not state.get("result"):
            raise RuntimeError("investigation paused before a terminal report")
        result = InvestigationResult.model_validate(state["result"])
        report = result.report.model_dump(mode="json")
        ticket = report.pop("ticket_draft")
        content = {
            **report,
            "trace": list(result.trace),
            "verification": result.verification.model_dump(mode="json"),
            "initial_passages": [
                passage.model_dump(mode="json") for passage in result.initial_passages
            ],
            "refined_passages": [
                passage.model_dump(mode="json") for passage in result.refined_passages
            ],
            "retrieval_count": result.retrieval_count,
            "llm_call_count": result.llm_call_count,
        }
        if ensure_owned:
            ensure_owned()
        self.database.save_report(
            incident["id"],
            revision=int(investigation["revision"]),
            status="ready" if result.status == "ready_for_review" else "insufficient_evidence",
            content=content,
            ticket_draft=ticket if result.status == "ready_for_review" else None,
        )


def _load_retriever(path: str | Path) -> InMemoryRetriever:
    return InMemoryRetriever(ingest_documents(load_reference_corpus(path)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="claim at most one event")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--references", default="references/corpus.json")
    parser.add_argument("--retrieval", choices=("postgres", "memory"), default="postgres")
    parser.add_argument("--adaptive", action="store_true")
    args = parser.parse_args()
    settings = RuntimeSettings()
    if settings.llm_timeout_seconds * 4 > settings.run_deadline_seconds - 10:
        raise RuntimeError("LLM timeout/retry budget exceeds the investigation deadline")
    database = Database(settings.database_url, artifact_root=settings.artifact_root)
    llm = (
        OpenAICompatibleLLM(
            api_key=settings.llm_api_key or "",
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        if settings.llm_provider in {"groq", "openai_compatible"}
        else LocalTemplateLLM()
    )
    if settings.llm_provider != "offline" and not settings.llm_api_key:
        raise RuntimeError("FABGUARD_LLM_API_KEY is required for the configured provider")
    investigation = InvestigationGraph(
        retriever=(
            postgres_retriever(settings.database_url)
            if args.retrieval == "postgres"
            else _load_retriever(args.references)
        ),
        llm=llm,
        adaptive=args.adaptive,
        deadline_seconds=settings.run_deadline_seconds,
    )
    with postgres_saver_factory(settings.checkpoint_url) as saver:
        saver.setup()
        runner = RuntimeGraphRunner(
            database,
            build_langgraph(investigation, checkpointer=saver),
            graph_version=settings.graph_version,
            prompt_version=settings.prompt_version,
            audio_enabled=settings.audio_enabled,
        )
        worker = InvestigationWorker(
            database,
            runner,
            lease_seconds=settings.run_deadline_seconds + 30,
        )
        while True:
            result = worker.run_once()
            print(json.dumps(asdict(result)))
            if args.once:
                return
            if result.outcome == "idle":
                time.sleep(max(0.1, args.poll_seconds))


if __name__ == "__main__":
    main()
