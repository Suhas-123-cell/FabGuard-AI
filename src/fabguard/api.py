"""Narrow FastAPI surface for ingestion, review, and human approval."""

import hmac
import os
from dataclasses import dataclass
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .storage import Conflict, Database, NotFound, StaleRevision, validate_artifact_id, validate_key


@dataclass(frozen=True)
class Principal:
    subject: str
    roles: frozenset[str]


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: float = Field(allow_inf_nan=False)
    threshold: float = Field(allow_inf_nan=False)
    decision: Literal["healthy", "abnormal"]
    evaluation_scope: str = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def consistent_decision(self) -> "PredictionInput":
        expected = "abnormal" if self.score >= self.threshold else "healthy"
        if self.decision != expected:
            raise ValueError("decision must match score >= threshold")
        return self


class EventInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_artifact_id: str = Field(min_length=36, max_length=36)
    prediction: PredictionInput
    evidence_version: str = Field(min_length=1, max_length=100)
    model_version: str = Field(min_length=1, max_length=100)
    graph_version: str = Field(min_length=1, max_length=100)
    prompt_version: str = Field(min_length=1, max_length=100)


class ApprovalInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    report_revision: int = Field(ge=1)
    action: Literal["create_internal_ticket"]
    reviewer_id: str | None = Field(default=None, min_length=1, max_length=100)


def create_app(
    database: Database,
    *,
    api_keys: dict[str, Principal],
    graph_version: str | None = None,
    prompt_version: str | None = None,
) -> FastAPI:
    """Build an app with explicit dependencies; secrets never have defaults."""
    app = FastAPI(title="FabGuard API", version="1.0")

    def get_principal(authorization: Annotated[str | None, Header()] = None) -> Principal:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
        supplied = authorization[7:]
        match = next(
            (
                principal
                for token, principal in api_keys.items()
                if hmac.compare_digest(token, supplied)
            ),
            None,
        )
        if match is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid bearer token")
        return match

    def require(*roles: str):
        def dependency(principal: Annotated[Principal, Depends(get_principal)]) -> Principal:
            if not principal.roles.intersection(roles):
                raise HTTPException(status.HTTP_403_FORBIDDEN, "role is not permitted")
            return principal

        return dependency

    @app.exception_handler(NotFound)
    async def not_found(_: Request, exc: NotFound):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(StaleRevision)
    async def stale(_: Request, exc: StaleRevision):
        return JSONResponse(status_code=409, content={"detail": str(exc), "code": "stale_revision"})

    @app.exception_handler(Conflict)
    async def conflict(_: Request, exc: Conflict):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.post("/v1/events", status_code=201)
    def ingest_event(
        body: EventInput,
        response: Response,
        principal: Annotated[Principal, Depends(require("producer", "admin"))],
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ):
        del principal
        if idempotency_key is None:
            raise HTTPException(400, "Idempotency-Key header is required")
        try:
            validate_key(idempotency_key, name="idempotency key")
            validate_artifact_id(body.evidence_artifact_id)
            # A valid-looking ID is not enough: callers cannot nominate a path
            # or attach an unknown artifact.
            database.resolve_artifact(body.evidence_artifact_id)
            pinned_graph = graph_version or body.graph_version
            pinned_prompt = prompt_version or body.prompt_version
            incident, created = database.ingest_event(
                delivery_key=idempotency_key,
                evidence_artifact_id=body.evidence_artifact_id,
                prediction=body.prediction.model_dump(),
                evidence_version=body.evidence_version,
                model_version=body.model_version,
                graph_version=pinned_graph,
                prompt_version=pinned_prompt,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if not created:
            response.status_code = status.HTTP_200_OK
        return {"created": created, "incident": incident}

    @app.get("/v1/incidents/{incident_id}")
    def get_incident(
        incident_id: str, _: Annotated[Principal, Depends(require("analyst", "reviewer", "admin"))]
    ):
        try:
            return database.get_incident(incident_id)
        except ValueError as exc:
            raise HTTPException(404, "incident not found") from exc

    @app.post("/v1/incidents/{incident_id}/investigations", status_code=202)
    def investigate(
        incident_id: str,
        _: Annotated[Principal, Depends(require("analyst", "admin"))],
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ):
        if idempotency_key is None:
            raise HTTPException(400, "Idempotency-Key header is required")
        try:
            investigation, created = database.request_investigation(
                incident_id, request_key=idempotency_key
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"created": created, "investigation": investigation}

    @app.post("/v1/incidents/{incident_id}/approve")
    def approve(
        incident_id: str,
        body: ApprovalInput,
        principal: Annotated[Principal, Depends(require("reviewer", "admin"))],
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ):
        if idempotency_key is None:
            raise HTTPException(400, "Idempotency-Key header is required")
        if body.reviewer_id is not None and body.reviewer_id != principal.subject:
            raise HTTPException(403, "reviewer must match authenticated identity")
        try:
            ticket, created = database.approve_ticket(
                incident_id,
                report_revision=body.report_revision,
                reviewer_id=principal.subject,
                action=body.action,
                idempotency_key=idempotency_key,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"created": created, "ticket": ticket}

    return app


def create_runtime_app() -> FastAPI:
    """Uvicorn factory using environment-only credentials."""

    database_url = os.environ.get(
        "FABGUARD_DATABASE_URL",
        "postgresql+psycopg://fabguard:fabguard@localhost:5432/fabguard",
    )
    tokens = {
        "producer": os.environ.get("FABGUARD_PRODUCER_TOKEN"),
        "analyst": os.environ.get("FABGUARD_ANALYST_TOKEN"),
        "reviewer": os.environ.get("FABGUARD_REVIEWER_TOKEN"),
    }
    if any(not token or len(token) < 16 for token in tokens.values()):
        raise RuntimeError("distinct producer, analyst, and reviewer tokens are required")
    if len(set(tokens.values())) != 3:
        raise RuntimeError("runtime role tokens must be distinct")
    database = Database(
        database_url,
        artifact_root=os.environ.get("FABGUARD_ARTIFACT_ROOT", "runs/replays"),
    )
    principals = {
        tokens["producer"]: Principal("replay-producer", frozenset({"producer"})),
        tokens["analyst"]: Principal("local-analyst", frozenset({"analyst"})),
        tokens["reviewer"]: Principal(
            os.environ.get("FABGUARD_REVIEWER_SUBJECT", "local-reviewer"),
            frozenset({"reviewer"}),
        ),
    }
    return create_app(
        database,
        api_keys=principals,
        graph_version=os.environ.get("FABGUARD_GRAPH_VERSION", "fabguard-graph-v1"),
        prompt_version=os.environ.get("FABGUARD_PROMPT_VERSION", "fabguard-report-v1"),
    )


def main() -> None:
    import uvicorn

    uvicorn.run("fabguard.api:create_runtime_app", factory=True, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
