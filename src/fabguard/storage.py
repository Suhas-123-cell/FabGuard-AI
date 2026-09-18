"""Durable incident storage and opaque artifact references.

PostgreSQL is the runtime database. The SQLite compatibility path exists only
so transaction rules can be tested without a running service.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    and_,
    create_engine,
    func,
    insert,
    or_,
    select,
    text,
    update,
)
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import IntegrityError

metadata = MetaData()

incidents = Table(
    "incidents",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("source_event_key", String(200), nullable=False, unique=True),
    Column("evidence_artifact_id", String(80), nullable=False),
    Column("prediction", JSON, nullable=False),
    Column("state", String(24), nullable=False),
    Column("evidence_version", String(100), nullable=False),
    Column("model_version", String(100), nullable=False),
    Column("graph_version", String(100), nullable=False),
    Column("prompt_version", String(100), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)
event_deliveries = Table(
    "event_deliveries",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("delivery_key", String(200), nullable=False, unique=True),
    Column("incident_id", String(36), ForeignKey("incidents.id"), nullable=False),
    Column("investigation_revision", Integer, nullable=False),
    Column("payload_hash", String(64), nullable=False),
    Column("status", String(20), nullable=False),
    Column("lease_owner", String(100)),
    Column("lease_expires_at", DateTime(timezone=True)),
    Column("attempts", Integer, nullable=False),
    Column("last_error", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    ForeignKeyConstraint(
        ["incident_id", "investigation_revision"],
        ["investigation_revisions.incident_id", "investigation_revisions.revision"],
    ),
)
Index("ix_event_claim", event_deliveries.c.status, event_deliveries.c.lease_expires_at)
investigation_revisions = Table(
    "investigation_revisions",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("incident_id", String(36), ForeignKey("incidents.id"), nullable=False),
    Column("revision", Integer, nullable=False),
    Column("request_key", String(200), nullable=False, unique=True),
    Column("thread_id", String(36), nullable=False),
    Column("checkpoint_namespace", String(40), nullable=False),
    Column("status", String(20), nullable=False),
    Column("graph_version", String(100), nullable=False),
    Column("prompt_version", String(100), nullable=False),
    Column("model_version", String(100), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("incident_id", "revision"),
)
reports = Table(
    "reports",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("incident_id", String(36), ForeignKey("incidents.id"), nullable=False),
    Column("revision", Integer, nullable=False),
    Column("status", String(24), nullable=False),
    Column("content", JSON, nullable=False),
    Column("ticket_draft", JSON),
    Column("content_hash", String(64), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("incident_id", "revision"),
    ForeignKeyConstraint(
        ["incident_id", "revision"],
        ["investigation_revisions.incident_id", "investigation_revisions.revision"],
    ),
)
approvals = Table(
    "approvals",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("incident_id", String(36), ForeignKey("incidents.id"), nullable=False),
    Column("report_revision", Integer, nullable=False),
    Column("reviewer_id", String(100), nullable=False),
    Column("action", String(32), nullable=False),
    Column("idempotency_key", String(200), nullable=False, unique=True),
    Column("report_hash", String(64), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("incident_id", "report_revision", "reviewer_id", "action"),
    ForeignKeyConstraint(
        ["incident_id", "report_revision"],
        ["reports.incident_id", "reports.revision"],
    ),
)
tickets = Table(
    "tickets",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("approval_id", String(36), ForeignKey("approvals.id"), nullable=False, unique=True),
    Column("incident_id", String(36), ForeignKey("incidents.id"), nullable=False),
    Column("idempotency_key", String(200), nullable=False, unique=True),
    Column("payload", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
artifact_references = Table(
    "artifact_references",
    metadata,
    Column("id", String(80), primary_key=True),
    Column("artifact_key", String(200), nullable=False, unique=True),
    Column("owner_incident_id", String(36), ForeignKey("incidents.id")),
    Column("relative_path", Text, nullable=False),
    Column("sha256", String(64), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)


class StorageError(RuntimeError):
    """Base class for storage contract failures."""


class NotFound(StorageError):
    pass


class Conflict(StorageError):
    pass


class StaleRevision(Conflict):
    pass


class LeaseLost(Conflict):
    pass


@dataclass(frozen=True)
class ClaimedEvent:
    id: str
    incident_id: str
    delivery_key: str
    lease_owner: str
    attempts: int
    investigation_revision: int


def utcnow() -> datetime:
    return datetime.now(UTC)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row(row: Any | None) -> dict[str, Any] | None:
    return None if row is None else dict(row._mapping)


def _uuid(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError) as exc:
        raise ValueError("invalid UUID") from exc


_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,199}$")
_ARTIFACT_RE = re.compile(r"^art_[a-f0-9]{32}$")


def validate_key(value: str, *, name: str = "key") -> str:
    if not _KEY_RE.fullmatch(value):
        raise ValueError(f"invalid {name}")
    return value


def validate_artifact_id(value: str) -> str:
    if not _ARTIFACT_RE.fullmatch(value):
        raise ValueError("invalid artifact id")
    return value


class Database:
    """Transaction boundary for FabGuard runtime state."""

    def __init__(
        self,
        url: str,
        *,
        artifact_root: str | Path = "runs/artifacts",
        engine: Engine | None = None,
    ):
        self.engine = engine or create_engine(url, pool_pre_ping=True)
        self.artifact_root = Path(artifact_root).resolve()
        self._fallback_locks: dict[str, threading.Lock] = {}
        self._fallback_guard = threading.Lock()

    @property
    def is_postgresql(self) -> bool:
        return self.engine.dialect.name == "postgresql"

    def create_schema_for_tests(self) -> None:
        """Create portable tables. Production startup must run migrations."""
        metadata.create_all(self.engine)

    def ingest_event(
        self,
        *,
        delivery_key: str,
        evidence_artifact_id: str,
        prediction: Mapping[str, Any],
        evidence_version: str,
        model_version: str,
        graph_version: str,
        prompt_version: str,
    ) -> tuple[dict[str, Any], bool]:
        validate_key(delivery_key, name="delivery key")
        validate_artifact_id(evidence_artifact_id)
        now = utcnow()
        payload = {
            "evidence_artifact_id": evidence_artifact_id,
            "prediction": dict(prediction),
            "evidence_version": evidence_version,
            "model_version": model_version,
            "graph_version": graph_version,
            "prompt_version": prompt_version,
        }
        payload_hash = canonical_hash(payload)
        incident_id = str(uuid.uuid4())
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(incidents).values(
                        id=incident_id,
                        source_event_key=delivery_key,
                        state="pending",
                        created_at=now,
                        updated_at=now,
                        **payload,
                    )
                )
                initial_key = f"event:{hashlib.sha256(delivery_key.encode()).hexdigest()}"
                conn.execute(
                    insert(investigation_revisions).values(
                        id=str(uuid.uuid4()),
                        incident_id=incident_id,
                        revision=1,
                        request_key=initial_key,
                        thread_id=incident_id,
                        checkpoint_namespace="revision:1",
                        status="pending",
                        graph_version=graph_version,
                        prompt_version=prompt_version,
                        model_version=model_version,
                        created_at=now,
                        updated_at=now,
                    )
                )
                conn.execute(
                    insert(event_deliveries).values(
                        id=str(uuid.uuid4()),
                        delivery_key=delivery_key,
                        incident_id=incident_id,
                        investigation_revision=1,
                        payload_hash=payload_hash,
                        status="pending",
                        attempts=0,
                        created_at=now,
                        updated_at=now,
                    )
                )
            return self.get_incident(incident_id), True
        except IntegrityError as exc:
            with self.engine.connect() as conn:
                existing = _row(
                    conn.execute(
                        select(incidents).where(incidents.c.source_event_key == delivery_key)
                    ).first()
                )
                event = _row(
                    conn.execute(
                        select(event_deliveries).where(
                            event_deliveries.c.delivery_key == delivery_key
                        )
                    ).first()
                )
            if existing is None or event is None:
                raise
            if event["payload_hash"] != payload_hash:
                raise Conflict("delivery key was already used with a different payload") from exc
            return existing, False

    def get_incident(self, incident_id: str) -> dict[str, Any]:
        incident_id = _uuid(incident_id)
        with self.engine.connect() as conn:
            incident = _row(
                conn.execute(select(incidents).where(incidents.c.id == incident_id)).first()
            )
            if incident is None:
                raise NotFound("incident not found")
            incident["reports"] = [
                _row(r)
                for r in conn.execute(
                    select(reports)
                    .where(reports.c.incident_id == incident_id)
                    .order_by(reports.c.revision)
                )
            ]
            incident["investigations"] = [
                _row(r)
                for r in conn.execute(
                    select(investigation_revisions)
                    .where(investigation_revisions.c.incident_id == incident_id)
                    .order_by(investigation_revisions.c.revision)
                )
            ]
        return incident

    def request_investigation(
        self, incident_id: str, *, request_key: str
    ) -> tuple[dict[str, Any], bool]:
        incident_id = _uuid(incident_id)
        validate_key(request_key, name="request key")
        now = utcnow()
        try:
            with self.engine.begin() as conn:
                incident = _row(
                    conn.execute(
                        select(incidents).where(incidents.c.id == incident_id).with_for_update()
                    ).first()
                )
                if incident is None:
                    raise NotFound("incident not found")
                latest = conn.execute(
                    select(func.max(investigation_revisions.c.revision)).where(
                        investigation_revisions.c.incident_id == incident_id
                    )
                ).scalar_one()
                revision = int(latest or 0) + 1
                values = dict(
                    id=str(uuid.uuid4()),
                    incident_id=incident_id,
                    revision=revision,
                    request_key=request_key,
                    thread_id=incident_id,
                    checkpoint_namespace=f"revision:{revision}",
                    status="pending",
                    graph_version=incident["graph_version"],
                    prompt_version=incident["prompt_version"],
                    model_version=incident["model_version"],
                    created_at=now,
                    updated_at=now,
                )
                conn.execute(insert(investigation_revisions).values(**values))
                event_key = f"investigation:{hashlib.sha256(request_key.encode()).hexdigest()}"
                conn.execute(
                    insert(event_deliveries).values(
                        id=str(uuid.uuid4()),
                        delivery_key=event_key,
                        incident_id=incident_id,
                        investigation_revision=revision,
                        payload_hash=canonical_hash(
                            {"incident_id": incident_id, "revision": revision}
                        ),
                        status="pending",
                        attempts=0,
                        created_at=now,
                        updated_at=now,
                    )
                )
                conn.execute(
                    update(incidents)
                    .where(incidents.c.id == incident_id)
                    .values(state="investigating", updated_at=now)
                )
            return values, True
        except IntegrityError as exc:
            with self.engine.connect() as conn:
                existing = _row(
                    conn.execute(
                        select(investigation_revisions).where(
                            investigation_revisions.c.request_key == request_key
                        )
                    ).first()
                )
            if existing is None:
                raise
            if existing["incident_id"] != incident_id:
                raise Conflict("request key belongs to another incident") from exc
            return existing, False

    def resumable_investigation(
        self, incident_id: str, revision: int | None = None
    ) -> dict[str, Any]:
        """Return an unfinished revision; checkpoints determine its resume node."""
        incident_id = _uuid(incident_id)
        filters = [
            investigation_revisions.c.incident_id == incident_id,
            investigation_revisions.c.status.in_(("pending", "running")),
        ]
        if revision is not None:
            filters.append(investigation_revisions.c.revision == revision)
        with self.engine.connect() as conn:
            result = _row(
                conn.execute(
                    select(investigation_revisions)
                    .where(and_(*filters))
                    .order_by(investigation_revisions.c.revision)
                    .limit(1)
                ).first()
            )
        if result is None:
            raise NotFound("no unfinished investigation")
        return result

    def get_investigation_revision(self, incident_id: str, revision: int) -> dict[str, Any]:
        incident_id = _uuid(incident_id)
        with self.engine.connect() as conn:
            result = _row(
                conn.execute(
                    select(investigation_revisions).where(
                        and_(
                            investigation_revisions.c.incident_id == incident_id,
                            investigation_revisions.c.revision == revision,
                        )
                    )
                ).first()
            )
        if result is None:
            raise NotFound("investigation revision not found")
        return result

    def mark_investigation_running(self, investigation_id: str) -> None:
        with self.engine.begin() as conn:
            result = conn.execute(
                update(investigation_revisions)
                .where(
                    and_(
                        investigation_revisions.c.id == investigation_id,
                        investigation_revisions.c.status.in_(("pending", "running")),
                    )
                )
                .values(status="running", updated_at=utcnow())
            )
            if result.rowcount != 1:
                raise Conflict("investigation is not resumable")

    def mark_investigation_failed(self, investigation_id: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(investigation_revisions)
                .where(investigation_revisions.c.id == investigation_id)
                .values(status="failed", updated_at=utcnow())
            )

    def save_report(
        self,
        incident_id: str,
        *,
        revision: int,
        status: str,
        content: Mapping[str, Any],
        ticket_draft: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        incident_id = _uuid(incident_id)
        if status not in {"ready", "insufficient_evidence"}:
            raise ValueError("invalid terminal report status")
        if revision < 1:
            raise ValueError("revision must be positive")
        if status == "insufficient_evidence" and ticket_draft is not None:
            raise ValueError("insufficient-evidence reports cannot include a ticket draft")
        now = utcnow()
        report_hash = canonical_hash({"content": content, "ticket_draft": ticket_draft})
        values = dict(
            id=str(uuid.uuid4()),
            incident_id=incident_id,
            revision=revision,
            status=status,
            content=dict(content),
            ticket_draft=None if ticket_draft is None else dict(ticket_draft),
            content_hash=report_hash,
            created_at=now,
        )
        try:
            with self.engine.begin() as conn:
                inv = _row(
                    conn.execute(
                        select(investigation_revisions)
                        .where(
                            and_(
                                investigation_revisions.c.incident_id == incident_id,
                                investigation_revisions.c.revision == revision,
                            )
                        )
                        .with_for_update()
                    ).first()
                )
                if inv is None:
                    raise NotFound("investigation revision not found")
                conn.execute(insert(reports).values(**values))
                conn.execute(
                    update(investigation_revisions)
                    .where(investigation_revisions.c.id == inv["id"])
                    .values(status="completed", updated_at=now)
                )
                conn.execute(
                    update(incidents)
                    .where(incidents.c.id == incident_id)
                    .values(state="review", updated_at=now)
                )
            return values
        except IntegrityError as exc:
            with self.engine.connect() as conn:
                existing = _row(
                    conn.execute(
                        select(reports).where(
                            and_(
                                reports.c.incident_id == incident_id, reports.c.revision == revision
                            )
                        )
                    ).first()
                )
            if (
                existing
                and existing["content_hash"] == report_hash
                and existing["status"] == status
            ):
                return existing
            raise Conflict("report revision is immutable") from exc

    def approve_ticket(
        self,
        incident_id: str,
        *,
        report_revision: int,
        reviewer_id: str,
        action: str,
        idempotency_key: str,
    ) -> tuple[dict[str, Any], bool]:
        incident_id = _uuid(incident_id)
        validate_key(idempotency_key, name="idempotency key")
        if action != "create_internal_ticket":
            raise ValueError("unsupported approval action")
        if not reviewer_id or len(reviewer_id) > 100:
            raise ValueError("invalid reviewer")
        now = utcnow()
        try:
            with self.engine.begin() as conn:
                locked_incident = _row(
                    conn.execute(
                        select(incidents).where(incidents.c.id == incident_id).with_for_update()
                    ).first()
                )
                if locked_incident is None:
                    raise NotFound("incident not found")
                existing_approval = _row(
                    conn.execute(
                        select(approvals).where(approvals.c.idempotency_key == idempotency_key)
                    ).first()
                )
                if existing_approval is not None:
                    expected = (incident_id, report_revision, reviewer_id, action)
                    actual = (
                        existing_approval["incident_id"],
                        existing_approval["report_revision"],
                        existing_approval["reviewer_id"],
                        existing_approval["action"],
                    )
                    if actual != expected:
                        raise Conflict("idempotency key was already used for another approval")
                    existing_ticket = _row(
                        conn.execute(
                            select(tickets).where(tickets.c.approval_id == existing_approval["id"])
                        ).first()
                    )
                    if existing_ticket is None:
                        raise Conflict("approval exists without its ticket")
                    return existing_ticket, False
                latest = conn.execute(
                    select(func.max(investigation_revisions.c.revision)).where(
                        investigation_revisions.c.incident_id == incident_id
                    )
                ).scalar_one()
                if latest is None:
                    raise NotFound("report not found")
                if int(latest) != report_revision:
                    raise StaleRevision("only the latest report revision can be approved")
                report = _row(
                    conn.execute(
                        select(reports)
                        .where(
                            and_(
                                reports.c.incident_id == incident_id,
                                reports.c.revision == report_revision,
                            )
                        )
                        .with_for_update()
                    ).first()
                )
                if report is None:
                    raise NotFound("report not found")
                if report["status"] != "ready" or not report["ticket_draft"]:
                    raise Conflict("report has no approvable ticket draft")
                approval_id = str(uuid.uuid4())
                conn.execute(
                    insert(approvals).values(
                        id=approval_id,
                        incident_id=incident_id,
                        report_revision=report_revision,
                        reviewer_id=reviewer_id,
                        action=action,
                        idempotency_key=idempotency_key,
                        report_hash=report["content_hash"],
                        created_at=now,
                    )
                )
                ticket_values = dict(
                    id=str(uuid.uuid4()),
                    approval_id=approval_id,
                    incident_id=incident_id,
                    idempotency_key=f"ticket:{idempotency_key}",
                    payload=report["ticket_draft"],
                    created_at=now,
                )
                conn.execute(insert(tickets).values(**ticket_values))
            return ticket_values, True
        except IntegrityError as exc:
            with self.engine.connect() as conn:
                approval = _row(
                    conn.execute(
                        select(approvals).where(approvals.c.idempotency_key == idempotency_key)
                    ).first()
                )
                ticket = (
                    None
                    if approval is None
                    else _row(
                        conn.execute(
                            select(tickets).where(tickets.c.approval_id == approval["id"])
                        ).first()
                    )
                )
            if approval is None or ticket is None:
                raise Conflict("approval conflicts with an existing approval") from exc
            expected = (incident_id, report_revision, reviewer_id, action)
            actual = (
                approval["incident_id"],
                approval["report_revision"],
                approval["reviewer_id"],
                approval["action"],
            )
            if actual != expected:
                raise Conflict("idempotency key was already used for another approval") from exc
            return ticket, False

    def claim_event(self, owner: str, *, lease_seconds: int = 120) -> ClaimedEvent | None:
        if not owner or len(owner) > 100 or lease_seconds < 1:
            raise ValueError("invalid lease parameters")
        now, expires = utcnow(), utcnow() + timedelta(seconds=lease_seconds)
        with self.engine.begin() as conn:
            stmt = (
                select(event_deliveries)
                .where(
                    or_(
                        event_deliveries.c.status == "pending",
                        and_(
                            event_deliveries.c.status == "leased",
                            event_deliveries.c.lease_expires_at < now,
                        ),
                    )
                )
                .order_by(event_deliveries.c.created_at)
                .limit(1)
            )
            if self.is_postgresql:
                stmt = stmt.with_for_update(skip_locked=True)
            candidate = _row(conn.execute(stmt).first())
            if candidate is None:
                return None
            claimed = conn.execute(
                update(event_deliveries)
                .where(
                    and_(
                        event_deliveries.c.id == candidate["id"],
                        or_(
                            event_deliveries.c.status == "pending",
                            and_(
                                event_deliveries.c.status == "leased",
                                event_deliveries.c.lease_expires_at < now,
                            ),
                        ),
                    )
                )
                .values(
                    status="leased",
                    lease_owner=owner,
                    lease_expires_at=expires,
                    attempts=event_deliveries.c.attempts + 1,
                    updated_at=now,
                )
            )
            if claimed.rowcount != 1:
                return None
            return ClaimedEvent(
                candidate["id"],
                candidate["incident_id"],
                candidate["delivery_key"],
                owner,
                int(candidate["attempts"]) + 1,
                int(candidate["investigation_revision"]),
            )

    def lease_is_owned(self, event_id: str, owner: str) -> bool:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(event_deliveries.c.id).where(
                    and_(
                        event_deliveries.c.id == event_id,
                        event_deliveries.c.status == "leased",
                        event_deliveries.c.lease_owner == owner,
                        event_deliveries.c.lease_expires_at > utcnow(),
                    )
                )
            ).first()
        return row is not None

    def finish_event(self, event_id: str, owner: str) -> None:
        with self.engine.begin() as conn:
            result = conn.execute(
                update(event_deliveries)
                .where(
                    and_(
                        event_deliveries.c.id == event_id,
                        event_deliveries.c.status == "leased",
                        event_deliveries.c.lease_owner == owner,
                        event_deliveries.c.lease_expires_at > utcnow(),
                    )
                )
                .values(
                    status="completed", lease_owner=None, lease_expires_at=None, updated_at=utcnow()
                )
            )
            if result.rowcount != 1:
                raise LeaseLost("event lease is no longer owned")

    def fail_event(self, event_id: str, owner: str, error: str, *, max_attempts: int = 3) -> str:
        with self.engine.begin() as conn:
            current = _row(
                conn.execute(
                    select(event_deliveries)
                    .where(
                        and_(
                            event_deliveries.c.id == event_id,
                            event_deliveries.c.lease_owner == owner,
                        )
                    )
                    .with_for_update()
                ).first()
            )
            if current is None:
                raise LeaseLost("event lease is no longer owned")
            status = "dead" if current["attempts"] >= max_attempts else "pending"
            conn.execute(
                update(event_deliveries)
                .where(event_deliveries.c.id == event_id)
                .values(
                    status=status,
                    lease_owner=None,
                    lease_expires_at=None,
                    last_error=error[:2000],
                    updated_at=utcnow(),
                )
            )
        return status

    @contextmanager
    def incident_lock(self, incident_id: str) -> Iterator[Connection | None]:
        """Hold a session advisory lock for one incident."""
        incident_id = _uuid(incident_id)
        if self.is_postgresql:
            conn = self.engine.connect()
            acquired = bool(
                conn.execute(
                    text("SELECT pg_try_advisory_lock(hashtextextended(:key, 0))"),
                    {"key": incident_id},
                ).scalar_one()
            )
            if not acquired:
                conn.close()
                yield None
                return
            try:
                yield conn
            finally:
                try:
                    conn.execute(
                        text("SELECT pg_advisory_unlock(hashtextextended(:key, 0))"),
                        {"key": incident_id},
                    )
                finally:
                    conn.close()
            return
        with self._fallback_guard:
            lock = self._fallback_locks.setdefault(incident_id, threading.Lock())
        acquired = lock.acquire(blocking=False)
        connection = self.engine.connect() if acquired else None
        try:
            yield connection
        finally:
            if connection is not None:
                connection.close()
            if acquired:
                lock.release()

    def register_artifact(
        self,
        *,
        artifact_key: str,
        path: str | Path,
        sha256: str,
        owner_incident_id: str | None = None,
    ) -> str:
        validate_key(artifact_key, name="artifact key")
        if not re.fullmatch(r"[a-f0-9]{64}", sha256):
            raise ValueError("invalid sha256")
        absolute = Path(path).resolve()
        try:
            absolute.relative_to(self.artifact_root)
        except ValueError as exc:
            raise ValueError("artifact path escapes configured root") from exc
        actual_sha256 = file_sha256(absolute)
        if actual_sha256 != sha256:
            raise ValueError("artifact content does not match its declared sha256")
        object_directory = self.artifact_root / "objects"
        object_directory.mkdir(parents=True, exist_ok=True)
        immutable = object_directory / actual_sha256
        if immutable.exists():
            if file_sha256(immutable) != actual_sha256:
                raise StorageError("content-addressed artifact is corrupted")
        else:
            temporary = object_directory / f".{actual_sha256}.{secrets.token_hex(8)}.tmp"
            shutil.copyfile(absolute, temporary)
            if file_sha256(temporary) != actual_sha256:
                temporary.unlink(missing_ok=True)
                raise StorageError("artifact changed while it was being registered")
            os.chmod(temporary, 0o444)
            os.replace(temporary, immutable)
        relative = immutable.relative_to(self.artifact_root)
        if owner_incident_id is not None:
            owner_incident_id = _uuid(owner_incident_id)
        artifact_id = f"art_{secrets.token_hex(16)}"
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(artifact_references).values(
                        id=artifact_id,
                        artifact_key=artifact_key,
                        owner_incident_id=owner_incident_id,
                        relative_path=str(relative),
                        sha256=sha256,
                        created_at=utcnow(),
                    )
                )
            return artifact_id
        except IntegrityError as exc:
            with self.engine.connect() as conn:
                existing = _row(
                    conn.execute(
                        select(artifact_references).where(
                            artifact_references.c.artifact_key == artifact_key
                        )
                    ).first()
                )
            if (
                existing is None
                or existing["sha256"] != sha256
                or existing["relative_path"] != str(relative)
            ):
                raise Conflict("artifact key was already used for different content") from exc
            return existing["id"]

    def resolve_artifact(self, artifact_id: str, *, incident_id: str | None = None) -> Path:
        validate_artifact_id(artifact_id)
        with self.engine.connect() as conn:
            record = _row(
                conn.execute(
                    select(artifact_references).where(artifact_references.c.id == artifact_id)
                ).first()
            )
        if record is None:
            raise NotFound("artifact not found")
        if record["owner_incident_id"] is not None and record["owner_incident_id"] != incident_id:
            raise NotFound("artifact not found")
        resolved = (self.artifact_root / record["relative_path"]).resolve()
        try:
            resolved.relative_to(self.artifact_root)
        except ValueError as exc:
            raise StorageError("stored artifact path escapes configured root") from exc
        if not resolved.is_file() or file_sha256(resolved) != record["sha256"]:
            raise StorageError("artifact content failed integrity verification")
        return resolved
