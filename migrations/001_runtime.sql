BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS incidents (
    id varchar(36) PRIMARY KEY,
    source_event_key varchar(200) NOT NULL UNIQUE,
    evidence_artifact_id varchar(80) NOT NULL,
    prediction jsonb NOT NULL,
    state varchar(24) NOT NULL CHECK (state IN ('pending', 'investigating', 'review', 'failed')),
    evidence_version varchar(100) NOT NULL,
    model_version varchar(100) NOT NULL,
    graph_version varchar(100) NOT NULL,
    prompt_version varchar(100) NOT NULL,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS event_deliveries (
    id varchar(36) PRIMARY KEY,
    delivery_key varchar(200) NOT NULL UNIQUE,
    incident_id varchar(36) NOT NULL REFERENCES incidents(id),
    investigation_revision integer NOT NULL CHECK (investigation_revision > 0),
    payload_hash varchar(64) NOT NULL,
    status varchar(20) NOT NULL CHECK (status IN ('pending', 'leased', 'completed', 'dead')),
    lease_owner varchar(100),
    lease_expires_at timestamptz,
    attempts integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    last_error text,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    CHECK ((status = 'leased') = (lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS ix_event_claim ON event_deliveries(status, lease_expires_at);

CREATE TABLE IF NOT EXISTS investigation_revisions (
    id varchar(36) PRIMARY KEY,
    incident_id varchar(36) NOT NULL REFERENCES incidents(id),
    revision integer NOT NULL CHECK (revision > 0),
    request_key varchar(200) NOT NULL UNIQUE,
    thread_id varchar(36) NOT NULL,
    checkpoint_namespace varchar(40) NOT NULL,
    status varchar(20) NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    graph_version varchar(100) NOT NULL,
    prompt_version varchar(100) NOT NULL,
    model_version varchar(100) NOT NULL,
    created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL,
    UNIQUE (incident_id, revision)
);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'event_investigation_fk') THEN
        ALTER TABLE event_deliveries ADD CONSTRAINT event_investigation_fk
            FOREIGN KEY (incident_id, investigation_revision)
            REFERENCES investigation_revisions(incident_id, revision);
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS reports (
    id varchar(36) PRIMARY KEY,
    incident_id varchar(36) NOT NULL REFERENCES incidents(id),
    revision integer NOT NULL CHECK (revision > 0),
    status varchar(24) NOT NULL CHECK (status IN ('ready', 'insufficient_evidence')),
    content jsonb NOT NULL,
    ticket_draft jsonb,
    content_hash varchar(64) NOT NULL,
    created_at timestamptz NOT NULL,
    UNIQUE (incident_id, revision),
    FOREIGN KEY (incident_id, revision) REFERENCES investigation_revisions(incident_id, revision)
);

CREATE TABLE IF NOT EXISTS approvals (
    id varchar(36) PRIMARY KEY,
    incident_id varchar(36) NOT NULL REFERENCES incidents(id),
    report_revision integer NOT NULL,
    reviewer_id varchar(100) NOT NULL,
    action varchar(32) NOT NULL CHECK (action = 'create_internal_ticket'),
    idempotency_key varchar(200) NOT NULL UNIQUE,
    report_hash varchar(64) NOT NULL,
    created_at timestamptz NOT NULL,
    UNIQUE (incident_id, report_revision, reviewer_id, action),
    FOREIGN KEY (incident_id, report_revision) REFERENCES reports(incident_id, revision)
);

CREATE TABLE IF NOT EXISTS tickets (
    id varchar(36) PRIMARY KEY,
    approval_id varchar(36) NOT NULL UNIQUE REFERENCES approvals(id),
    incident_id varchar(36) NOT NULL REFERENCES incidents(id),
    idempotency_key varchar(200) NOT NULL UNIQUE,
    payload jsonb NOT NULL,
    created_at timestamptz NOT NULL
);

CREATE TABLE IF NOT EXISTS artifact_references (
    id varchar(80) PRIMARY KEY CHECK (id ~ '^art_[a-f0-9]{32}$'),
    artifact_key varchar(200) NOT NULL UNIQUE,
    owner_incident_id varchar(36) REFERENCES incidents(id),
    relative_path text NOT NULL CHECK (relative_path !~ '(^|/)\.\.(/|$)' AND relative_path !~ '^/'),
    sha256 varchar(64) NOT NULL CHECK (sha256 ~ '^[a-f0-9]{64}$'),
    created_at timestamptz NOT NULL
);

COMMIT;
