-- A report revision yields at most one ticket, whichever reviewer approves it first.
CREATE UNIQUE INDEX IF NOT EXISTS approvals_one_per_revision
    ON approvals (incident_id, report_revision, action);
