BEGIN;

-- Records which embedding backend produced each stored vector, so a query is never
-- compared against vectors from a different model.
ALTER TABLE reference_passages
    ADD COLUMN IF NOT EXISTS embedder varchar(32) NOT NULL DEFAULT 'hash';

COMMIT;
