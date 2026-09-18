BEGIN;

CREATE TABLE IF NOT EXISTS reference_passages (
    passage_id varchar(128) PRIMARY KEY,
    source_id varchar(128) NOT NULL,
    title varchar(300) NOT NULL,
    body text NOT NULL,
    section varchar(300) NOT NULL,
    page varchar(80),
    applicability text[] NOT NULL DEFAULT '{}',
    reviewed boolean NOT NULL DEFAULT false,
    untrusted_directive boolean NOT NULL DEFAULT false,
    embedding vector(384),
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(title, '') || ' ' || coalesce(section, '') || ' ' || coalesce(body, ''))
    ) STORED
);

CREATE INDEX IF NOT EXISTS ix_reference_passages_search
    ON reference_passages USING gin(search_vector);
CREATE INDEX IF NOT EXISTS ix_reference_passages_embedding
    ON reference_passages USING hnsw (embedding vector_cosine_ops);

COMMIT;
