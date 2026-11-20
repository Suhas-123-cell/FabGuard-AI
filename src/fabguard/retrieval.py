"""Reference retrieval with explicit provenance.

Two implementations share one ``search`` contract. ``InMemoryRetriever`` is a lexical
BM25 ranking for tests and offline use. ``PostgresRetriever`` fuses a PostgreSQL
full-text ranking with a pgvector cosine ranking using Reciprocal Rank Fusion.
Retrieved text is always data: callers must never concatenate it into a system
message or execute directives found in it.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .embeddings import Embedder, get_embedder, hash_embedding

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SAFE_TAG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_TOKEN = re.compile(r"[a-z0-9]+")
_INJECTION_MARKERS = (
    "ignore previous",
    "ignore all previous",
    "system prompt",
    "developer message",
    "follow these instructions",
    "you are chatgpt",
    "tool call",
    "ignore all rules",
    "system:",
    "reveal api",
    "api key",
    "approve a",
    "create a ticket",
)


class Passage(BaseModel):
    """A reviewed reference fragment returned to the investigation graph."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    passage_id: str
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1, max_length=20_000)
    section: str = Field(min_length=1, max_length=300)
    page: str | None = Field(default=None, max_length=80)
    applicability: tuple[str, ...] = ()
    reviewed: bool = True
    untrusted_directive: bool = False
    score: float = 0.0

    @field_validator("source_id", "passage_id")
    @classmethod
    def safe_identifier(cls, value: str) -> str:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError("identifier must be opaque and path-free")
        return value


class ReferenceDocument(BaseModel):
    """Manually reviewed source material before chunking."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    title: str = Field(min_length=1, max_length=300)
    section: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1, max_length=200_000)
    page: str | None = Field(default=None, max_length=80)
    applicability: tuple[str, ...] = ()
    reviewed: bool = True

    @field_validator("source_id")
    @classmethod
    def safe_source_id(cls, value: str) -> str:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError("source_id must be opaque and path-free")
        return value

    @field_validator("applicability")
    @classmethod
    def safe_applicability(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not _SAFE_TAG.fullmatch(item) for item in value):
            raise ValueError("applicability must contain safe, explicit tags")
        return value


class Retriever(Protocol):
    def search(
        self,
        query: str,
        *,
        limit: int = 4,
        applicability: Sequence[str] = (),
    ) -> list[Passage]: ...


def contains_untrusted_directive(text: str) -> bool:
    """Flag obvious instruction-shaped text without pretending to sanitize it."""

    folded = " ".join(text.casefold().split())
    return any(marker in folded for marker in _INJECTION_MARKERS)


def chunk_document(
    document: ReferenceDocument,
    *,
    max_chars: int = 1_800,
    overlap_chars: int = 180,
) -> list[Passage]:
    """Split on paragraph boundaries while retaining source provenance."""

    if max_chars < 200:
        raise ValueError("max_chars must be at least 200")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be non-negative and below max_chars")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", document.text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            step = max_chars - overlap_chars
            chunks.extend(paragraph[i : i + max_chars] for i in range(0, len(paragraph), step))
        elif not current:
            current = paragraph
        elif len(current) + 2 + len(paragraph) <= max_chars:
            current += "\n\n" + paragraph
        else:
            chunks.append(current)
            available = max(0, max_chars - len(paragraph) - 2)
            retained = min(overlap_chars, available)
            prefix = current[-retained:] if retained else ""
            current = (prefix + "\n\n" + paragraph).strip()
    if current:
        chunks.append(current)

    return [
        Passage(
            source_id=document.source_id,
            passage_id=f"{document.source_id}:p{index}",
            title=document.title,
            text=text,
            section=document.section,
            page=document.page,
            applicability=document.applicability,
            reviewed=document.reviewed,
            untrusted_directive=contains_untrusted_directive(text),
        )
        for index, text in enumerate(chunks, start=1)
    ]


def ingest_documents(documents: Iterable[ReferenceDocument]) -> list[Passage]:
    passages: list[Passage] = []
    seen: set[str] = set()
    for document in documents:
        if document.source_id in seen:
            raise ValueError(f"duplicate source_id: {document.source_id}")
        seen.add(document.source_id)
        passages.extend(chunk_document(document))
    return passages


def load_reference_corpus(path: str | Path) -> list[ReferenceDocument]:
    """Load the reviewed corpus without widening its declared applicability."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("reference corpus must be a non-empty JSON array")
    return [
        ReferenceDocument(
            source_id=item["source_id"],
            title=item["title"],
            section=item["section"],
            text=item["text"],
            page=item.get("page"),
            applicability=tuple(item["applicability"]),
            reviewed=item.get("reviewed", True),
        )
        for item in payload
    ]


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


@dataclass(slots=True)
class InMemoryRetriever:
    """Stable BM25-style lexical ranking for tests and local/offline use."""

    passages: Sequence[Passage]
    k1: float = 1.2
    b: float = 0.75
    _document_frequency: Counter[str] = field(init=False, repr=False)
    _lengths: list[int] = field(init=False, repr=False)
    _average_length: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.passages:
            raise ValueError("retriever needs at least one passage")
        ids = [passage.passage_id for passage in self.passages]
        if len(ids) != len(set(ids)):
            raise ValueError("passage_id values must be unique")
        self._lengths = [max(1, len(_tokens(p.text))) for p in self.passages]
        self._average_length = sum(self._lengths) / len(self._lengths)
        self._document_frequency = Counter()
        for passage in self.passages:
            text = f"{passage.title} {passage.section} {passage.text}"
            self._document_frequency.update(set(_tokens(text)))

    def search(
        self,
        query: str,
        *,
        limit: int = 4,
        applicability: Sequence[str] = (),
    ) -> list[Passage]:
        if not query.strip():
            raise ValueError("query cannot be blank")
        if limit < 1 or limit > 20:
            raise ValueError("limit must be between 1 and 20")
        query_tokens = Counter(_tokens(query))
        if not query_tokens:
            return []
        required = {item.casefold() for item in applicability}
        total = len(self.passages)
        ranked: list[tuple[float, str, Passage]] = []
        for index, passage in enumerate(self.passages):
            if not passage.reviewed:
                continue
            tags = {item.casefold() for item in passage.applicability}
            if required and not (required & tags or "general" in tags):
                continue
            terms = Counter(_tokens(f"{passage.title} {passage.section} {passage.text}"))
            score = 0.0
            for term, query_weight in query_tokens.items():
                frequency = terms[term]
                if not frequency:
                    continue
                df = self._document_frequency[term]
                inverse_frequency = math.log(1 + (total - df + 0.5) / (df + 0.5))
                norm = frequency + self.k1 * (
                    1 - self.b + self.b * self._lengths[index] / self._average_length
                )
                score += query_weight * inverse_frequency * frequency * (self.k1 + 1) / norm
            if score > 0:
                ranked.append((score, passage.passage_id, passage))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [item[2].model_copy(update={"score": item[0]}) for item in ranked[:limit]]


SearchMode = Literal["hybrid", "fts", "fts_and", "vector"]
RRF_K = 60
RRF_CANDIDATES = 50

_ELIGIBLE = """
    eligible AS (
        SELECT * FROM reference_passages
         WHERE reviewed = TRUE
           AND (%(applicability)s::text[] = '{}' OR applicability && %(applicability)s::text[]
                OR applicability @> ARRAY['general']::text[])
    )"""
# websearch_to_tsquery ANDs every term, so a natural-language question rarely matches.
# Turning the ANDs into ORs keeps ts_rank_cd ordering while allowing partial matches.
_TSQUERY_OR = "replace(websearch_to_tsquery('english', %(query)s)::text, ' & ', ' | ')::tsquery"
_TSQUERY_AND = "websearch_to_tsquery('english', %(query)s)"
_FTS_BRANCH = """
    fts AS (
        SELECT passage_id,
               row_number() OVER (
                   ORDER BY ts_rank_cd(search_vector, {tsquery}) DESC, passage_id
               ) AS rnk
          FROM eligible
         WHERE search_vector @@ {tsquery}
         ORDER BY rnk
         LIMIT %(candidates)s
    )"""
_VECTOR_BRANCH = """
    vec AS (
        SELECT passage_id,
               row_number() OVER (
                   ORDER BY embedding <=> %(embedding)s::vector, passage_id
               ) AS rnk
          FROM eligible
         WHERE embedding IS NOT NULL AND embedder = %(embedder)s
         ORDER BY rnk
         LIMIT %(candidates)s
    )"""
_COLUMNS = """p.source_id, p.passage_id, p.title, p.body AS text, p.section, p.page,
           p.applicability, p.reviewed, p.untrusted_directive"""


def build_search_sql(mode: SearchMode) -> str:
    """Compose the ranking query from fixed fragments; user input is only ever a parameter."""

    if mode == "hybrid":
        ctes = [_ELIGIBLE, _FTS_BRANCH.format(tsquery=_TSQUERY_OR), _VECTOR_BRANCH]
        # Reciprocal Rank Fusion: sum of 1 / (k + rank) over the rankings that contain the
        # passage. Scores from the two rankings are never mixed, only their positions.
        score = (
            "COALESCE(1.0 / (%(rrf_k)s + fts.rnk), 0) + COALESCE(1.0 / (%(rrf_k)s + vec.rnk), 0)"
        )
        source = "fts FULL OUTER JOIN vec USING (passage_id)"
    elif mode in ("fts", "fts_and"):
        tsquery = _TSQUERY_OR if mode == "fts" else _TSQUERY_AND
        ctes = [_ELIGIBLE, _FTS_BRANCH.format(tsquery=tsquery)]
        score = "1.0 / (%(rrf_k)s + fts.rnk)"
        source = "fts"
    else:
        ctes = [_ELIGIBLE, _VECTOR_BRANCH]
        score = "1.0 / (%(rrf_k)s + vec.rnk)"
        source = "vec"
    return (
        "WITH"
        + ",".join(ctes)
        + f"""
    SELECT {_COLUMNS}, ({score})::float8 AS score
      FROM {source}
      JOIN reference_passages p USING (passage_id)
     ORDER BY score DESC, p.passage_id ASC
     LIMIT %(limit)s
"""
    )


@dataclass(slots=True)
class PostgresRetriever:
    """PostgreSQL retrieval behind injected SQL/embedding boundaries.

    ``hybrid`` (default) ranks by full-text and by vector similarity separately and fuses
    the two rankings with Reciprocal Rank Fusion. Without an ``embed`` function it falls
    back to full-text only. ``score`` is the fused RRF value, not a probability.
    """

    execute: Callable[[str, Mapping[str, object]], Iterable[Mapping[str, object]]]
    embed: Callable[[str], Sequence[float]] | None = None
    embedder_name: str = "hash"
    mode: SearchMode = "hybrid"

    def search(
        self,
        query: str,
        *,
        limit: int = 4,
        applicability: Sequence[str] = (),
    ) -> list[Passage]:
        if not query.strip():
            raise ValueError("query cannot be blank")
        if limit < 1 or limit > 20:
            raise ValueError("limit must be between 1 and 20")
        mode: SearchMode = self.mode
        if self.embed is None:
            if mode == "vector":
                raise ValueError("vector search needs an embedding function")
            if mode == "hybrid":
                mode = "fts"
        embedding = list(self.embed(query)) if self.embed and mode in ("hybrid", "vector") else None
        parameters: dict[str, object] = {
            "query": query,
            "applicability": list(applicability),
            "limit": limit,
            "candidates": RRF_CANDIDATES,
            "rrf_k": RRF_K,
        }
        if embedding is not None:
            parameters["embedding"] = embedding
            parameters["embedder"] = self.embedder_name
        rows = self.execute(build_search_sql(mode), parameters)
        return [Passage.model_validate(dict(row)) for row in rows]


def _resolve_embedder(name: str | None) -> Embedder:
    if name is None:
        from .config import RuntimeSettings

        name = RuntimeSettings().embedder
    return get_embedder(name)


def _psycopg_url(url: str) -> str:
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


def postgres_retriever(
    database_url: str,
    *,
    embedder: str | None = None,
    mode: SearchMode = "hybrid",
) -> PostgresRetriever:
    """Create a small connection-per-query PostgreSQL/pgvector retriever.

    ``embedder`` defaults to FABGUARD_EMBEDDER and must match the one used at index time.
    """

    backend = _resolve_embedder(embedder)

    def execute(sql: str, parameters: Mapping[str, object]) -> Iterable[Mapping[str, object]]:
        import psycopg
        from pgvector.psycopg import register_vector
        from psycopg.rows import dict_row

        with psycopg.connect(
            _psycopg_url(database_url), connect_timeout=5, row_factory=dict_row
        ) as connection:
            register_vector(connection)
            with connection.cursor() as cursor:
                cursor.execute("SET LOCAL statement_timeout = 10000")
                adapted = dict(parameters)
                if adapted.get("embedding") is not None:
                    adapted["embedding"] = np.asarray(adapted["embedding"], dtype=np.float32)
                cursor.execute(sql, adapted)
                return list(cursor.fetchall())

    return PostgresRetriever(
        execute=execute, embed=backend.embed_query, embedder_name=backend.name, mode=mode
    )


def index_reference_corpus(
    database_url: str, corpus_path: str | Path, *, embedder: str | None = None
) -> int:
    """Upsert the reviewed corpus with embeddings from the chosen backend.

    ``embedder`` defaults to FABGUARD_EMBEDDER. Re-run indexing after changing it.
    """

    import psycopg
    from pgvector.psycopg import register_vector

    backend = _resolve_embedder(embedder)
    documents = load_reference_corpus(corpus_path)
    passages = ingest_documents(documents)
    vectors = backend.embed_documents(
        [f"{passage.title} {passage.section} {passage.text}" for passage in passages]
    )
    sql = """
        INSERT INTO reference_passages (
            passage_id, source_id, title, body, section, page, applicability,
            reviewed, untrusted_directive, embedding, embedder
        ) VALUES (
            %(passage_id)s, %(source_id)s, %(title)s, %(body)s, %(section)s,
            %(page)s, %(applicability)s, %(reviewed)s, %(untrusted_directive)s, %(embedding)s,
            %(embedder)s
        )
        ON CONFLICT (passage_id) DO UPDATE SET
            source_id = EXCLUDED.source_id, title = EXCLUDED.title, body = EXCLUDED.body,
            section = EXCLUDED.section, page = EXCLUDED.page,
            applicability = EXCLUDED.applicability, reviewed = EXCLUDED.reviewed,
            untrusted_directive = EXCLUDED.untrusted_directive, embedding = EXCLUDED.embedding,
            embedder = EXCLUDED.embedder
    """
    with psycopg.connect(_psycopg_url(database_url), connect_timeout=5) as connection:
        register_vector(connection)
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL statement_timeout = 30000")
            for passage, vector in zip(passages, vectors, strict=True):
                cursor.execute(
                    sql,
                    {
                        "passage_id": passage.passage_id,
                        "source_id": passage.source_id,
                        "title": passage.title,
                        "body": passage.text,
                        "section": passage.section,
                        "page": passage.page,
                        "applicability": list(passage.applicability),
                        "reviewed": passage.reviewed,
                        "untrusted_directive": passage.untrusted_directive,
                        "embedding": np.asarray(vector, dtype=np.float32),
                        "embedder": backend.name,
                    },
                )
            passage_ids = [passage.passage_id for passage in passages]
            cursor.execute(
                "DELETE FROM reference_passages WHERE NOT (passage_id = ANY(%s))",
                (passage_ids,),
            )
    return len(passages)


def main() -> None:
    from .config import RuntimeSettings

    parser = argparse.ArgumentParser(description="Index reviewed FabGuard references")
    parser.add_argument("--corpus", default="references/corpus.json")
    parser.add_argument("--embedder", choices=("hash", "bge"), help="default: FABGUARD_EMBEDDER")
    args = parser.parse_args()
    settings = RuntimeSettings()
    count = index_reference_corpus(settings.database_url, args.corpus, embedder=args.embedder)
    print(f"indexed {count} reviewed passages")


__all__ = [
    "InMemoryRetriever",
    "Passage",
    "PostgresRetriever",
    "RRF_K",
    "ReferenceDocument",
    "Retriever",
    "build_search_sql",
    "chunk_document",
    "contains_untrusted_directive",
    "ingest_documents",
    "hash_embedding",
    "index_reference_corpus",
    "load_reference_corpus",
    "postgres_retriever",
]


if __name__ == "__main__":
    main()
