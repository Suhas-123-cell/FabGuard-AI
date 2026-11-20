"""Measure reference retrieval against hand-labelled queries.

Runs the real SQL ranking in PostgreSQL/pgvector for each configuration and writes the
metrics and every per-query ranking to a JSON file. The queries and their relevant passage
ids are in ``evals/retrieval/queries.json``; they were written by reading the corpus, by the
same author, so the numbers are indicative of this corpus, not a general benchmark.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import psycopg

from .embeddings import BGE_MODEL
from .retrieval import (
    PostgresRetriever,
    ingest_documents,
    load_reference_corpus,
    postgres_retriever,
)

RECALL_AT = 5
RANKING_DEPTH = 20

# The ranking used before Reciprocal Rank Fusion: raw ts_rank_cd added to raw cosine
# similarity. Kept only so the evaluation can show the difference.
_SCORE_SUM_SQL = """
    SELECT p.source_id, p.passage_id, p.title, p.body AS text, p.section, p.page,
           p.applicability, p.reviewed, p.untrusted_directive,
           (ts_rank_cd(p.search_vector, websearch_to_tsquery('english', %(query)s))
            + (1 - (p.embedding <=> %(embedding)s::vector)))::float8 AS score
      FROM reference_passages p
     WHERE p.reviewed = TRUE AND p.embedder = %(embedder)s
     ORDER BY score DESC, p.passage_id ASC
     LIMIT %(limit)s
"""


@dataclass(frozen=True, slots=True)
class Config:
    name: str
    embedder: str
    mode: str  # a PostgresRetriever mode, or "score_sum"


CONFIGS = (
    Config("fts_all_terms", "hash", "fts_and"),
    Config("fts_any_term", "hash", "fts"),
    Config("hash_vector", "hash", "vector"),
    Config("hash_score_sum", "hash", "score_sum"),
    Config("hash_rrf", "hash", "hybrid"),
    Config("bge_vector", "bge", "vector"),
    Config("bge_score_sum", "bge", "score_sum"),
    Config("bge_rrf", "bge", "hybrid"),
)


def load_queries(path: str | Path, valid_ids: set[str]) -> list[dict]:
    queries = json.loads(Path(path).read_text(encoding="utf-8"))
    for item in queries:
        missing = [pid for pid in item["relevant"] if pid not in valid_ids]
        if missing or not item["relevant"]:
            raise ValueError(f"{item['id']}: unknown or empty relevant ids {missing}")
    return queries


def score_ranking(ranked: Sequence[str], relevant: Sequence[str]) -> dict[str, float]:
    """Recall@5, reciprocal rank (0 when absent from the ranking) and hit@1 for one query."""

    wanted = set(relevant)
    found = len(wanted & set(ranked[:RECALL_AT]))
    reciprocal = next((1 / rank for rank, pid in enumerate(ranked, 1) if pid in wanted), 0.0)
    return {
        "recall_at_5": found / len(wanted),
        "reciprocal_rank": reciprocal,
        "hit_at_1": 1.0 if ranked[:1] and ranked[0] in wanted else 0.0,
    }


def _rank(retriever: PostgresRetriever, config: Config, query: str) -> list[str]:
    if config.mode != "score_sum":
        return [p.passage_id for p in retriever.search(query, limit=RANKING_DEPTH)]
    assert retriever.embed is not None
    rows = retriever.execute(
        _SCORE_SUM_SQL,
        {
            "query": query,
            "embedding": list(retriever.embed(query)),
            "embedder": config.embedder,
            "limit": RANKING_DEPTH,
        },
    )
    return [row["passage_id"] for row in rows]


def _summarise(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        "recall_at_5": round(statistics.fmean(r["recall_at_5"] for r in rows), 4),
        "mrr": round(statistics.fmean(r["reciprocal_rank"] for r in rows), 4),
        "hit_at_1": round(statistics.fmean(r["hit_at_1"] for r in rows), 4),
    }


def run_evaluation(
    database_url: str, corpus: str | Path, queries_path: str | Path, migrations: str | Path
) -> dict:
    passages = ingest_documents(load_reference_corpus(corpus))
    queries = load_queries(queries_path, {p.passage_id for p in passages})
    plain_url = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    with psycopg.connect(plain_url, autocommit=True, connect_timeout=5) as connection:
        for script in sorted(Path(migrations).glob("[0-9]*.sql")):
            connection.execute(script.read_text(encoding="utf-8"))
        version = connection.execute("SHOW server_version").fetchone()[0]

    from .retrieval import index_reference_corpus

    results = []
    for embedder in ("hash", "bge"):
        index_reference_corpus(database_url, corpus, embedder=embedder)
        for config in (c for c in CONFIGS if c.embedder == embedder):
            mode = "hybrid" if config.mode == "score_sum" else config.mode
            retriever = postgres_retriever(database_url, embedder=embedder, mode=mode)
            per_query = []
            for item in queries:
                ranked = _rank(retriever, config, item["query"])
                per_query.append(
                    {
                        "id": item["id"],
                        "style": item["style"],
                        "top5": ranked[:RECALL_AT],
                        **score_ranking(ranked, item["relevant"]),
                    }
                )
            summary = {
                "config": config.name,
                "embedder": config.embedder,
                "mode": config.mode,
                "overall": _summarise(per_query),
                "by_style": {
                    style: _summarise([r for r in per_query if r["style"] == style])
                    for style in sorted({r["style"] for r in per_query})
                },
                "queries": per_query,
            }
            results.append(summary)
    return {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "postgres_version": version,
        "bge_model": BGE_MODEL,
        "passages": len(passages),
        "queries": len(queries),
        "style_counts": {
            style: sum(1 for q in queries if q["style"] == style)
            for style in sorted({q["style"] for q in queries})
        },
        "recall_at": RECALL_AT,
        "ranking_depth": RANKING_DEPTH,
        "results": results,
    }


def to_markdown(report: dict) -> str:
    lines = [
        "# Retrieval evaluation",
        "",
        f"{report['queries']} labelled queries over {report['passages']} passages; "
        f"Recall@{report['recall_at']}, MRR and hit@1 (first result relevant). "
        f"MRR looks at the top {report['ranking_depth']}. Generated {report['generated_utc']} "
        f"on PostgreSQL {report['postgres_version']}; embedding model {report['bge_model']}.",
        "",
        "| Configuration | Recall@5 | MRR | Hit@1 | Recall@5 terminology | Recall@5 descriptive |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["results"]:
        overall, style = item["overall"], item["by_style"]
        lines.append(
            f"| {item['config']} | {overall['recall_at_5']:.3f} | {overall['mrr']:.3f} | "
            f"{overall['hit_at_1']:.3f} | {style['terminology']['recall_at_5']:.3f} | "
            f"{style['descriptive']['recall_at_5']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    from .config import RuntimeSettings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=None, help="a scratch database; it is reindexed")
    parser.add_argument("--corpus", default="references/corpus.json")
    parser.add_argument("--queries", default="evals/retrieval/queries.json")
    parser.add_argument("--migrations", default="migrations")
    parser.add_argument("--output", default="evals/retrieval/results.json")
    args = parser.parse_args()
    url = args.database_url or RuntimeSettings().database_url
    report = run_evaluation(url, args.corpus, args.queries, args.migrations)
    output = Path(args.output)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(to_markdown(report), encoding="utf-8")
    print(to_markdown(report))


if __name__ == "__main__":
    main()
