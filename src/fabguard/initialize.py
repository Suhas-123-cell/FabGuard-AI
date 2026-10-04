"""Initialize the local PostgreSQL runtime, checkpoints, and reviewed references."""

from __future__ import annotations

import argparse
from pathlib import Path

import psycopg

from .config import RuntimeSettings
from .graph import postgres_saver_factory
from .retrieval import index_reference_corpus


def initialize_runtime(
    settings: RuntimeSettings,
    *,
    migrations: str | Path = "migrations",
    corpus: str | Path = "references/corpus.json",
) -> int:
    scripts = sorted(Path(migrations).glob("[0-9]*.sql"))
    if not scripts:
        raise ValueError("no runtime migrations found; run from the project directory")
    database_url = settings.database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    with psycopg.connect(database_url, autocommit=True, connect_timeout=5) as connection:
        for script in scripts:
            connection.execute(script.read_text(encoding="utf-8"))
    with postgres_saver_factory(settings.checkpoint_url) as saver:
        saver.setup()
    return index_reference_corpus(settings.database_url, corpus)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--migrations", default="migrations")
    parser.add_argument("--corpus", default="references/corpus.json")
    args = parser.parse_args()
    count = initialize_runtime(RuntimeSettings(), migrations=args.migrations, corpus=args.corpus)
    print(f"Runtime initialized; indexed {count} reviewed passages.")


if __name__ == "__main__":
    main()
