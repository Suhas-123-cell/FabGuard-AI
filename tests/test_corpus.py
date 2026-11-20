from __future__ import annotations

import json
from pathlib import Path

from fabguard.retrieval import ingest_documents, load_reference_corpus


def test_corpus_is_large_unique_and_free_of_instruction_shaped_text() -> None:
    documents = load_reference_corpus("references/corpus.json")
    passages = ingest_documents(documents)

    assert len(documents) >= 40
    assert len(passages) == len(documents)  # every note fits in one chunk
    assert not [p.passage_id for p in passages if p.untrusted_directive]


def test_authored_notes_state_their_provenance() -> None:
    raw = json.loads(Path("references/corpus.json").read_text(encoding="utf-8"))
    notes = [item for item in raw if item["source_id"].startswith("note-")]

    assert len(notes) >= 40
    assert all("not copied from" in item["provenance"] for item in notes)
    assert all("url" not in item for item in notes)
