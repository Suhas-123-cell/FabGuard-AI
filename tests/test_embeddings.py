from __future__ import annotations

import math
import os
import sys

import pytest

from fabguard.embeddings import (
    EMBEDDING_DIMENSIONS,
    BgeEmbedder,
    HashEmbedder,
    get_embedder,
    hash_embedding,
)


def test_hash_embedding_is_deterministic_unit_length_and_384_dimensional() -> None:
    first = hash_embedding("Inner race fault, BPFI sidebands")
    assert first == hash_embedding("inner race fault bpfi sidebands")
    assert len(first) == EMBEDDING_DIMENSIONS
    assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0, rel_tol=1e-9)


def test_hash_embedding_only_sees_shared_tokens() -> None:
    query = hash_embedding("faulty bearing")
    same_tokens = hash_embedding("bearing faulty")
    synonym = hash_embedding("defective rolling element")

    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert cosine(query, same_tokens) == pytest.approx(1.0)
    assert abs(cosine(query, synonym)) < 0.5


def test_hash_backend_has_matching_query_and_document_vectors() -> None:
    backend = HashEmbedder()
    assert backend.embed_query("kurtosis") == backend.embed_documents(["kurtosis"])[0]


def test_default_backend_is_hash_and_follows_the_environment(monkeypatch) -> None:
    get_embedder.cache_clear()
    monkeypatch.delenv("FABGUARD_EMBEDDER", raising=False)
    assert get_embedder().name == "hash"
    monkeypatch.setenv("FABGUARD_EMBEDDER", "hash")
    assert get_embedder().name == "hash"


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown embedder"):
        get_embedder("word2vec")


def test_bge_backend_explains_the_missing_extra(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "fastembed", None)
    with pytest.raises(RuntimeError, match="uv sync --extra bge"):
        BgeEmbedder()


@pytest.mark.skipif(
    os.getenv("FABGUARD_TEST_BGE") != "1",
    reason="set FABGUARD_TEST_BGE=1 to download and run BAAI/bge-small-en-v1.5",
)
def test_bge_vectors_rank_a_paraphrase_above_an_unrelated_sentence() -> None:
    backend = BgeEmbedder()
    query = backend.embed_query("how often do rollers hit a crack on the stationary ring")
    related, unrelated = backend.embed_documents(
        [
            "Outer-race defect frequency: the rate at which rolling elements pass a spall.",
            "Mixing incompatible greases can soften the thickener.",
        ]
    )
    assert len(query) == EMBEDDING_DIMENSIONS
    assert sum(a * b for a, b in zip(query, related, strict=True)) > sum(
        a * b for a, b in zip(query, unrelated, strict=True)
    )
