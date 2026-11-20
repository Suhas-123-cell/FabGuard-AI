"""Embedding backends for reference retrieval.

``hash`` is a deterministic token-hashing vector. It captures only shared tokens, not
meaning, and exists so tests and CI run offline. ``bge`` is BAAI/bge-small-en-v1.5
(384 dimensions, matching the ``reference_passages.embedding`` column) run locally through
fastembed. It needs the optional ``bge`` extra and downloads the model on first use.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Literal, Protocol

EmbedderName = Literal["hash", "bge"]
EMBEDDING_DIMENSIONS = 384
BGE_MODEL = "BAAI/bge-small-en-v1.5"
# Recommended by the model card for short search queries against passages.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def hash_embedding(text: str, *, dimensions: int = EMBEDDING_DIMENSIONS) -> list[float]:
    """Signed bag-of-hashed-tokens vector, L2-normalised. Lexical overlap only."""

    values = [0.0] * dimensions
    for token in _TOKEN.findall(text.casefold()):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] & 1 else -1.0
        values[bucket] += sign
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


class HashEmbedder:
    name = "hash"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [hash_embedding(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return hash_embedding(text)


class BgeEmbedder:
    name = "bge"

    def __init__(self, model_name: str = BGE_MODEL, cache_dir: str | None = None) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as error:
            raise RuntimeError(
                "FABGUARD_EMBEDDER=bge needs fastembed; install it with `uv sync --extra bge`"
            ) from error
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for raw in self._model.embed(list(texts)):
            vector = [float(value) for value in raw]
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        if any(len(vector) != EMBEDDING_DIMENSIONS for vector in vectors):
            raise RuntimeError(f"{BGE_MODEL} must produce {EMBEDDING_DIMENSIONS}-dimension vectors")
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([BGE_QUERY_PREFIX + text])[0]


@lru_cache(maxsize=4)
def get_embedder(name: str | None = None) -> Embedder:
    """Return the named backend, defaulting to FABGUARD_EMBEDDER (``hash`` if unset)."""

    chosen = (name or os.getenv("FABGUARD_EMBEDDER") or "hash").strip().lower()
    if chosen == "hash":
        return HashEmbedder()
    if chosen == "bge":
        return BgeEmbedder(cache_dir=os.getenv("FABGUARD_EMBEDDER_CACHE") or None)
    raise ValueError(f"unknown embedder {chosen!r}; expected 'hash' or 'bge'")
