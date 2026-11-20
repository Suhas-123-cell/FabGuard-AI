from __future__ import annotations

import pytest

from fabguard.retrieval import ingest_documents, load_reference_corpus
from fabguard.retrieval_eval import load_queries, score_ranking


def test_score_ranking_matches_hand_computed_values() -> None:
    ranked = ["a", "x", "b", "y", "z", "c"]

    scores = score_ranking(ranked, ["b", "c"])

    assert scores["recall_at_5"] == pytest.approx(0.5)  # only b is inside the top 5
    assert scores["reciprocal_rank"] == pytest.approx(1 / 3)
    assert scores["hit_at_1"] == 0.0
    assert score_ranking(["a"], ["a"]) == {
        "recall_at_5": 1.0,
        "reciprocal_rank": 1.0,
        "hit_at_1": 1.0,
    }
    assert score_ranking(["q", "r"], ["a"])["reciprocal_rank"] == 0.0


def test_labelled_queries_point_at_passages_that_exist() -> None:
    passages = ingest_documents(load_reference_corpus("references/corpus.json"))
    queries = load_queries("evals/retrieval/queries.json", {p.passage_id for p in passages})

    assert len(queries) >= 30
    assert len({q["id"] for q in queries}) == len(queries)
    with pytest.raises(ValueError, match="unknown or empty"):
        load_queries("evals/retrieval/queries.json", {"nothing"})
