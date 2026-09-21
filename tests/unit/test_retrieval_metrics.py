"""
Pure-function tests, zero I/O - should be near-instant.
"""
import pytest

from app.evaluation.retrieval_metrics import mrr, ndcg_at_k, recall_at_k


def test_recall_at_k_hit_within_k():
    assert recall_at_k(["a", "b", "c"], "b", k=3) == 1.0


def test_recall_at_k_miss_within_k():
    assert recall_at_k(["a", "b", "c"], "d", k=3) == 0.0


def test_recall_at_k_hit_outside_k():
    assert recall_at_k(["a", "b", "c", "d"], "d", k=2) == 0.0


def test_mrr_first_position():
    assert mrr(["a", "b", "c"], "a") == 1.0


def test_mrr_third_position():
    assert mrr(["a", "b", "c"], "c") == pytest.approx(1 / 3)


def test_mrr_not_found():
    assert mrr(["a", "b", "c"], "d") == 0.0


def test_ndcg_at_k_first_position_is_maximal():
    assert ndcg_at_k(["a", "b", "c"], "a", k=3) == 1.0


def test_ndcg_at_k_later_position_is_lower_than_first():
    first = ndcg_at_k(["a", "b", "c"], "a", k=3)
    third = ndcg_at_k(["a", "b", "c"], "c", k=3)
    assert third < first


def test_ndcg_at_k_not_found_is_zero():
    assert ndcg_at_k(["a", "b", "c"], "d", k=3) == 0.0