from app.retrieval.hybrid import reciprocal_rank_fusion


def test_item_ranked_in_both_lists_beats_item_in_only_one():
    ranking_a = ["x", "y", "z"]
    ranking_b = ["y", "x", "w"]
    fused = dict(reciprocal_rank_fusion([ranking_a, ranking_b]))

    assert fused["x"] > fused["z"]
    assert fused["y"] > fused["z"]


def test_top_ranked_in_both_lists_wins_overall():
    ranking_a = ["a", "b", "c"]
    ranking_b = ["a", "c", "b"]
    fused = reciprocal_rank_fusion([ranking_a, ranking_b])
    assert fused[0][0] == "a"


def test_empty_rankings_return_empty_result():
    assert reciprocal_rank_fusion([[], []]) == []


def test_item_appearing_in_only_one_ranking_is_still_scored():
    fused = dict(reciprocal_rank_fusion([["only_here"], []]))
    assert fused["only_here"] > 0