from app.retrieval.embeddings import embed_query, embed_texts


def test_embed_query_returns_correct_dimension():
    vector = embed_query("hello world")
    assert len(vector) == 384


def test_embed_texts_batch_returns_one_vector_per_input():
    vectors = embed_texts(["first sentence", "second sentence", "third sentence"])
    assert len(vectors) == 3
    assert all(len(v) == 384 for v in vectors)


def test_embed_texts_empty_list_returns_empty():
    assert embed_texts([]) == []


def test_embed_query_similar_texts_have_higher_similarity_than_dissimilar():
    import numpy as np

    v1 = np.array(embed_query("The cat sat on the mat."))
    v2 = np.array(embed_query("A cat was sitting on a mat."))
    v3 = np.array(embed_query("Quarterly revenue exceeded forecasts."))

    sim_related = np.dot(v1, v2)
    sim_unrelated = np.dot(v1, v3)

    assert sim_related > sim_unrelated