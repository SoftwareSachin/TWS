import pytest

from app.utils.azure_fns.get_scores import get_precision


# Test when k is valid and contexts have relevant and irrelevant items
def test_get_precision():
    query = "Where is France located?"
    contexts = [
        "France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches.",
        "The sky is blue because there is water in it.",
        "France is located on the planet earth",
    ]

    doc_scores = [1.0, 0.1, 0.8]  # example ones
    precision, ndcg, total_input_tokens = get_precision(
        query, contexts, doc_scores, k=3
    )

    # assert precision == pytest.approx(0.66666666, rel=1e-7)
    # assert ndcg == 1.0
    assert total_input_tokens == 1262


# Test when k is more than the number of contexts
def test_get_precision_k_greater_than_contexts():
    with pytest.raises(ValueError, match="k is more than number of contexts given"):
        get_precision("query", ["context1", "context2"], [0.9, 0.8], k=3)


# Test when k is negative
def test_get_precision_k_negative():
    with pytest.raises(ValueError, match="k can't be negative"):
        get_precision("query", ["context1"], [0.9], k=-1)


# Test when k is zero, should return all zeros
def test_get_precision_k_zero():
    result = get_precision("query", ["context1"], [0.9], k=0)
    assert result == (0.0, 0.0, 0)
