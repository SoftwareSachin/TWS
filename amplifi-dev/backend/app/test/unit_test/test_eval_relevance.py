from unittest.mock import MagicMock, patch

import pytest

from app.utils.azure_fns.eval_relevance import eval_relevance


@pytest.mark.skip
def test_eval_relevance_true():
    result, tokens_used = eval_relevance(
        "What is the capital of France?", "Paris is the capital of France."
    )
    assert result is True
    assert tokens_used == 409


@pytest.mark.skip
def test_eval_relevance_false():
    result, tokens_used = eval_relevance(
        "What is the capital of France?", "The capital of Germany is Berlin."
    )
    assert result is False
    assert tokens_used == 409


@patch("app.utils.azure_fns.eval_relevance.client")
def test_eval_relevance_max_retries(mock_client):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Not True Nor False"
    mock_client.chat.completions.create.return_value = mock_response

    with pytest.raises(TypeError, match="Reached max retries"):
        eval_relevance(
            "What is the capital of France?",
            "The capital of Germany is Berlin.",
            max_retries=3,
        )
