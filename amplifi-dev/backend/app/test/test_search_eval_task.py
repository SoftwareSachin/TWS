# import pytest
# from app.api.search_eval_task import eval_task
# from app.schemas.eval_schema import PrecisionScores, SearchEvalValues


# @pytest.mark.parametrize(
#     "query, dataset_id, search_limit, search_index_type, probes, ef_search, expected_precision_scores",
#     [
#         (
#             "Where is the best place to sit in the rain and watch ducks fly?",
#             "wrong_dataset_id",
#             5,
#             "cosine_distance",
#             10,
#             40,
#             PrecisionScores(precision=0, mean_reciprocal_rank=0, top_reciprocal_rank=0),
#         ),
#     ],
# )
# def test_eval_task(
#     query,
#     dataset_id,
#     search_limit,
#     search_index_type,
#     probes,
#     ef_search,
#     expected_precision_scores,
# ):
#     # Call the eval_task function with real parameters
#     response = eval_task(
#         query, dataset_id, search_limit, search_index_type, probes, ef_search
#     )

#     # Assertions
#     assert isinstance(response, SearchEvalValues)
#     assert response.query == query
#     assert isinstance(response.vector_search_results_text, list)  # Ensure it's a list
#     assert len(response.vector_search_results_text) == 0
#     assert response.precision_scores == expected_precision_scores
#     assert response.gpt_input_tokens == 0
