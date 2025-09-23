# from app.utils.azure_fns.eval_attribution import eval_attribution
from typing import List, Optional

from sklearn.metrics import ndcg_score

from app.be_core.logger import logger  # # for logging
from app.schemas.eval_schema import SearchMetrics, SearchMetricsDataset
from app.utils.azure_fns.eval_relevance import (
    eval_relevance_4o,
    eval_relevance_4o_ground_truth,
)

# from app.utils.azure_fns.get_statements import get_statements


def eval_contexts(
    contexts: List[str], doc_scores: List[float], query: str, k: int
) -> tuple[SearchMetrics, int]:
    if k > len(contexts):
        logger.warning(
            f"Eval asked to evaluate {k} pieces of context, but only {len(contexts)} pieces returned. Setting k = {len(contexts)} instead and proceeding"
        )
        k = len(contexts)

    precision, ndcg_score, in_tokens = get_precision(
        query=query, contexts=contexts, doc_scores=doc_scores, k=k
    )

    return (
        SearchMetrics(
            precision=precision, ndcg_score=ndcg_score, latency=0.0  ##placeholder
        ),
        in_tokens,
    )


def eval_contexts_dataset(
    contexts: List[str],
    doc_scores: List[float],
    query: str,
    k: int,
    true_answer: Optional[str] = None,
) -> tuple[SearchMetricsDataset, int]:
    # if k > len(contexts):
    #     logger.warning(
    #         f"Eval asked to evaluate {k} pieces of context, but only {len(contexts)} pieces returned. Setting k = {len(contexts)} instead and proceeding"
    #     )
    #     k = len(contexts)

    logger.info(f"the number of contexts sent to eval contexts are are {len(contexts)}")

    logger.info(
        f"For query {query} the contexts are {len(contexts)} long doc_scores of {doc_scores} and k equal {k}"
    )

    if not true_answer:
        precision, ndcg_score, in_tokens = get_precision(
            query=query, contexts=contexts, doc_scores=doc_scores, k=k
        )

    precision, ndcg_score, in_tokens = get_precision_ground_truth(
        query=query,
        contexts=contexts,
        doc_scores=doc_scores,
        k=k,
        true_answer=true_answer,
    )

    logger.info(
        f"For query {query} the ndcg_score is {ndcg_score} and precision is {precision} with doc_scores of {doc_scores}"
    )

    return (
        SearchMetricsDataset(precision=precision, ndcg_score=ndcg_score),
        in_tokens,
    )


# Get's precision metrics @k, if k = 0, evaluates over entire list
def get_precision(query: str, contexts: List[str], doc_scores: List[float], k):
    if k > len(contexts):
        raise (ValueError("k is more than number of contexts given"))
    if k < 0:
        raise (ValueError("k can't be negative"))
    if k == 0:
        logger.warn("K is 0")
        return 0.0, 0.0, 0.0
    contexts = contexts[0:k]

    total_input_tokens = 0
    predicted_scores = []

    for context in contexts:
        relevance_score, in_tokens = eval_relevance_4o(query=query, context=context)
        total_input_tokens += in_tokens
        predicted_scores.append(relevance_score)

    precision = sum(predicted_scores) / k
    logger.debug(f"cosine scores are {doc_scores}")
    logger.debug(f"predicted scores are {predicted_scores}")
    if k < 2:
        logger.warning("skipping ndcg score calculation for k < 2")
        ndcg = 0.0
    else:
        ndcg = ndcg_score([predicted_scores], [doc_scores], k=k)

    return precision, ndcg, total_input_tokens


# Get's precision metrics @k, if k = 0, evaluates over entire list
def get_precision_ground_truth(
    query: str, contexts: List[str], doc_scores: List[float], k: int, true_answer: str
):
    # if k > len(contexts):
    #     raise (ValueError("k is more than number of contexts given"))
    # if k < 0:
    #     raise (ValueError("k can't be negative"))
    # if k == 0:
    #     logger.warn("K is 0")
    #     return 0.0, 0.0, 0.0
    # contexts = contexts[0:k]

    logger.info(
        f"the number of contexts sent to get precision ground truth are {len(contexts)}"
    )
    total_input_tokens = 0
    y_true = []  ### using LLM as a judge to rate 1 or 0 for relevance

    for context in contexts:
        relevance_score, in_tokens = eval_relevance_4o_ground_truth(
            query=query, context=context, true_answer=true_answer
        )
        total_input_tokens += in_tokens
        y_true.append(relevance_score)

    logger.debug(f"k is {k}")
    logger.debug(f"length of contexts is {len(contexts)}")
    logger.debug(f"length of y_true is {len(y_true)}")
    logger.debug(f"length of doc_scores is {len(doc_scores)}")
    logger.debug(
        f"total input tokens for {len(contexts)} chunks is {total_input_tokens}"
    )
    logger.debug(f"query is {query}")

    if y_true:
        precision = sum(y_true) / k
        ndcg = ndcg_score([y_true], [doc_scores], k=k)
    else:
        logger.debug("no scores were retrieved. setting precision and ndcg to 0")
        precision = 0
        ndcg = 0

    logger.debug(f"cosine scores are {doc_scores}")
    logger.debug(f"LLM judge scores are {y_true}")
    logger.info(f"LLM judge scores for query {query} are {y_true}")

    return precision, ndcg, total_input_tokens
