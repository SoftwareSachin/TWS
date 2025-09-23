from app.api.deps import (
    get_azure_client,
    get_gpt4o_client,
    get_numb_tokens,
    get_numb_tokens4o,
    gpt4o_deployment,
    gpt_deployment,
)
from app.be_core.logger import logger

client = get_azure_client()

system_prompt = """Given a question a piece of context verify if the context given is useful for coming to an answer for the question. Respond only 'True' or 'False'.

Below is an Example:
Question: What can you tell me about albert Albert Einstein?
Context:Albert Einstein (14 March 1879 - 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass-energy equivalence formula E = mc2, which arises from relativity theory, has been called "the world's most famous equation". He received the 1921 Nobel Prize in Physics "for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect", a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.
Answer: True

Below is another Example:
Question: who won 2020 icc world cup?
Context: The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title.
Answer: False
"""


# Given a query and a piece of context, determin if the context is relavent to the query
def eval_relevance(query: str, context: str, max_retries=5) -> tuple[bool, int]:
    tries = 0
    input_tokens_used = 0
    output_tokens = 0
    isRelavent = ""
    query = f"Is the context '{context}' relevant to the query '{query}'? Respond only with 'True' or 'False'."
    query_tokens = get_numb_tokens(query)
    logger.debug(f"Query has {query_tokens} tokens.")
    system_prompt_tokens = get_numb_tokens(system_prompt)
    logger.debug(f"System Prompt has {system_prompt_tokens} tokens.")
    while isRelavent.lower() not in ["true", "false"]:
        response = client.chat.completions.create(
            model=gpt_deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": query,
                },
            ],
            temperature=0,
            max_tokens=1,
        )
        output_tokens += get_numb_tokens(response.choices[0].message.content)
        input_tokens_used += query_tokens
        input_tokens_used += system_prompt_tokens
        isRelavent = response.choices[0].message.content
        tries += 1
        if tries >= max_retries:
            raise TypeError(f"Reached max retries ({max_retries}) with query: {query}")
    logger.info(
        f"Total Input Tokens Used: {input_tokens_used}, Total Output Tokens: {output_tokens}"
    )
    return (isRelavent.lower() == "true", input_tokens_used)


client_gpt4o = get_gpt4o_client()
system_prompt_gpt4o = """ YOU ARE AN EXPERT IN NLP EVALUATION METRICS, SPECIALLY TRAINED TO ASSESS ANSWER RELEVANCE IN RAG RETRIEVAL. YOUR TASK IS TO EVALUATE THE RELEVANCE OF A GIVEN CONTEXT CHUNK BASED ON A GIVEN USER QUESTION.

###INSTRUCTIONS###
- YOU MUST ANALYZE THE USER QUESTION TO DETERMINE THE MOST RELEVANT RESPONSE.
- EVALUATE THE CONTEXT CHUNK RETRIEVED BASED ON ITS ALIGNMENT WITH THE USER'S QUESTION.
- ASSIGN A RELEVANCE SCORE of 0 (IRRELEVANT) or 1(RELEVANT).

###WHAT NOT TO DO###
- DO NOT GIVE A SCORE WITHOUT FULLY ANALYZING THE GIVEN CONTEXT CHUNK AND THE USER QUESTION.
- AVOID SCORES THAT DO NOT MATCH THE EXPLANATION PROVIDED.
- DO NOT INCLUDE ADDITIONAL FIELDS OR INFORMATION.
- A SCORE OF 0 SHOULD ONLY BE GIVEN IF CONTEXT CONTAINS NO RELEVANT INFORMATION OR SUPPORTING DETAILS.
"""


# Given a query and a piece of context, determine if the context is relevant to the query
def eval_relevance_4o(query: str, context: str, max_retries=5) -> int:
    formatted_query = f"""
    You are comparing a reference text and trying to determine if the reference text contains information relevant to answering the question. Here is the data:
    [BEGIN DATA]
    ************
    [Question]: {query}
    ************
    [Reference text]: {context}
    [END DATA]

    Compare the question above to the reference text. You must determine whether the Reference text contains information that can answer the Question. Respond only with a 0 or 1.
    Please focus on whether the very specific question can be answered by the information in the Reference text.

    Example:

    query = "Where is France located?"
    contexts = [
        "France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches.",
        "The sky is blue because there is water in it.",
        "France is located on the planet earth",
    ]

    Relevance score for each context is 1, 0, 1 respectively.
    """

    tries = 0
    relevance_score = -1.0
    input_tokens_used = 0

    system_prompt_tokens = get_numb_tokens4o(system_prompt_gpt4o)
    logger.debug(f"Query has {system_prompt_tokens} tokens.")

    query_tokens = get_numb_tokens4o(formatted_query)
    logger.debug(f"System Prompt has {query_tokens} tokens.")

    while not (0.0 <= relevance_score <= 1.0):
        response = client_gpt4o.chat.completions.create(
            model=gpt4o_deployment,
            messages=[
                {"role": "system", "content": system_prompt_gpt4o},
                {"role": "user", "content": formatted_query},
            ],
            temperature=0,
            max_tokens=5,
        )
        relevance_score_str = response.choices[0].message.content.strip()
        input_tokens_used += system_prompt_tokens + query_tokens
        tries += 1
        output_tokens_used = get_numb_tokens4o(response.choices[0].message.content)

        val = float(relevance_score_str)

        if 0.0 <= val <= 1.0:
            relevance_score = val

        if tries >= max_retries:
            raise TypeError(f"Reached max retries ({max_retries}) with query: {query}")

    logger.info(
        f"Relevance evaluation completed with result: {relevance_score} and {output_tokens_used} output tokens used"
    )

    return (relevance_score, input_tokens_used)


# template from https://docs.llamaindex.ai/en/stable/examples/low_level/evaluation/
system_prompt_gpt4o_ground_truth = """
You are an expert retrieval evaluation system for a NLP RAG system.

You are given the following information:
- a query,
- a reference (ground-truth) answer, and
- a retrieved text.

Your job is to judge whether the retrieved text is relevant for answering the query based on the reference answer. There can be multiple relevant retrieved texts for a query.
The retrieved text does not need to contain the full answer, but it must provide factual and helpful information aligned with the ground truth.
Output a single score that represents a holistic evaluation.
You must return your response in a line with only the score.
Do not return answers in any other format.

Follow these guidelines for scoring. Your answer will be a single score either 1 o 0, where 0 is not relevant and 1 is relevant.

- Score **1** if the retrieved text supports **any part** of the reference answer or contains information directly useful for answering the query
- Score **0** only if the text is irrelevant or misleading
- The retrieved text does **not** need to match the reference answer completely — even **partial factual matches are valid**.

Example:

query = "Where is France located?"
true_answer = "Europe."

Context 1: "France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches.", gets a score of 1.
Reasoning: This context receives a score of 1 because it provides a clear and accurate description of France's geographical location.
The mention of "Western Europe" directly corresponds with the ground truth answer "Europe," offering specific and relevant information that precisely answers the query. Therefore, it is fully relevant.

Context 2: "The sky is blue because there is water in it." gets a score of 0.
Reasoning: This context receives a score of 0 because it has no connection to the query.
The user is asking about the location of France, while the retrieved text discusses a completely unrelated topic — the color of the sky.
It offers no factual support or relevance to the reference answer and is therefore not useful.

Context 3: "France is located on the planet Earth." gets a score of 1.
Reasoning: This context receives a score of 1 because it affirms a geographically correct fact that includes the reference answer.
Although the statement is broader than "Europe," it still accurately places France within a valid geographical context that encompasses the ground truth.
It contributes to answering the question, making it relevant and factually correct.

"""


# Given a query, ground truth answer and a piece of context, determine if the context is relevant to the query for answering given question
def eval_relevance_4o_ground_truth(
    query: str, context: str, true_answer: str, max_retries=5
) -> int:

    formatted_query = f"""
    Here is the data to evaluate and return an integer relevance score:

    [BEGIN DATA]
    ************
    [QUERY]: {query}
    ************
    [REFERENCE ANSWER]: {true_answer}
    ************
    [RETRIEVED TEXT]: {context}
    [END DATA]
    """

    tries = 0
    relevance_score = -1.0
    input_tokens_used = 0

    system_prompt_tokens = get_numb_tokens4o(system_prompt_gpt4o_ground_truth)
    logger.debug(f"Query has {system_prompt_tokens} tokens.")

    query_tokens = get_numb_tokens4o(formatted_query)
    logger.debug(f"System Prompt has {query_tokens} tokens.")

    while not (0.0 <= relevance_score <= 1.0):
        response = client_gpt4o.chat.completions.create(
            model=gpt4o_deployment,
            messages=[
                {"role": "system", "content": system_prompt_gpt4o_ground_truth},
                {"role": "user", "content": formatted_query},
            ],
            temperature=0,
            max_tokens=5,
        )
        relevance_score_str = response.choices[0].message.content.strip()
        input_tokens_used += system_prompt_tokens + query_tokens
        tries += 1
        output_tokens_used = get_numb_tokens4o(response.choices[0].message.content)

        val = float(relevance_score_str)

        if 0.0 <= val <= 1.0:
            relevance_score = val

        if tries >= max_retries:
            raise TypeError(f"Reached max retries ({max_retries}) with query: {query}")

    logger.info(
        f"Relevance evaluation completed with result: {relevance_score} and {output_tokens_used} output tokens used"
    )

    return (relevance_score, input_tokens_used)
