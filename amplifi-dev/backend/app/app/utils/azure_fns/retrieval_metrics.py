import json
from uuid import uuid4

import numpy as np
from sklearn.metrics import ndcg_score

# from app.be_core.logger import logger
from app.api.deps import (
    gpt4o_deployment_batch,
)

# Template from arize-phoenix can change later: https://colab.research.google.com/github/Arize-ai/phoenix/blob/main/tutorials/evals/evaluate_rag.ipynb#scrollTo=BA9x2zjd7Y79
## generating question answer pairs using phoenix
generate_questions_template = """
Context information is provided below.

---------------------
{text}
---------------------

Given the context information and not prior knowledge generate only questions based on the below query.

You are a Teacher/ Professor. Your task is to setup a question for an upcoming quiz/examination.

The questions should be diverse in nature. Refrain from making it hyper-specific to the chunk. Please make them generalized questions.

You are an expert assistant tasked with formulating distinct and relevant questions. The questions should cover different aspects of the context.

Output the question in JSON format with the key: question.
"""


# refer to gnis for metrics calculation: https://colab.research.google.com/github/Arize-ai/phoenix/blob/main/tutorials/evals/evaluate_rag.ipynb#scrollTo=BA9x2zjd7Y79
# Compute evaluation metrics
def compute_ndcg(df, k):
    n = max(2, len(df))
    n = max(2, len(df))
    eval_scores = np.zeros(n)
    doc_scores = np.zeros(n)
    eval_scores[: len(df)] = df["relevance_score"]
    doc_scores[: len(df)] = df["search_score"].astype(float)

    try:
        return ndcg_score([eval_scores], [doc_scores], k=k)
    except ValueError:
        return np.nan


def compute_metrics(df, k):
    ndcg_at_k = df.groupby("question_id").apply(lambda x: compute_ndcg(x, k=k))

    precision_at_k = df.groupby("question_id").apply(
        lambda x: x["relevance_score"][:k].sum(skipna=False) / k
    )

    average_ndcg_at_k = ndcg_at_k.mean(skipna=True)
    average_precision_at_k = precision_at_k.mean(skipna=True)

    return [average_ndcg_at_k, average_precision_at_k]


def output_parser(response: str, index: int):
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        return {"__error__": str(e)}


system_prompt_gpt4o = """You will be provided with the following fields of information: question_id, question_text, true_answer, retrieved_chunk, search_score, chunk_position. Do not modify these. You will be answering only within a new field called "relevance_score".

YOU ARE AN EXPERT IN NLP EVALUATION METRICS, SPECIALLY TRAINED TO ASSESS ANSWER RELEVANCE IN RAG RETRIEVAL. YOUR TASK IS TO EVALUATE THE RELEVANCE OF A GIVEN CONTEXT CHUNK in "retrieved_chunk" BASED ON A GIVEN USER QUESTION in "question_text". You must determine whether the Reference text
contains information that can answer the Question.

###INSTRUCTIONS###
- YOU MUST ANALYZE THE USER QUESTION TO DETERMINE THE MOST RELEVANT RESPONSE.
- EVALUATE THE CONTEXT CHUNK RETRIEVED BASED ON ITS ALIGNMENT WITH THE USER'S QUESTION.
- ASSIGN A RELEVANCE SCORE BETWEEN 0 (COMPLETELY IRRELEVANT) and 1(HIGHLY RELEVANT).
- RETURN THE RESULT AS A JSON OBJECT, INCLUDING THE RELEVANCE SCORE.
###CHAIN OF THOUGHTS###
1. **Understanding the context and input:**
    1.1. READ AND COMPREHEND THE CONTEXT PROVIDED.
    1.2. IDENTIFY THE KEY POINTS OR QUESTIONS IN THE USER'S INPUT THAT THE ANSWER SHOULD ADDRESS.
2. **Evaluating the Answer:**
    2.1. COMPARE THE CONTEXT CHUNK TO THE QUESTION.
    2.2. DETERMINE WHETHER THE CONTEXT ADDRESSES THE USER'S QUERY OR PROVIDES RELEVANT INFORMATION.
    2.3. CONSIDER ANY EXTRANEOUS OR OFF-TOPIC INFORMATION THAT MAY DECREASE RELEVANCE.
3. **Assigning a Relevance Score:**
    3.1. ASSIGN A SCORE BASED ON HOW WELL THE GIVEN CONTEXT CHUNK MATCHES THE USER QUESTION.
4. **Generating the JSON Output:**
    4.1. ONLY OUTPUT THE SCORE. ENSURE THE SCORE AS a FLOAT BETWEEN 0-1.
###WHAT NOT TO DO###
- DO NOT GIVE A SCORE WITHOUT FULLY ANALYZING THE GIVEN CONTEXT CHUNK, THE USER QUESTION AND GROUND-TRUTH CONTEXT.
- AVOID SCORES THAT DO NOT MATCH THE EXPLANATION PROVIDED.
- DO NOT INCLUDE ADDITIONAL FIELDS OR INFORMATION.
- NEVER ASSIGN A PERFECT SCORE UNLESS THE GIVEN CONTEXT CHUNK IS THE SAME AS THE CONTEXT IN "true_answer" AND FREE OF ANY IRRELEVANT INFORMATION.
- NEVER ASSIGN A SCORE OF 0 UNLESS THE GIVEN CONTEXT CHUNK IS COMPLETELY UNRELATED TO THE "true_answer." A SCORE OF 0 SHOULD ONLY BE GIVEN IF CONTEXT CONTAINS NO RELEVANT INFO, SUPPORTING DETAILS, OR PARTIAL OVERLAPS WITH THE TRUE ANSWER. IF THE CONTEXT HAS EVEN ANY MINOR RELEVANCE, ASSIGN A NONZERO SCORE.
- BE VERY CAUTIOUS OF ASSIGNING A SCORE BELOW 0.5. YOU MUST HAVE A GOOD EXPLANATION IF ASSIGNING A SCORE BELOW 0.5.

Now, for each given question and context, return a JSON object that maintains the order of the input and appends a `relevance_score` field at the end based on the relevance of the context chunk to the true answer determined.

Example output for each question-context pair:
{
    "question_id": 1,
    "question_text": "When was Albert Einstein born?",
    "true_answer": "Albert Einstein (14 March 1879 - 18 April 1955)   was a renowned theoretical physicist who   developed the theory of relativity, one   of the two pillars of modern physics.",
    "retrieved_chunk": "Albert Einstein (14 March 1879 - 18 April 1955) was a renowned theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics.",
    "search_score": 0.9,
    "chunk_position": 1,
    "relevance_score": 1
},

{
    "question_id": 5,
    "question": "When was Socrates?",
    "chunk_position": 3,
    "true_answer": "Socrates Socrates A marble head of Socrates in the Louvre (copy of a lost bronze head by Lysippus)[1] Born c. 470 BC Deme Alopece, Athens Died 399 BC (aged approximately 71) Athens Cause of death Forced suicide by poisoning Spouse(s) Xanthippe, Myrto (disputed) Children Lamprocles, Menexenus, Sophroniscus Family Sophroniscus (father), Phaenarete (mother), Patrocles (half-brother) Era Ancient Greek philosophy Region Western philosophy School Classical Greek philosophy Notable students Plato, Xenophon, Antisthenes, Aristippus, Alcibiades, Critias Main interests Epistemology, ethics, teleology Notable ideas Social gadfly, Socratic dialogue, Socratic intellectualism, Socratic irony, Socratic method, Socratic paradox, Socratic questioning, 'The unexamined life is not worth living.'",
    "context_chunk": "Socrates (/sɒkrətiz/,[2] Greek: Σωκράτης; c. 470  399 BC) was a Greek philosopher from Athens who is credited as the founder of Western philosophy[3] and as among the first moral philosophers of the ethical tradition of thought. An enigmatic figure, Socrates authored no texts and is known mainly through the posthumous accounts of classical writers, particularly his students Plato and Xenophon. These accounts are written as dialogues, in which Socrates and his interlocutors examine a subject in the style of question and answer; they gave rise to the Socratic dialogue literary genre. Contradictory accounts of Socrates make a reconstruction of his philosophy nearly impossible, a situation known as the Socratic problem. Socrates was a polarizing figure in Athenian society. In 399 BC, he was accused of impiety and corrupting the youth. After a trial that lasted a day, he was sentenced to death. He spent his last day in prison, refusing offers to help him escape.",
    "search_score": 0.8,
    "relevance_score": 0.98
},

{
    "question_id": 6,
    "question": "What was Socrates' main philosophy?",
    "chunk_position": 2,
    "true_answer": "Socrates was known for his contributions to Western philosophy, particularly his emphasis on ethics, epistemology, and the Socratic method. He believed in questioning assumptions and encouraging critical thinking through dialogue. His ideas were preserved through the works of his students, most notably Plato and Xenophon.",
    "context_chunk": "Socrates is best known for developing the Socratic method, a form of cooperative argumentative dialogue in which participants ask and answer questions to stimulate critical thinking and draw out underlying beliefs. He challenged conventional wisdom and emphasized the importance of self-examination, famously stating, 'The unexamined life is not worth living.' His teachings laid the foundation for classical Greek philosophy and greatly influenced thinkers like Plato and Aristotle.",
    "search_score": 0.75,
    "relevance_score": 0.7
},

{
    "question_id": 7,
    "question": "What were Socrates' main teachings?",
    "chunk_position": 1,
    "true_answer": "Socrates emphasized critical thinking, self-examination, and the pursuit of truth. He developed the Socratic method, which involves asking questions to challenge assumptions. His philosophy influenced his students, including Plato and Aristotle, and shaped Western philosophical thought.",
    "context_chunk": "The Roman Empire was one of the largest empires in history, spanning across Europe, Africa, and Asia. It was known for its military prowess, legal system, and cultural advancements. The empire was founded in 27 BC and fell in AD 476 in the West, though the Eastern Roman Empire (Byzantine Empire) continued for another thousand years.",
    "search_score": 0.35,
    "relevance_score": 0.1
}

"""


def extract_question(output):
    """Extracts the 'question' key from the JSON string in the 'output' column."""
    if (
        not isinstance(output, str) or output.strip() == ""
    ):  # Ensure it's a valid string
        return None
    try:
        parsed_output = json.loads(output)  # Convert string to dict
        return parsed_output.get("question", None)  # Extract "question"
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e} - Invalid JSON: {output}")  # Debugging info
        return None


def create_batch_input_file(all_results):
    batch_input_data = []

    for question_data in all_results:

        true_answer = question_data["true_answer"]
        reference_text = question_data["retrieved_chunk"]
        question = question_data["question_text"]
        chunk_position = question_data["chunk_position"]
        question_id = question_data["question_id"]
        custom_id = str(uuid4())
        search_score = question_data["search_score"]

        query = f"""
        question_id is '{question_id}', chunk_position is '{chunk_position}', true_answer is '{true_answer}', context_chunk is '{reference_text}', and search_score is '{search_score}'.
        You are comparing a reference text to and trying to determine if the reference text contains information relevant to answering the question. Here is the data:
        [BEGIN DATA]
        ************
        [Question]: {question}
        ************
        [Reference text]: {reference_text}
        [END DATA]

        Compare the question above to the reference text. You must determine whether the Reference text contains information that can answer the Question.
        Please focus on whether the very specific question can be answered by the information in the Reference text.

        Append relevance_score at the end and return a JSON object. Your response for the relevance score must be a float between 0 and 1 and should not contain any text or characters aside from that.
        """

        batch_input_data.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/chat/completions",
                "body": {
                    "model": gpt4o_deployment_batch,  ### set to gpt-4o-2 for batch api. IF need standard deployment rename, set model to "gpt-4o"
                    "messages": [
                        {"role": "system", "content": system_prompt_gpt4o},
                        {
                            "role": "user",
                            "content": query,
                        },
                    ],
                },
            }
        )

    # Create a .jsonl file to hold the batch input data
    input_file_path = "batchinput.jsonl"
    with open(input_file_path, "w") as f:
        for item in batch_input_data:
            f.write(json.dumps(item) + "\n")

    return input_file_path
