# from app.api.deps import gpt_deployment, get_instructor_client
# from typing import List

# from app.schemas.synth_data_schema import SyntheticFactoid, QAPair

# instructor_client = get_instructor_client()

# system_prompt = """You are a question generation asistant.
# Given a list of sentences, you will generate a question and answer based on the sentences provided.
# The question should be regarding information specific to the sentences given.
# Return only the question and answer.

# Example input: France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches. Paris, its capital, is famed for its fashion houses, classical art museums including the Louvre and monuments like the Eiffel Tower.

# Example question: Where is France and what is it's capital?

# Example Answer: France is in Western Europe and France's captial is Paris"""


# # Given some context, generate questions related to the context
# def generate_question(context: str | List[str]) -> SyntheticFactoid:
#     if isinstance(context, List):
#         context = " ".join(context)
#     response = instructor_client.chat.completions.create(
#         model=gpt_deployment,
#         response_model=QAPair,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {
#                 "role": "user",
#                 "content": f"Generate question based on context below:\n{context}",
#             },
#         ],
#         temperature=0,
#     )
#     print(response.question)
#     print(response.answer)
#     return SyntheticFactoid(Context=context, QA=response)


# if __name__ == "__main__":
#     generate_question(
#         "The Taj Mahal is a symbol of love and architectural marvel located in Agra, India. It was built by the Mughal emperor Shah Jahan in memory of his beloved wife, Mumtaz Mahal. The structure is renowned for its intricate marble work and beautiful gardens surrounding it."
#     )
