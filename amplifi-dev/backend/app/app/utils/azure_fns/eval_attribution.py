# from typing import List

# from app.api.deps import (
#     gpt_deployment,
#     get_azure_client,
#     get_instructor_client,
# )
# from pydantic import BaseModel

# client = get_azure_client()
# instructor_client = get_instructor_client()


# # Given a truth statement and some context, return if context supports statement
# def eval_attribution(
#     truth_statement: str, context: str | List[str], max_retries=5
# ) -> bool:
#     if isinstance(context, List):
#         context = "\n".join(context)
#     system_prompt = """Given a statement and some context, analyze the statement and context to determine if any of the context directly supports the statement. If any piece of context directly supports the statement, return 'True'. If no piece of context directly supports the statement, return 'False'. Only return 'True' or 'False'."""
#     tries = 0
#     isAttributed = ""
#     query = f"Is the statement '{truth_statement}' directly supported by any of the context below:\n {context}\n Respond only with 'True' or 'False'."
#     print(query)
#     while isAttributed.lower() not in ["true", "false"]:
#         response = client.chat.completions.create(
#             model=gpt_deployment,
#             messages=[
#                 {
#                     "role": "system",
#                     "content": system_prompt,
#                 },
#                 {
#                     "role": "user",
#                     "content": query,
#                 },
#             ],
#             temperature=0,
#         )
#         isAttributed = response.choices[0].message.content
#         tries += 1
#         if tries >= max_retries:
#             raise TypeError(f"Reached max retries ({max_retries}) with query: {query}")
#     return isAttributed.lower() == "true"


# class BoolList(BaseModel):
#     booleans: List[int]


# # Given a list of truth statements and some context, return if context supports each statement (as a list)
# def eval_attribution_list(
#     truth_statements: List[str], context: str | List[str], max_retries=5
# ) -> bool:
#     if isinstance(context, List):
#         context = "\n".join(context)
#     system_prompt = """Given a list of statements and some context, analyze each statement in conjunction with all the context to determine if any of the context directly supports each statement. Return 1 for True, 0 for False with one value for each statement in the list of statements."""
#     tries = 0
#     isAttributed = BoolList(booleans=[])
#     first_query = f"Are the statements the statement '{truth_statements}' supported by any of the context below:\n {context}\n"
#     response = (
#         client.chat.completions.create(
#             model=gpt_deployment,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {
#                     "role": "user",
#                     "content": first_query,
#                 },
#             ],
#         )
#         .choices[0]
#         .message.content
#     )
#     # print(response)
#     while len(isAttributed.booleans) != len(truth_statements):
#         response = instructor_client.chat.completions.create(
#             model=gpt_deployment,
#             response_model=BoolList,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {
#                     "role": "user",
#                     "content": first_query,
#                 },
#                 {"role": "assistant", "content": response},
#                 {
#                     "role": "user",
#                     "content": "Now return that as only a list of 1s and 0s.",
#                 },
#             ],
#         )
#         isAttributed = response
#         # print(f"response: {isAttributed.booleans}")
#         tries += 1
#         if tries >= max_retries:
#             # TODO: maybe not TypeError?
#             raise TypeError(
#                 f"Reached max retries ({max_retries}) with query: Is the statement '{truth_statements}' supported by any of the context below:\n {context}\n"
#             )
#     return isAttributed.booleans


# if __name__ == "__main__":
#     statements = [
#         "France is in Western Europe.",
#         "The capital of France is Paris.",
#         "The sky is blue",
#         "1 + 3 = 4",
#     ]
#     contexts = [
#         "France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches. The country is also renowned for its wines and sophisticated cuisine. Lascaux's ancient cave drawings, Lyon's Roman theater and the vast Palace of Versailles attest to its rich history."
#     ]
#     # Below should return [1, 0, 0], but might get confused and return [1, 1, 0]
#     print(eval_attribution_list(statements, context=contexts))
#     # Below should return True, False, False
#     for statement in statements:
#         print(eval_attribution(context=contexts, truth_statement=statement))
