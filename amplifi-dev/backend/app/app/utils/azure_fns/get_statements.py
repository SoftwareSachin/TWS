# from app.api.deps import gpt_deployment, get_instructor_client
# from pydantic import BaseModel, Field

# from typing import List

# instructor_client = get_instructor_client()


# class Statements(BaseModel):
#     statements: List[str] = Field(
#         "List of individual statements extracted from the text."
#     )


# # TODO: add some examples?
# system_prompt = "Extract and resolve the statement given by the user into a list of one or more individual statements. Don't use pronouns like 'he', 'she', 'they', 'it', etc. Instead, replace them with the actual noun the pronoun represents."


# # Split a ground truth into multiple individual statements
# def get_statements(ground_truth: str) -> List[str]:
#     response = instructor_client.chat.completions.create(
#         model=gpt_deployment,
#         response_model=Statements,
#         messages=[
#             {
#                 "role": "system",
#                 "content": system_prompt,
#             },
#             {
#                 "role": "user",
#                 "content": f"Statement: {ground_truth}\n\n REMEMBER: Don't use pronouns like 'he', 'she', 'they', 'it', etc. Instead, replace them with the actual noun the pronoun represents.",
#             },
#         ],
#     )
#     return response.statements


# if __name__ == "__main__":
#     statements = get_statements(
#         "France is in Western Europe and its capital is Paris. The sky's blue since water flows upwards."
#     )
#     print(statements)
