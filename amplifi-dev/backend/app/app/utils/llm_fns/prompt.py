from typing import List

from app.schemas.rag_generation_schema import CustomGenerationPrompt


def get_user_prompt(
    query: str,
    context: List[str],
    tables: List[str] = [],
    settings: CustomGenerationPrompt = None,
) -> str:
    if len(context) == 0:
        return query
    prompt = "Answer the question based on the context below. \n" + "Context:\n"
    prompt_end = f"\n\nQuestion: {query}\nAnswer:"
    for c in context:
        prompt += "\n\n---\n\n"
        prompt += c
    if tables:
        prompt += "There are some contexts that may have been chunks of tables, below are the full tables in markdown format that you can use for further context:"
        for md_table in tables:
            prompt += md_table
            prompt += "\n\n---\n\n"
    return prompt + prompt_end


def get_system_prompt(settings: CustomGenerationPrompt = None) -> str:
    return "You are a helpful assistant that answers queries based on context."
