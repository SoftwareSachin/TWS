from typing import List

from app.api.deps import (
    get_async_azure_client,
    get_async_gpt4o_client,
    get_gpt5_client_async,
    get_gpt41_client_async,
)
from app.be_core.logger import logger
from app.schemas.rag_generation_schema import (
    ChatModelEnum,
    GenerationSettings,
    IWorkspaceGenerationResponse,
    RagContextWrapper,
)
from app.schemas.search_schema import ImageSearchResult
from app.utils.llm_fns.lang_detect import construct_ssml_multilingual

client = get_async_azure_client()
client_4o = get_async_gpt4o_client()
client_41 = get_gpt41_client_async()
client_5 = get_gpt5_client_async()


async def query_openai_chat_history(
    history: List[dict],
    model_name: ChatModelEnum = ChatModelEnum.GPT35,
    **chat_model_kwargs,
):
    """
    Query the model asynchronously, providing chat history

    Args:
        history (List[dict]): list of
        user_prompt (str): The user prompt for the model.
        **chat_model_kwargs (Any): Additional arguments for the chat completion API.

    Returns:
        Any: The response from the GPT model.
    """
    if model_name == ChatModelEnum.GPT4o:
        used_client = client_4o
        model = "gpt-4o"
    elif model_name == ChatModelEnum.GPT35:
        used_client = client
        model = "gpt-35-turbo"
    elif model_name == ChatModelEnum.GPT41:
        used_client = client_41
        model = "gpt-4.1"
    elif model_name == ChatModelEnum.GPT5:
        used_client = client_5
        model = "gpt-5"
    else:
        raise ValueError(f"Model {model_name} Not Found or Not Implemented")
    response = await used_client.chat.completions.create(
        model=model,
        messages=history,
        **chat_model_kwargs,
    )
    return response


async def query_llm_with_history(
    query: str,
    history: List[dict],
    contexts: RagContextWrapper,
    generation_settings: GenerationSettings,
    tables: List[str] = [],
) -> IWorkspaceGenerationResponse:

    # Build GPT-4o vision-compatible messages
    messages = build_vision_messages(
        query=query,
        contexts=contexts,
        system_prompt=history[0]["content"],  # system prompt
        chat_histories=history[1:],  # Exclude system message
    )

    # Query OpenAI
    response = await query_openai_chat_history(
        history=messages,
        model_name=generation_settings.chat_model,
        **generation_settings.chat_model_kwargs,
    )
    logger.info(f"LLM response: {response}")

    return IWorkspaceGenerationResponse(
        answer=response.choices[0].message.content,
        contexts_found=contexts,  # Return the wrapper
        full_response=str(response),
        ssml=construct_ssml_multilingual(response.choices[0].message.content),
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
    )


def build_vision_messages(
    query: str,
    contexts: RagContextWrapper,
    system_prompt: str,
    chat_histories: List[dict],
) -> List[dict]:
    messages = [{"role": "system", "content": system_prompt}]

    # Add chat history
    for history in chat_histories:
        messages.append({"role": "user", "content": history["content"]})
        messages.append(
            {"role": "assistant", "content": history.get("llm_response", "")}
        )

    # Create a mapping of file_id to base64 from raw_results
    base64_mapping = {
        str(result.file_id): result.chunk_metadata.base64
        for result in contexts.raw_results
        if isinstance(result, ImageSearchResult) and result.chunk_metadata.base64
    }

    # Add context: text + image format for GPT-4o
    prompt_parts = []
    image_prompts = []
    for ctx in contexts.rag_contexts:
        if ctx.file and ctx.file.get("mime_type", "").startswith(
            "image/"
        ):  # Added check for ctx.file
            file_id = ctx.file.get("file_id")
            base64_data = base64_mapping.get(file_id)
            if base64_data:
                mime = ctx.file.get("mime_type", "image/jpeg")
                image_prompts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{base64_data}"},
                    }
                )
        else:
            prompt_parts.append(ctx.text)

    # User message with both context and image
    content = []
    if prompt_parts:
        content.append({"type": "text", "text": "\n\n".join(prompt_parts)})
    content.extend(image_prompts)
    content.append({"type": "text", "text": f"Question: {query}"})

    messages.append({"role": "user", "content": content})
    return messages
