import asyncio
from uuid import UUID

from app.schemas.rag_generation_schema import (
    ChatModelEnum,
    GenerationSettings,
    IWorkspaceGenerationResponse,
    RagContext,
    RagContextWrapper,
)
from app.utils.llm_fns.azure_openai import query_llm_with_history


def test_query_llm_with_history():
    history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Why's the sky blue?"},
    ]

    # Wrap RagContext inside RagContextWrapper
    contexts = RagContextWrapper(
        rag_contexts=[
            RagContext(text="Rayleigh scattering causes the sky to appear blue.")
        ],
        raw_results=[],  # You can keep raw_results empty if not required for this test
    )

    generation_settings = GenerationSettings(
        chat_model=ChatModelEnum.GPT35,
        chat_model_kwargs={},
        custom_prompt=None,
    )

    response = asyncio.run(
        query_llm_with_history(
            query="Why's the sky blue?",
            history=history,
            contexts=contexts,
            generation_settings=generation_settings,
            tables=[],
        )
    )

    assert isinstance(
        response, IWorkspaceGenerationResponse
    ), "Response should be an instance of IWorkspaceGenerationResponse"
    assert response is not None, "Response should not be None"
