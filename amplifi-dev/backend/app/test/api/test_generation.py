import pytest

from app.schemas.rag_generation_schema import (
    GenerationSettings,
    IWorkspaceGenerationResponse,
    IWorkspacePerformGenerationRequest,
)
from app.utils.uuid6 import uuid7

context_list = [
    "John lives near the sea. Jack lives in the mountains.",
    "Julie love icecream. John is a student that studies marine biology.",
]
request_body = IWorkspacePerformGenerationRequest(
    query="Who's John?",
    context=context_list,
    generation_settings=GenerationSettings(
        chat_model="GPT4o", chat_model_kwargs={"temperature": 0}
    ),
)


@pytest.mark.skip(reason="Unused Endpoint. 4o not impl for it yet")
@pytest.mark.asyncio
async def test_generation(test_client_admin, workspace_id):
    response = await test_client_admin.post(
        f"/workspace/{workspace_id}/llm_generation",
        json=request_body.model_dump(),
    )
    assert response.status_code == 200
    response = response.json()
    print(response["data"])
    response_data = IWorkspaceGenerationResponse(**response["data"])
    assert response_data.contexts_found == context_list
    assert response_data.input_tokens == 72
    assert response_data.output_tokens != -1
