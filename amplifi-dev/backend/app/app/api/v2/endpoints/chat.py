from fastapi import APIRouter, Depends, HTTPException

from app.api import deps
from app.schemas.chat_schema import ChatRequest
from app.schemas.role_schema import IRoleEnum
from app.schemas.user_schema import UserData
from app.utils.chat.chat_utils import getChatResponse

router = APIRouter()


@router.post("/chat")
async def chat_endpoint(
    request: ChatRequest,
    current_user: UserData = Depends(
        deps.get_current_user_or_api_client(
            required_roles=[IRoleEnum.admin, IRoleEnum.developer, "api_client"]
        )
    ),
):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    return await getChatResponse(
        chatRequest=request, organization_id=current_user.organization_id
    )
