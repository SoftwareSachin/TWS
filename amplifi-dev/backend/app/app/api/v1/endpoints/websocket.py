from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from redis.asyncio import Redis

from app.api.deps import get_authenticated_userdata, get_redis_client
from app.schemas.long_task_schema import ITaskType

router = APIRouter()


@router.websocket("/")
async def websocket_endpoint(
    websocket: WebSocket,
    task_type: ITaskType,
    redis_client: Redis = Depends(get_redis_client),
):
    """
    Handles WebSocket connections for different task types.
    The client must send a valid token and specify a task type.
    """
    try:
        await websocket.accept()

        # Extract Authorization header
        headers = websocket.headers
        auth_header = headers.get("Authorization")

        if not auth_header or not auth_header.startswith("Bearer "):
            await websocket.close(
                code=4001, reason="Missing or invalid Authorization header"
            )
            return

        token = auth_header.split(" ")[1]
        try:
            current_user = await get_authenticated_userdata(token, redis_client)
        except HTTPException as e:
            await websocket.close(code=4000, reason=str(e.detail))
            return

        pubsub = redis_client.pubsub()
        await pubsub.subscribe(
            f"{current_user.id}:{task_type}"  # Dynamic channel for task type
        )  # Subscribe to client-specific task updates

        try:
            while True:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message:
                    # Send message to WebSocket client
                    await websocket.send_text(message["data"])
        except WebSocketDisconnect:
            print(f"User {current_user.id} disconnected from {task_type} updates")

    except HTTPException as e:
        await websocket.close(code=4000, reason=str(e.detail))
