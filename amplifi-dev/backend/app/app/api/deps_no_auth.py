from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Callable, Optional, Union
from uuid import UUID, uuid4

import instructor
import openai
import redis
import redis.asyncio as aioredis
import tiktoken
from fastapi import Depends, HTTPException, Path, status
from fastapi.security import OAuth2PasswordBearer
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from redis.asyncio import Redis, RedisCluster
from redis.cluster import ClusterNode
from sqlmodel.ext.asyncio.session import AsyncSession
from tavily import TavilyClient

from app import crud
from app.be_core.config import settings
from app.be_core.logger import logger
from app.db.session import SessionLocal, SessionLocalCelery
from app.schemas.common_schema import IMetaGeneral
from app.schemas.rag_generation_schema import ChatModelEnum
from app.schemas.user_schema import UserData

# Mock authentication - always return the same user
MOCK_USER_DATA = UserData(
    id="test-user-123",
    email="test@amplifi.com", 
    first_name="Test",
    last_name="User",
    is_active=True,
    is_superuser=True,
    role=["admin", "developer", "user"],
    organization_id="test-org-123",
    phone=None,
    state=None,
    country=None,
    address=None,
)

# Bypass OAuth2 - not actually used
reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V2_STR}/login/access-token", auto_error=False
)

async_redis_client: Optional[Union[Redis, RedisCluster]] = None


async def get_redis_client() -> Union[Redis, RedisCluster]:
    global async_redis_client
    if not async_redis_client:
        if settings.REDIS_MODE == "cluster":
            startup_nodes = [
                ClusterNode(
                    f"redis-cluster-{i}.{settings.REDIS_HOST}", settings.REDIS_PORT
                )
                for i in range(settings.REDIS_CLUSTER_SIZE)
            ]
            async_redis_client = await RedisCluster(
                startup_nodes=startup_nodes,
                max_connections=10,
                encoding="utf8",
                decode_responses=True,
                socket_timeout=5.0,
            )
        else:
            async_redis_client = await aioredis.from_url(
                f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",
                max_connections=25,
                encoding="utf8",
                decode_responses=True,
            )
    return async_redis_client


async def close_async_redis_client():
    global async_redis_client
    if async_redis_client:
        await async_redis_client.close()
        async_redis_client = None


def get_redis_client_sync() -> Union[redis.Redis, redis.RedisCluster]:
    if settings.REDIS_MODE == "cluster":
        startup_nodes = [
            ClusterNode(f"redis-cluster-{i}.{settings.REDIS_HOST}", settings.REDIS_PORT)
            for i in range(settings.REDIS_CLUSTER_SIZE)
        ]
        return redis.RedisCluster(
            startup_nodes=startup_nodes,
            max_connections=25,
            encoding="utf8",
            decode_responses=True,
            socket_timeout=5.0,
        )
    else:
        return redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            max_connections=25,
            encoding="utf8",
            decode_responses=True,
        )


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session


async def get_jobs_db() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocalCelery() as session:
        yield session


async def get_general_meta() -> IMetaGeneral:
    current_roles = await crud.role.get_multi(skip=0, limit=100)
    return IMetaGeneral(roles=current_roles)


# *** AUTHENTICATION BYPASSED - MOCK FUNCTIONS ***

async def organization_check(
    organization_id: UUID = Path(...),
    access_token: Optional[str] = Depends(reusable_oauth2),
    redis_client: Redis = Depends(get_redis_client),
):
    """BYPASSED: Allow all organization access"""
    logger.debug(f"BYPASSED organization_check for {organization_id}")
    return True


async def workspace_check(
    workspace_id: UUID = Path(...),
    access_token: Optional[str] = Depends(reusable_oauth2),
    redis_client: Redis = Depends(get_redis_client),
    db: AsyncSession = Depends(get_db),
):
    """BYPASSED: Allow all workspace access"""
    logger.debug(f"BYPASSED workspace_check for {workspace_id}")
    return True


async def user_workspace_access_check(
    workspace_id: UUID = Path(...),
    access_token: Optional[str] = Depends(reusable_oauth2),
    redis_client: Redis = Depends(get_redis_client),
    db: AsyncSession = Depends(get_db),
):
    """BYPASSED: Allow all workspace access"""
    logger.debug(f"BYPASSED workspace access for {workspace_id}")
    return True


async def dataset_check(
    dataset_id: UUID = Path(...),
    access_token: Optional[str] = Depends(reusable_oauth2),
    redis_client: Redis = Depends(get_redis_client),
    db: AsyncSession = Depends(get_db),
):
    """BYPASSED: Allow all dataset access"""
    logger.debug(f"BYPASSED dataset_check for {dataset_id}")
    return True


async def chatapp_check(
    chatapp_id: UUID = Path(...),
    access_token: Optional[str] = Depends(reusable_oauth2),
    redis_client: Redis = Depends(get_redis_client),
    db: AsyncSession = Depends(get_db),
):
    """BYPASSED: Allow all chatapp access"""
    logger.debug(f"BYPASSED chatapp check for {chatapp_id}")
    return True


async def chatsession_check(
    chat_session_id: UUID = Path(...),
    chatapp_id: Optional[UUID] = Path(...),
    access_token: Optional[str] = Depends(reusable_oauth2),
    redis_client: Redis = Depends(get_redis_client),
    db: AsyncSession = Depends(get_db),
):
    """BYPASSED: Allow all chatsession access"""
    logger.debug(f"BYPASSED chatsession check for {chat_session_id}")
    return True


async def get_authenticated_userdata(
    access_token: Optional[str] = None, 
    redis_client: Optional[Redis] = None, 
    required_roles: list[str] = None
) -> UserData:
    """BYPASSED: Always return mock user data"""
    logger.debug("BYPASSED authentication - returning mock user")
    return MOCK_USER_DATA


async def get_api_client_userdata(
    access_token: Optional[str] = None, 
    db: Optional[AsyncSession] = None, 
    required_roles: list[str] = None
) -> UserData:
    """BYPASSED: Always return mock user data for API clients"""
    logger.debug("BYPASSED API client authentication - returning mock user")
    return MOCK_USER_DATA


def get_current_user(required_roles: list[str] = None) -> Callable[[], UserData]:
    """BYPASSED: Always return mock user regardless of roles"""
    async def current_user(
        access_token: Optional[str] = Depends(reusable_oauth2),
        redis_client: Redis = Depends(get_redis_client),
        db: AsyncSession = Depends(get_db),
    ) -> UserData:
        logger.debug(f"BYPASSED get_current_user with roles {required_roles}")
        return MOCK_USER_DATA

    return current_user


def get_api_client_user(required_roles: list[str] = None) -> Callable[[], UserData]:
    """BYPASSED: Always return mock user for API clients"""
    async def api_client_user(
        access_token: Optional[str] = Depends(reusable_oauth2),
        db: AsyncSession = Depends(get_db),
    ) -> UserData:
        logger.debug(f"BYPASSED get_api_client_user with roles {required_roles}")
        return MOCK_USER_DATA

    return api_client_user


def get_current_user_or_api_client(required_roles: list[str] = None) -> Callable[[], UserData]:
    """BYPASSED: Always return mock user for both user and API client auth"""
    async def current_user_or_api_client(
        access_token: Optional[str] = Depends(reusable_oauth2),
        redis_client: Redis = Depends(get_redis_client),
        db: AsyncSession = Depends(get_db),
    ) -> UserData:
        logger.debug(f"BYPASSED get_current_user_or_api_client with roles {required_roles}")
        return MOCK_USER_DATA

    return current_user_or_api_client


# Additional utility functions that might be needed
async def check_loginradius_errors(response):
    """BYPASSED: No login errors to check"""
    return True


def get_instructor_model(
    model: ChatModelEnum = ChatModelEnum.gpt_4o_mini,
) -> instructor.AsyncInstructor:
    """Get instructor model for structured data extraction."""
    if model == ChatModelEnum.gpt_4o:
        provider = OpenAIProvider(
            api_key=settings.AZURE_OPENAI_API_KEY,
            base_url=settings.AZURE_BASE_URL,
        )
        openai_model = OpenAIModel(model_name="gpt-4o", openai_client=provider.client)
    else:
        provider = OpenAIProvider(
            api_key=settings.AZURE_OPENAI_API_KEY,
            base_url=settings.AZURE_BASE_URL,
        )
        openai_model = OpenAIModel(model_name="gpt-4o-mini", openai_client=provider.client)

    model = instructor.from_openai(openai_model.openai_client)
    return model


def get_openai_model(
    model_name: str = ChatModelEnum.gpt_4o_mini,
    api_key: str = settings.AZURE_OPENAI_API_KEY,
    base_url: str = settings.AZURE_BASE_URL,
) -> openai.AsyncOpenAI:
    """Get OpenAI model."""
    return openai.AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def get_tokenizer(model: str = "gpt-4o-mini") -> tiktoken.Encoding:
    """Get tokenizer for the model."""
    return tiktoken.encoding_for_model(model)


def get_tavily_client() -> TavilyClient:
    """Get Tavily client for web search."""
    return TavilyClient(api_key=settings.BRAVE_SEARCH_API_KEY)