from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Callable, Optional, Union
from uuid import UUID

import instructor
import openai
import redis
import redis.asyncio as aioredis
import tiktoken
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from fastapi import Depends, HTTPException, Path, status
from fastapi.security import OAuth2PasswordBearer
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from redis.asyncio import Redis, RedisCluster
from redis.cluster import ClusterNode
from sqlmodel.ext.asyncio.session import AsyncSession
from tavily import TavilyClient

from app import crud
from app.be_core.api_client_security import (
    build_api_client_user_data,
    is_api_client_token,
    validate_api_client_token,
)
from app.be_core.config import settings
from app.be_core.logger import logger
from app.be_core.security import (
    amplifi_authentication,
    cache_get_user,
    is_amplifi_generated_token,
    loginradius_authentication,
    microsoft_authentication,
)
from app.db.session import SessionLocal, SessionLocalCelery
from app.schemas.common_schema import IMetaGeneral
from app.schemas.rag_generation_schema import ChatModelEnum
from app.schemas.user_schema import UserData

# Azure Key Vault URL
key_vault_url = f"https://{settings.AZURE_KEY_VAULT_NAME}.vault.azure.net"

reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V2_STR}/login/access-token"
)

async_redis_client: Optional[Union[Redis, RedisCluster]] = None


def get_secret_client() -> SecretClient:
    """
    Dependency to provide an instance of Azure Key Vault SecretClient.
    This is now synchronous because SecretClient doesn't support async.
    """
    credential = DefaultAzureCredential()
    secret_client = SecretClient(vault_url=key_vault_url, credential=credential)
    return secret_client


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


async def organization_check(
    organization_id: UUID = Path(...),
    access_token: str = Depends(reusable_oauth2),
    redis_client: Redis = Depends(get_redis_client),
):
    logger.debug(f"organization_check for {organization_id}")
    cached_value = await cache_get_user(access_token, redis_client)
    if cached_value:
        user_data = cached_value.user_data

        # Check if user belongs to an organization
        if organization_id and user_data.organization_id != organization_id:
            logger.warning(
                f"Unauthorized user {user_data.id} trying to perform action on organization {organization_id}"
            )
            raise HTTPException(
                status_code=403,
                detail="You are not authorized to perform this action",
            )


async def workspace_check(
    workspace_id: UUID = Path(...),
    access_token: str = Depends(reusable_oauth2),
    redis_client: Redis = Depends(get_redis_client),
    db: AsyncSession = Depends(get_db),
):
    logger.debug(f"workspace_check for {workspace_id}")
    cached_value = await cache_get_user(access_token, redis_client)
    if cached_value:
        user_data = cached_value.user_data

        if workspace_id:
            try:
                organization_id = await crud.workspace.get_organization_id_of_workspace(
                    workspace_id=workspace_id, db_session=db
                )
            except HTTPException:
                logger.warning(
                    f"User {user_data.id} trying to perform action on workspace {workspace_id}, which doesn't exist. Returning 404"
                )
                raise HTTPException(
                    status_code=404,
                    detail="You are not authorized to perform this action",
                )

            # Check if user belongs to an organization
            if user_data.organization_id != organization_id:
                logger.warn(
                    f"Unauthorized user {user_data.id} trying to perform action on organization {organization_id}"
                )
                raise HTTPException(
                    status_code=403,
                    detail="You are not authorized to perform this action",
                )


async def check_loginradius_errors(response):
    if response.get("ErrorCode") == 991:
        raise HTTPException(status_code=400, detail="Your account is blocked")
    elif response.get("ErrorCode") == 970:
        raise HTTPException(status_code=400, detail=response.get("Description"))
    elif "access_token" not in response:
        raise HTTPException(status_code=400, detail="Invalid credentials")


async def user_workspace_access_check(
    workspace_id: UUID = Path(...),
    access_token: str = Depends(reusable_oauth2),
    redis_client: Redis = Depends(get_redis_client),
    db: AsyncSession = Depends(get_db),
):
    logger.debug(f"Checking workspace access for {workspace_id}")
    cached_value = await cache_get_user(access_token, redis_client)
    if cached_value:
        user_data = cached_value.user_data

        await workspace_check(
            workspace_id=workspace_id,
            access_token=access_token,
            redis_client=redis_client,
            db=db,
        )
        # check if user can access the workspace
        if not any(role in user_data.role for role in ["Ampli Admin", "Amplifi_Admin"]):
            workspace_link = await crud.workspace.user_belongs_to_workspace(
                user_id=user_data.id, workspace_id=workspace_id
            )

            if not workspace_link:
                logger.warning(
                    f"User {user_data.id} does not have access to workspace {workspace_id}. Raising 403."
                )
                raise HTTPException(
                    status_code=403,
                    detail="You are not authorized to perform this action",
                )


async def dataset_check(
    dataset_id: UUID = Path(...),
    access_token: str = Depends(reusable_oauth2),
    redis_client: Redis = Depends(get_redis_client),
    db: AsyncSession = Depends(get_db),
):
    logger.debug(f"Dataset check for {dataset_id}")
    cached_value = await cache_get_user(access_token, redis_client)

    if cached_value:
        user_data = cached_value.user_data
        if dataset_id:
            try:
                workspace_id = await crud.dataset_v2.get_workspace_id_of_dataset(
                    dataset_id=dataset_id
                )
            except HTTPException:
                logger.warning(
                    f"User {user_data.id} trying to perform action on dataset {dataset_id}, which doesn't exist. Returning 404."
                )
                raise HTTPException(
                    status_code=404,
                    detail="You are not authorized to perform this action",
                )
            await user_workspace_access_check(
                workspace_id=workspace_id,
                access_token=access_token,
                redis_client=redis_client,
                db=db,
            )


async def chatapp_check(
    chatapp_id: UUID = Path(...),
    access_token: str = Depends(reusable_oauth2),
    redis_client: Redis = Depends(get_redis_client),
    db: AsyncSession = Depends(get_db),
):
    logger.debug(f"chatapp check for {chatapp_id}")
    cached_value = await cache_get_user(access_token, redis_client)
    if cached_value:
        user_data = cached_value.user_data
        if chatapp_id:
            try:
                workspace_id = await crud.chatapp.get_workspace_id_of_chatapp(
                    chatapp_id=chatapp_id, db_session=db
                )
            except HTTPException:
                logger.warning(
                    f"User {user_data.id} trying to perform action on chatapp {chatapp_id}, which doesn't exist. Returning 404."
                )
                raise HTTPException(
                    status_code=404,
                    detail="You are not authorized to perform this action",
                )
            await user_workspace_access_check(
                workspace_id=workspace_id,
                access_token=access_token,
                redis_client=redis_client,
                db=db,
            )


async def chatsession_check(
    chat_session_id: UUID = Path(...),
    chatapp_id: Optional[UUID] = Path(...),
    access_token: str = Depends(reusable_oauth2),
    redis_client: Redis = Depends(get_redis_client),
    db: AsyncSession = Depends(get_db),
):
    logger.debug(f"chatsession check for {chat_session_id}")
    cached_value = await cache_get_user(access_token, redis_client)
    if cached_value:
        user_data = cached_value.user_data
        if chat_session_id:
            chat_session_record = await crud.chatsession.get_chat_session_by_id(
                chat_session_id=chat_session_id,
                chatapp_id=chatapp_id,
            )
            if not chat_session_record:
                raise HTTPException(status_code=404, detail="Chat Session not found")
            if chat_session_record.user_id != user_data.id:
                raise HTTPException(
                    status_code=403,
                    detail=f"Chat Session does not belong to you (user {user_data.id}) .",
                )
            pulled_chatapp_id = chat_session_record.chatapp_id
            if not pulled_chatapp_id:
                logger.warning(
                    f"User {user_data.id} trying to perform action on chatsession {chat_session_id}, which doesn't exist. Returning 403."
                )
                raise HTTPException(
                    status_code=403,
                    detail="You are not authorized to perform this action",
                )
            await chatapp_check(
                chatapp_id=pulled_chatapp_id,
                access_token=access_token,
                redis_client=redis_client,
                db=db,
            )
            if chatapp_id and chatapp_id != pulled_chatapp_id:
                logger.warning(
                    f"User {user_data.id} tried to access Chatsession {chat_session_id} via chatapp {chatapp_id}, but the chatsession is in Chatapp {pulled_chatapp_id}. Returning 400."
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Chatsession {chat_session_id} is in Chatapp {pulled_chatapp_id}, not given chatapp {chatapp_id}.",
                )


async def get_authenticated_userdata(
    access_token, redis_client: Redis, required_roles: list[str] = None
) -> UserData:
    if is_amplifi_generated_token(access_token=access_token):
        return await amplifi_authentication(
            access_token=access_token,
            redis_client=redis_client,
            required_roles=required_roles,
        )
    elif settings.LOGIN_APP == "ONGC":
        return await microsoft_authentication(
            access_token=access_token,
            redis_client=redis_client,
            required_roles=required_roles,
        )
    else:
        return await loginradius_authentication(
            access_token=access_token,
            redis_client=redis_client,
            required_roles=required_roles,
        )


async def get_api_client_userdata(
    access_token: str, db: AsyncSession, required_roles: list[str] = None
) -> UserData:
    """Authenticate API client and return UserData"""
    # Validate the API client token
    payload = validate_api_client_token(access_token)

    # Get the API client from database
    api_client = await crud.api_client.get_by_client_id(
        client_id=payload["sub"], db_session=db
    )

    if not api_client:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API client not found",
        )

    if api_client.expires_at and api_client.expires_at < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API client has expired",
        )

    # Check required roles if specified
    if required_roles and "api_client" not in required_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Role {required_roles} is required for this action",
        )

    return build_api_client_user_data(api_client)


def get_current_user(required_roles: list[str] = None) -> Callable[[], UserData]:
    async def current_user(
        access_token: str = Depends(reusable_oauth2),
        redis_client: Redis = Depends(get_redis_client),
        db: AsyncSession = Depends(get_db),
    ) -> UserData:
        return await get_authenticated_userdata(
            access_token, redis_client, required_roles
        )

    return current_user


def get_api_client_user(required_roles: list[str] = None) -> Callable[[], UserData]:
    """Dependency for API client authentication"""

    async def api_client_user(
        access_token: str = Depends(reusable_oauth2),
        db: AsyncSession = Depends(get_db),
    ) -> UserData:
        return await get_api_client_userdata(access_token, db, required_roles)

    return api_client_user


def get_current_user_or_api_client(
    required_roles: list[str] = None,
) -> Callable[[], UserData]:
    """Dependency that supports both regular user and API client authentication"""

    async def current_user_or_api_client(
        access_token: str = Depends(reusable_oauth2),
        redis_client: Redis = Depends(get_redis_client),
        db: AsyncSession = Depends(get_db),
    ) -> UserData:
        # Check if this is an API client token
        if is_api_client_token(access_token):
            return await get_api_client_userdata(access_token, db, required_roles)
        else:
            return await get_authenticated_userdata(
                access_token, redis_client, required_roles
            )

    return current_user_or_api_client


azure_endpoint = settings.AZURE_BASE_URL
azure_api_key = settings.AZURE_OPENAI_API_KEY
gpt_deployment = "gpt-35-turbo"
azure_api_version = settings.AZURE_API_VERSION

azure_client = openai.AzureOpenAI(
    azure_endpoint=azure_endpoint, api_key=azure_api_key, api_version=azure_api_version
)

async_azure_client = openai.AsyncAzureOpenAI(
    azure_endpoint=azure_endpoint, api_key=azure_api_key, api_version=azure_api_version
)

azure_instructor_client = instructor.from_openai(azure_client)


def get_async_azure_client() -> openai.AsyncAzureOpenAI:
    return async_azure_client


def get_azure_client() -> openai.AzureOpenAI:
    return azure_client


def get_instructor_client() -> instructor.Instructor:
    return azure_instructor_client


gpt4o_endpoint = settings.AZURE_GPT_4o_URL
gpt4o_key = settings.AZURE_GPT_4o_KEY
gpt4o_deployment = "gpt-4o"  #### this is standard gpt4o api
gpt4o_version = settings.AZURE_GPT_4o_VERSION

gpt4o_endpoint_batch = settings.AZURE_GPT_4o_URL_BATCH
gpt4o_key_batch = settings.AZURE_GPT_4o_KEY_BATCH
gpt4o_deployment_batch = "gpt-4o-2"  #### this is batch gpt4o name for Amplifi. Change to gpt-4o-Global-batch if using on ONGC
gpt4o_version_batch = settings.AZURE_GPT_4o_VERSION_BATCH

# blob_sas_token_rag = settings.BLOB_SAS_TOKEN_RAG
# blob_url_rag = settings.BLOB_URL_RAG

gpt4o_client = openai.AzureOpenAI(
    azure_endpoint=gpt4o_endpoint,
    api_key=gpt4o_key,
    api_version=gpt4o_version,
    azure_deployment=gpt4o_deployment,
)

if gpt4o_endpoint_batch and gpt4o_key_batch and gpt4o_version_batch:
    gpt4o_client_batch = openai.AzureOpenAI(
        azure_endpoint=gpt4o_endpoint_batch,
        api_key=gpt4o_key_batch,
        api_version=gpt4o_version_batch,
    )
    logger.debug("gpt4o_client_batch initialized")
else:
    gpt4o_client_batch = None
    logger.debug("gpt4o_client_batch is not initialized due to missing variables")


gpt4o_client_async = openai.AsyncAzureOpenAI(
    azure_endpoint=gpt4o_endpoint, api_key=gpt4o_key, api_version=gpt4o_version
)


def get_model_client(
    lm_model: str,
) -> Union[openai.AzureOpenAI, openai.AsyncAzureOpenAI]:
    """
    Returns the appropriate OpenAI client based on the provided model name.
    """
    if lm_model == ChatModelEnum.GPT4o:
        return gpt4o_client_async
    elif lm_model == ChatModelEnum.GPT35:
        return gpt35_client_async
    elif lm_model == ChatModelEnum.GPTo3:
        return gpto3mini_client_async
    elif lm_model == ChatModelEnum.GPT41:
        return gpt41_client_async
    elif lm_model == ChatModelEnum.GPT5:
        return gpt5_client_async
    else:
        raise ValueError(f"Unsupported model: {lm_model}")


def get_llm_model_name(lm_model: ChatModelEnum) -> str:
    if lm_model == ChatModelEnum.GPT4o:
        return "gpt-4o"
    elif lm_model == ChatModelEnum.GPT35:
        return "gpt-35-turbo"
    elif lm_model == ChatModelEnum.GPTo3:
        return "o3-mini"
    elif lm_model == ChatModelEnum.GPT41:
        return "gpt-4.1"
    elif lm_model == ChatModelEnum.GPT5:
        return "gpt-5"
    else:
        raise ValueError(f"Unsupported model: {lm_model}")


def get_gpt4o_client() -> openai.AzureOpenAI:
    return gpt4o_client


def get_gpt4o_client_batch() -> openai.AzureOpenAI:
    return gpt4o_client_batch


def get_async_gpt4o_client() -> openai.AsyncAzureOpenAI:
    return gpt4o_client_async


def get_async_gpto3_client() -> openai.AsyncAzureOpenAI:
    return gpto3mini_client_async


gpt41_client = openai.AzureOpenAI(
    azure_endpoint=settings.AZURE_GPT_41_URL,
    api_key=settings.AZURE_GPT_41_KEY,
    api_version=settings.AZURE_GPT_41_VERSION,
    azure_deployment=settings.AZURE_GPT_41_DEPLOYMENT_NAME,
)

gpt41_client_async = openai.AsyncAzureOpenAI(
    azure_endpoint=settings.AZURE_GPT_41_URL,
    api_key=settings.AZURE_GPT_41_KEY,
    api_version=settings.AZURE_GPT_41_VERSION,
)

gpt35_client = openai.AzureOpenAI(
    azure_endpoint=settings.AZURE_BASE_URL,
    api_key=settings.AZURE_OPENAI_API_KEY,
    api_version=settings.AZURE_API_VERSION,
    azure_deployment=settings.AZURE_GPT_35_DEPLOYMENT_NAME,
)

gpt35_client_async = openai.AsyncAzureOpenAI(
    azure_endpoint=settings.AZURE_BASE_URL,
    api_key=settings.AZURE_OPENAI_API_KEY,
    api_version=settings.AZURE_API_VERSION,
)


gpto3mini_client = openai.AzureOpenAI(
    azure_endpoint=settings.AZURE_GPT_o3_URL,
    api_key=settings.AZURE_GPT_o3_KEY,
    api_version=settings.AZURE_GPT_o3_VERSION,
    azure_deployment=settings.AZURE_GPT_o3_DEPLOYMENT_NAME,
)

gpto3mini_client_async = openai.AsyncAzureOpenAI(
    azure_endpoint=settings.AZURE_GPT_o3_URL,
    api_key=settings.AZURE_GPT_o3_KEY,
    api_version=settings.AZURE_GPT_o3_VERSION,
    azure_deployment=settings.AZURE_GPT_o3_DEPLOYMENT_NAME,
)

gpt5_client = openai.AzureOpenAI(
    azure_endpoint=settings.AZURE_GPT_41_URL,
    api_key=settings.AZURE_GPT_41_KEY,
    api_version=settings.AZURE_GPT_5_VERSION,
    azure_deployment=settings.AZURE_GPT_5_DEPLOYMENT_NAME,
)

gpt5_client_async = openai.AsyncAzureOpenAI(
    azure_endpoint=settings.AZURE_GPT_41_URL,
    api_key=settings.AZURE_GPT_41_KEY,
    api_version=settings.AZURE_GPT_5_VERSION,
    azure_deployment=settings.AZURE_GPT_5_DEPLOYMENT_NAME,
)


def get_gpto3_client() -> openai.AzureOpenAI:
    return gpto3mini_client


def get_gpto3_client_async() -> openai.AsyncAzureOpenAI:
    return gpto3mini_client_async


def get_gpt35_client() -> openai.AzureOpenAI:
    return gpt35_client


def get_gpt41_client() -> openai.AzureOpenAI:
    return gpt41_client


def get_gpt41_client_async() -> openai.AsyncAzureOpenAI:
    return gpt41_client_async


def get_gpt5_client() -> openai.AzureOpenAI:
    return gpt5_client


def get_gpt5_client_async() -> openai.AsyncAzureOpenAI:
    return gpt5_client_async


def get_numb_tokens(
    input_string: str, enc=tiktoken.encoding_for_model(gpt_deployment)
) -> int:
    """Returns number of tokens in a given string based on gpt_deployment as defined"""
    return len(enc.encode(input_string))


def get_numb_tokens4o(
    input_string: str, enc=tiktoken.encoding_for_model(gpt4o_deployment)
) -> int:
    """Returns number of tokens in a given string based on gpt_deployment as defined"""
    return len(enc.encode(input_string))


def get_filename(file_name: str) -> str:
    return file_name


def publish_websocket_message(channel: str, msg: str):
    redis_client = get_redis_client_sync()
    redis_client.publish(channel, msg)
    redis_client.close()


if settings.TAVILY_API_KEY:
    tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)
else:
    logger.warning("No Tavily API key found. Tavily client will not be initialized.")
    tavily_client = None


def get_tavily_client() -> Optional[TavilyClient]:
    return tavily_client


pydantic_ai_model_o3 = OpenAIModel(
    "o3-mini",
    provider=OpenAIProvider(
        openai_client=gpt41_client_async
    ),  # o3 mini is hosted on same client as 41 instance
)
