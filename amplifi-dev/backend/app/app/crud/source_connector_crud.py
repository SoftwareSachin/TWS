import asyncio
import base64
import io
import json
import os
import re
import ssl
from datetime import datetime
from typing import List, Optional, Type, Union
from urllib.parse import unquote
from uuid import UUID

import aioboto3
import aiofiles
import aiomysql
import asyncpg
import fitz  # PyMuPDF
import httpx  # ### ADDED FOR GROOVE ### - For making API requests
import hvac
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.keyvault.secrets import SecretClient
from azure.storage.blob.aio import BlobServiceClient
from botocore.exceptions import ClientError, NoCredentialsError
from fastapi import HTTPException, UploadFile
from fastapi_pagination import Page, Params, paginate
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from sqlalchemy import asc, delete, desc, func, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload
from sqlmodel import select

from app.api import deps
from app.be_core.celery import celery
from app.be_core.config import settings
from app.be_core.logger import logger
from app.crud.base_crud import CRUDBase
from app.models.file_model import File
from app.models.source_model import AWSS3Storage as S3Model
from app.models.source_model import (
    AzureFabric,
)
from app.models.source_model import AzureStorage as AzureModel
from app.models.source_model import (
    GrooveSource as GrooveModel,  # ### ADDED FOR GROOVE ###
)
from app.models.source_model import MySQLSource as MySQLModel
from app.models.source_model import (
    PGVector,
    Source,
)
from app.schemas.common_schema import IOrderEnum
from app.schemas.file_schema import FileStatusEnum
from app.schemas.source_schema import GrooveSourceResponse  # ### ADDED FOR GROOVE ###
from app.schemas.source_schema import IGrooveSource  # ### ADDED FOR GROOVE ###
from app.schemas.source_schema import (  # IPGVectorSource,; IMySQLSource,
    AWSS3SourceResponse,
    AzureFabricSourceResponse,
    AzureStorageSourceResponse,
    IAWSS3Source,
    IAzureFabricSource,
    IAzureStorageSource,
    ISourceConnectorResponse,
    ISourceUpdate,
    ISQLSource,
    MySQLSourceResponse,
    PGVectorSourceResponse,
)
from app.utils.feature_flags import is_video_ingestion_enabled
from app.utils.optional_params import OptionalParams
from app.utils.uuid6 import uuid7

timeout = 10
vault_client = hvac.Client(url=settings.VAULT_ADDR, token=settings.VAULT_TOKEN)


# Add this after other global variables
Image.MAX_IMAGE_PIXELS = None


# Helper function to build MySQL SSL config
def build_mysql_ssl_config(
    ssl_mode: Optional[str] = None,
) -> Optional[Union[ssl.SSLContext, dict]]:
    """
    Build MySQL SSL configuration based on ssl_mode.

    Returns:
        SSL configuration for aiomysql (SSLContext or dict)
    """
    if ssl_mode == "disabled":
        # Explicitly disable SSL
        return None

    elif ssl_mode == "required":
        # Create SSL context with encryption but no certificate verification
        # This should work with servers that have --require_secure_transport=ON
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        return ssl_ctx

    else:
        # Default to required SSL for unknown modes
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        return ssl_ctx


def make_safe_secret_name(source_id: UUID, source_type: str) -> str:
    safe_type = re.sub(r"[^a-zA-Z0-9-]", "-", source_type.replace("_", "-"))
    return f"source-{source_id}-{safe_type}-credentials"


# helper function (outside)
async def try_postgres_connection(dsn: str) -> tuple[bool, str]:
    try:
        conn = await asyncpg.connect(dsn=dsn)
        await conn.close()
        return True, "PostgreSQL connection successful"
    except asyncpg.InvalidPasswordError:
        return False, "PostgreSQL: Invalid username or password"
    except asyncpg.InvalidCatalogNameError:
        return False, "PostgreSQL: Invalid database name"
    except asyncpg.PostgresError as e:
        return False, f"PostgreSQL error: {str(e)}"


# helper function (outside)
async def try_mysql_connection(
    host: str,
    port: int,
    username: str,
    password: str,
    database_name: str,
    ssl_config: Optional[Union[ssl.SSLContext, dict]] = None,
) -> tuple[bool, str]:
    """
    Test MySQL connection with SSL support.

    Args:
        host: Database host
        port: Database port
        username: Database username
        password: Database password
        database_name: Database name
        ssl_config: SSL configuration (SSLContext or dict)

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        conn = await aiomysql.connect(
            host=host,
            port=port,
            user=username,
            password=password,
            db=database_name,
            ssl=ssl_config,
            autocommit=True,  # Add autocommit for better connection handling
        )

        # Test the connection with a simple query
        cursor = await conn.cursor()
        await cursor.execute("SELECT 1")
        await cursor.fetchone()
        await cursor.close()

        conn.close()

        return True, "MySQL connection successful"

    except aiomysql.OperationalError as e:
        error_code = e.args[0] if e.args else 0
        msg = str(e)

        # Handle specific MySQL error codes
        if error_code == 1045:  # Access denied
            return False, "MySQL: Invalid username or password"
        elif error_code == 1049:  # Unknown database
            return False, "MySQL: Invalid database name"
        elif error_code == 2003:  # Can't connect to server
            return False, "MySQL: Host or port unreachable"
        elif error_code == 3159:  # Secure transport required
            return False, "MySQL: Secure transport required. Enable SSL connection."
        elif "SSL" in msg or "certificate" in msg or "secure" in msg.lower():
            return False, f"MySQL SSL error: {msg}"
        else:
            return False, f"MySQL operational error: {msg}"

    except Exception as e:
        error_msg = str(e).lower()
        if "ssl" in error_msg or "certificate" in error_msg or "secure" in error_msg:
            return False, f"MySQL SSL error: {str(e)}"
        return False, f"MySQL connection error: {str(e)}"


class CRUDSource(CRUDBase[Source, IAzureStorageSource | IAWSS3Source, ISourceUpdate]):

    ### ADDED FOR GROOVE ###
    async def check_groove_connection(self, *, api_key: str) -> tuple[bool, str]:
        """
        Checks the connection to the Groove API using an API key.
        """
        if not api_key:
            return False, "Groove API key cannot be empty."

        try:
            async with httpx.AsyncClient() as client:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }

                # This endpoint only accepts POST with GraphQL body — GET will return 404
                test_query = """
                query q($n: Int!) {
                    conversation(number: $n) {
                        id
                        number
                        subject
                        state
                    }
                }
                """
                variables = {"n": 1}

                response = await client.post(
                    settings.GROOVE_API_URL,
                    headers=headers,
                    json={"query": test_query, "variables": variables},
                )

                if response.status_code == 200:
                    data = response.json()
                    if "errors" in data:
                        return (
                            False,
                            f"Groove API responded, but query failed: {data['errors'][0].get('message', 'Unknown error')}",
                        )
                    return True, "Groove API connection successful."
                elif response.status_code == 401:
                    return False, "Groove API connection failed: Invalid API Key."
                else:
                    return (
                        False,
                        f"Groove API connection failed with status: {response.status_code}.",
                    )

        except httpx.RequestError as e:
            logger.error(f"Error connecting to Groove API: {e}")
            return (
                False,
                "Could not connect to Groove API. Check your network connection.",
            )

        except Exception as e:
            logger.error(
                f"Unexpected error connecting to Groove API: {e}", exc_info=True
            )
            return (
                False,
                "An unexpected error occurred while connecting to the Groove API.",
            )

    async def _load_db_model(
        self, source_type: str, source_id: UUID, db_session: AsyncSession
    ):
        if source_type == "pg_db":
            db_model = await self._get_pg_db(source_id, db_session)
            logger.debug(f"[pg_vector] PGVector model loaded for source {source_id}")
            return db_model
        elif source_type == "mysql_db":
            result = await db_session.execute(
                select(MySQLModel).where(MySQLModel.source_id == source_id)
            )
            db_model = result.scalar_one_or_none()
            if not db_model:
                logger.error(
                    f"[mysql_db] MySQL model not found for source_id={source_id}"
                )
                raise HTTPException(status_code=404, detail="MySQL source not found.")
            logger.debug(f"[mysql_db] MySQL model loaded for source {source_id}")
            return db_model
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported source type: {source_type}"
            )

    async def _update_credentials_in_vault(self, obj_in, source_id, source_type):
        try:
            updated_credentials = await self._prepare_updated_credentials(
                obj_in, source_id, source_type
            )
            await self._store_credentials(source_id, source_type, updated_credentials)
            logger.info(f"[{source_type}] Credentials updated successfully in Vault.")
        except Exception as e:
            logger.exception(
                f"[{source_type}] Failed to update credentials in Vault: {str(e)}"
            )
            raise HTTPException(
                status_code=500, detail="Failed to update credentials in Vault."
            )

    def _update_model_fields(self, source_type, db_model, obj_in):
        logger.info("we are in the _update_model_field")
        logger.info(
            f"the source type we get '{source_type}' (type: {type(source_type)})"
        )
        logger.info(f"Comparison result: {source_type == 'pg_db'}")
        if source_type == "pg_db":
            self._update_pg_db_fields(db_model, obj_in)
        else:
            db_model.host = obj_in.host or db_model.host
            db_model.port = obj_in.port or db_model.port
            db_model.database_name = obj_in.database_name or db_model.database_name

            # Handle SSL fields
            db_model.ssl_mode = obj_in.ssl_mode or db_model.ssl_mode

    async def _commit_update(
        self, db_session, db_model, source, source_type, source_id
    ):
        try:
            await db_session.commit()
            await db_session.refresh(db_model)
            await db_session.refresh(source)
            logger.info(
                f"[{source_type}] Update committed successfully for source_id={source_id}"
            )
        except Exception as e:
            logger.exception(
                f"[{source_type}] Failed to commit updates to DB: {str(e)}"
            )
            raise HTTPException(
                status_code=500, detail="Failed to update source in the database."
            )

    async def get_by_id(
        self,
        *,
        workspace_id: UUID,
        source_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> Source | None:
        db_session = db_session or super().get_db().session
        result = await db_session.execute(
            select(Source).where(
                Source.id == source_id,
                Source.workspace_id == workspace_id,
                Source.deleted_at.is_(None),
            )
        )
        source = result.scalars().first()
        return source

    async def get_azure_storage_details(
        self,
        source_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> dict | None:
        db_session = db_session or super().get_db().session
        source_result = await db_session.execute(
            select(Source).where(Source.id == source_id, Source.deleted_at.is_(None))
        )
        source = source_result.scalar_one_or_none()

        if not source:
            raise HTTPException(status_code=404, detail="Source not found or deleted.")

        result = await db_session.execute(
            select(AzureModel).where(AzureModel.source_id == source_id)
        )
        azure_storage = result.scalars().first()

        if not azure_storage:
            return None

        try:
            secret_client: SecretClient = deps.get_secret_client()
            vault_reference = azure_storage.sas_url
            if vault_reference.startswith("vault:"):
                vault_path = vault_reference.replace("vault:", "")
                if settings.DEPLOYED_ENV.startswith("azure"):
                    secret = secret_client.get_secret(vault_path)
                    actual_sas_url = secret.value
                else:
                    secret = vault_client.secrets.kv.v2.read_secret_version(
                        path=vault_path
                    )
                    actual_sas_url = secret["data"]["data"]["sas_url"]
            else:
                actual_sas_url = azure_storage.sas_url
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to fetch SAS URL: {str(e)}"
            )

        return {
            "container_name": azure_storage.container_name,
            "sas_url": actual_sas_url,
        }

    async def get_azure_fabric_details(
        self, source_id: UUID, db_session: AsyncSession | None = None
    ) -> dict | None:
        """
        Fetch Azure Fabric storage details for a given source ID.
        Resolves SAS URL credentials from Azure Key Vault if required.
        """
        db_session = db_session or super().get_db().session

        source_result = await db_session.execute(
            select(Source).where(Source.id == source_id, Source.deleted_at.is_(None))
        )
        source = source_result.scalar_one_or_none()

        if not source:
            raise HTTPException(status_code=404, detail="Source not found or deleted.")

        result = await db_session.execute(
            select(AzureFabric).where(AzureFabric.source_id == source_id)
        )
        azure_fabric = result.scalars().first()

        if not azure_fabric:
            return None

        try:
            secret_client: SecretClient = deps.get_secret_client()
            vault_reference = azure_fabric.sas_url
            if vault_reference.startswith("vault:"):
                vault_path = vault_reference.replace("vault:", "")
                if settings.DEPLOYED_ENV.startswith("azure"):
                    secret = secret_client.get_secret(vault_path)
                    actual_sas_url = secret.value
                else:
                    secret = vault_client.secrets.kv.v2.read_secret_version(
                        path=vault_path
                    )
                    actual_sas_url = secret["data"]["data"]["sas_url"]
            else:
                actual_sas_url = azure_fabric.sas_url
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch Azure Fabric SAS URL: {str(e)}",
            )

        return {
            "container_name": azure_fabric.container_name,
            "sas_url": actual_sas_url,
        }

    async def get_s3_storage_details(
        self,
        source_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> dict | None:
        db_session = db_session or super().get_db().session

        result = await db_session.execute(
            select(S3Model).where(S3Model.source_id == source_id)
        )
        s3_storage = result.scalars().first()

        if not s3_storage:
            logger.error(f"S3 storage details not found for source_id: {source_id}")
            return None

        try:
            secret_client: SecretClient = deps.get_secret_client()

            access_key_reference = s3_storage.access_id
            secret_key_reference = s3_storage.access_secret

            if access_key_reference.startswith(
                "vault:"
            ) and secret_key_reference.startswith("vault:"):
                access_key_path = unquote(
                    access_key_reference.replace("vault:", "").split(":")[0]
                )
                secret_key_path = unquote(
                    secret_key_reference.replace("vault:", "").split(":")[0]
                )

                if settings.DEPLOYED_ENV.startswith("azure_"):
                    try:
                        access_key = json.loads(
                            secret_client.get_secret(access_key_path).value
                        ).get("access_key")
                        secret_key = json.loads(
                            secret_client.get_secret(secret_key_path).value
                        ).get("secret_key")
                    except Exception as azure_error:
                        logger.error(
                            f"Error while fetching secrets from Azure Key Vault: {str(azure_error)}"
                        )
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to fetch S3 credentials from Azure Key Vault: {str(azure_error)}",
                        )
                else:
                    try:
                        access_key_response = (
                            vault_client.secrets.kv.v2.read_secret_version(
                                path=access_key_path
                            )
                        )
                        secret_key_response = (
                            vault_client.secrets.kv.v2.read_secret_version(
                                path=secret_key_path
                            )
                        )

                        access_key = access_key_response["data"]["data"].get(
                            "access_key"
                        )
                        secret_key = secret_key_response["data"]["data"].get(
                            "secret_key"
                        )

                    except Exception as vault_error:
                        logger.error(
                            f"Error while fetching secrets from Vault: {str(vault_error)}"
                        )
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to fetch S3 credentials from Vault: {str(vault_error)}",
                        )

            else:
                logger.error(
                    f"Invalid Vault reference. Access key reference: {access_key_reference}, Secret key reference: {secret_key_reference}"
                )
                raise HTTPException(
                    status_code=400,
                    detail="Invalid Vault reference. Expected 'vault:' prefix.",
                )

            return {
                "bucket_name": s3_storage.bucket_name,
                "access_key": access_key,
                "secret_key": secret_key,
            }

        except Exception as e:
            logger.error(f"Error while fetching S3 credentials: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to fetch S3 credentials: {str(e)}"
            )

    ### ADDED FOR GROOVE ###
    async def get_groove_details(
        self, source_id: UUID, db_session: AsyncSession
    ) -> dict | None:
        """
        Fetches connection details for a Groove source.
        Uses Azure Key Vault if DEPLOYED_ENV starts with 'azure_', else HashiCorp Vault,
        but only if the stored value is a vault reference.
        """
        db_session = db_session or super().get_db().session

        # Step 1: Get the Groove model from the database
        result = await db_session.execute(
            select(GrooveModel).where(GrooveModel.source_id == source_id)
        )
        groove_entry = result.scalars().first()

        if not groove_entry:
            logger.warning(
                f"Groove source details not found for source_id: {source_id}"
            )
            return None

        # Step 2: Resolve API key
        try:
            vault_reference = groove_entry.api_key_vault_path

            if vault_reference.startswith("vault:"):
                secret_path = vault_reference.replace("vault:", "")

                if settings.DEPLOYED_ENV.startswith("azure_"):
                    # Azure Key Vault
                    secret_client: SecretClient = deps.get_secret_client()
                    raw_secret = secret_client.get_secret(secret_path).value
                    credentials = json.loads(raw_secret)
                else:
                    # HashiCorp Vault
                    secret_data = vault_client.secrets.kv.v2.read_secret_version(
                        path=secret_path
                    )
                    credentials = secret_data["data"]["data"]

                api_key = credentials.get("api_key")
            else:
                # Direct value (not from vault)
                api_key = vault_reference

            if not api_key:
                raise ValueError("API key not found.")

        except Exception as e:
            logger.error(f"Failed to fetch Groove credentials: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch Groove credentials: {str(e)}",
            )

        return {"api_key": api_key}

    def get_groove_details_sync(self, source_id, db_session):
        """Synchronous version for use in Celery tasks and monitoring."""
        source = (
            db_session.query(Source)
            .filter(Source.id == source_id, Source.deleted_at.is_(None))
            .first()
        )
        if not source:
            raise Exception("Source not found or deleted.")

        groove_source = (
            db_session.query(GrooveModel)
            .filter(GrooveModel.source_id == source_id)
            .first()
        )
        if not groove_source:
            return None

        api_key_vault_path = groove_source.api_key_vault_path
        try:
            if api_key_vault_path.startswith("vault:"):
                vault_path = api_key_vault_path.replace("vault:", "")
                if settings.DEPLOYED_ENV.startswith("azure"):
                    secret_client: SecretClient = deps.get_secret_client()
                    secret = secret_client.get_secret(vault_path)
                    credentials = json.loads(secret.value)
                else:
                    secret = vault_client.secrets.kv.v2.read_secret_version(
                        path=vault_path
                    )
                    credentials = secret["data"]["data"]
            else:
                credentials = {"api_key": api_key_vault_path}

            return {"api_key": credentials["api_key"]}
        except Exception as e:
            raise Exception(f"Failed to fetch API key: {str(e)}")

    async def get_sql_db_details(
        self,
        source_id: UUID,
        db_type: str,
        db_session: AsyncSession | None = None,
    ) -> dict | None:
        """
        Fetches connection details and credentials for SQL DBs (PostgreSQL or MySQL).
        """
        db_session = db_session or super().get_db().session

        # Step 1: Check if source exists
        source_result = await db_session.execute(
            select(Source).where(Source.id == source_id, Source.deleted_at.is_(None))
        )
        source = source_result.scalar_one_or_none()

        if not source:
            raise HTTPException(status_code=404, detail="Source not found or deleted.")

        ssl_mode = None

        # Step 2: Load the corresponding DB model
        if db_type == "pg_db":
            result = await db_session.execute(
                select(PGVector).where(PGVector.source_id == source_id)
            )
            db_entry = result.scalars().first()
            secret_path = make_safe_secret_name(source_id, db_type)

        elif db_type == "mysql_db":
            result = await db_session.execute(
                select(MySQLModel).where(MySQLModel.source_id == source_id)
            )
            db_entry = result.scalars().first()

            # Pull the vault path from the vault-prefixed username string
            secret_path = (
                db_entry.username.replace("vault:", "").split(":")[0]
                if db_entry
                else None
            )
            logger.info("Retrieved MySQL DB configuration for source")
            ssl_mode = db_entry.ssl_mode if db_entry else None  # 👈 Extract ssl_mode
            logger.info(f"ssl_mode : {ssl_mode}")
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported db_type: {db_type}"
            )

        if not db_entry:
            return None

        # Step 3: Fetch credentials from secret store
        try:
            secret_client: SecretClient = deps.get_secret_client()

            if settings.DEPLOYED_ENV.startswith("azure"):
                raw_secret = secret_client.get_secret(secret_path).value
            else:
                secret_data = vault_client.secrets.kv.v2.read_secret_version(
                    path=secret_path
                )
                raw_secret = secret_data["data"]["data"]

            credentials = (
                json.loads(raw_secret) if isinstance(raw_secret, str) else raw_secret
            )

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch {db_type.upper()} credentials: {str(e)}",
            )

        connection_details = {
            "host": db_entry.host,
            "port": db_entry.port,
            "database_name": db_entry.database_name,
            "username": credentials.get("username"),
            "password": credentials.get("password"),
        }

        # ✅ Only add ssl_mode for MySQL
        if db_type == "mysql_db":
            connection_details["ssl_mode"] = ssl_mode

        return connection_details

    async def create_with_azure_storage(
        self,
        *,
        workspace_id: UUID,
        obj_in: IAzureStorageSource,
        db_session: AsyncSession = None,
    ) -> Source:
        secret_client: SecretClient = deps.get_secret_client()
        db_session = db_session or super().get_db().session

        source = Source(
            workspace_id=workspace_id,
            source_type=obj_in.source_type,
            created_at=datetime.now(),
        )
        db_session.add(source)
        await db_session.flush()

        try:
            try:
                decoded_sas_url = base64.b64decode(obj_in.sas_url).decode("utf-8")
            except Exception as decode_error:
                raise ValueError("Invalid SAS URL encoding.") from decode_error

            secret_name = f"source-{source.id}-sas-url"

            if settings.DEPLOYED_ENV.startswith("azure_"):
                secret_client.set_secret(secret_name, decoded_sas_url)
                logger.info("Decoded SAS URL stored successfully in Azure Key Vault.")
            else:
                vault_client.secrets.kv.v2.create_or_update_secret(
                    path=secret_name,
                    secret={"sas_url": decoded_sas_url},
                )
                logger.info("Decoded SAS URL stored successfully in HashiCorp Vault.")

        except Exception as e:
            logger.error(f"Failed to update the vault: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Failed to store SAS URL in the vault."
            )

        azure_storage = AzureModel(
            source_id=source.id,
            container_name=obj_in.container_name,
            sas_url=f"vault:{secret_name}",
        )
        db_session.add(azure_storage)
        await db_session.flush()
        await db_session.refresh(azure_storage)
        logger.info("Azure storage record created successfully.")

        return source

    async def create_with_s3_storage(
        self,
        *,
        workspace_id: UUID,
        obj_in: IAWSS3Source,
        db_session: AsyncSession = None,
    ) -> Source:
        secret_client: SecretClient = deps.get_secret_client()
        db_session = db_session or super().get_db().session

        source = Source(
            workspace_id=workspace_id,
            source_type=obj_in.source_type,
            created_at=datetime.now(),
        )
        db_session.add(source)
        await db_session.flush()

        try:
            try:
                # Decode access_id and access_secret
                decoded_access_id = base64.b64decode(obj_in.access_id).decode("utf-8")
                decoded_access_secret = base64.b64decode(obj_in.access_secret).decode(
                    "utf-8"
                )
            except Exception as decode_error:
                raise ValueError(
                    "Invalid encoding for access_id or access_secret."
                ) from decode_error

            access_secret_name = f"source-{source.id}-access-key"
            secret_secret_name = f"source-{source.id}-secret-key"

            if settings.DEPLOYED_ENV.startswith("azure_"):
                secret_client.set_secret(access_secret_name, decoded_access_id)
                logger.info("Access key stored successfully in Azure Key Vault.")

                secret_client.set_secret(secret_secret_name, decoded_access_secret)
                logger.info("Secret key stored successfully in Azure Key Vault.")
            else:
                vault_client.secrets.kv.v2.create_or_update_secret(
                    path=access_secret_name,
                    secret={"access_key": decoded_access_id},
                )
                logger.info("Access key stored successfully in HashiCorp Vault.")

                vault_client.secrets.kv.v2.create_or_update_secret(
                    path=secret_secret_name,
                    secret={"secret_key": decoded_access_secret},
                )
                logger.info("Secret key stored successfully in HashiCorp Vault.")
        except Exception as e:
            logger.error(f"Failed to update the vault: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Failed to store S3 credentials in the vault."
            )

        s3_storage = S3Model(
            source_id=source.id,
            bucket_name=obj_in.bucket_name,
            access_id=f"vault:{access_secret_name}:access_key",
            access_secret=f"vault:{secret_secret_name}:secret_key",
        )
        db_session.add(s3_storage)
        await db_session.flush()
        await db_session.refresh(s3_storage)

        return source

    async def create_with_azure_fabric(
        self,
        *,
        workspace_id: UUID,
        obj_in: IAzureFabricSource,
        db_session: AsyncSession = None,
    ) -> Source:
        secret_client: SecretClient = deps.get_secret_client()
        db_session = db_session or super().get_db().session

        source = Source(
            workspace_id=workspace_id,
            source_type=obj_in.source_type,
            created_at=datetime.now(),
        )
        db_session.add(source)
        await db_session.flush()

        try:
            try:
                decoded_sas_url = base64.b64decode(obj_in.sas_url).decode("utf-8")
            except Exception as decode_error:
                raise ValueError("Invalid SAS URL encoding.") from decode_error

            secret_name = f"source-{source.id}-sas-url"
            if settings.DEPLOYED_ENV.startswith("azure_"):
                secret_client.set_secret(secret_name, decoded_sas_url)
                logger.info("Decoded SAS URL stored successfully in Azure Key Vault.")
            else:
                vault_client.secrets.kv.v2.create_or_update_secret(
                    path=secret_name, secret={"sas_url": decoded_sas_url}
                )
                logger.info("Decoded SAS URL stored successfully in HashiCorp Vault.")

        except Exception as e:
            logger.error(f"Failed to update the vault: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Failed to store SAS URL in the vault."
            )

        azure_fabric = AzureFabric(
            source_id=source.id,
            container_name=obj_in.container_name,
            sas_url=f"vault:{secret_name}",
        )
        db_session.add(azure_fabric)
        await db_session.flush()
        await db_session.refresh(azure_fabric)
        logger.info("Azure Fabric record created successfully.")

        return source

    ### ADDED FOR GROOVE ###
    async def create_with_groove_source(
        self,
        *,
        workspace_id: UUID,
        obj_in: IGrooveSource,
        db_session: AsyncSession = None,
    ) -> Source:
        """
        Creates a Groove source connector, storing the API key in Vault.
        """
        secret_client: SecretClient = deps.get_secret_client()
        db_session = db_session or super().get_db().session

        # Step 1: Create the main Source entry
        source = Source(
            workspace_id=workspace_id,
            source_type=obj_in.source_type,  # "groove_source"
            created_at=datetime.now(),
        )
        db_session.add(source)
        await db_session.flush()  # Flush to get the generated source.id

        # Step 2: Decode and store the API key in Vault
        secret_name = make_safe_secret_name(source.id, obj_in.source_type)
        try:
            decoded_api_key = base64.b64decode(obj_in.groove_api_key).decode("utf-8")
            credentials = {"api_key": decoded_api_key}

            if settings.DEPLOYED_ENV.startswith("azure_"):
                secret_client.set_secret(secret_name, json.dumps(credentials))
            else:
                vault_client.secrets.kv.v2.create_or_update_secret(
                    path=secret_name,
                    secret=credentials,
                )
            logger.info(f"Groove API key stored in Vault at path: {secret_name}")

        except Exception as e:
            logger.error(
                f"Failed to store Groove API key in vault: {str(e)}", exc_info=True
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to store API key in the vault.",
            )

        # Step 3: Create the GrooveSource database record with a reference to the Vault secret
        groove_record = GrooveModel(
            source_id=source.id,
            source_name=obj_in.source_name,
            api_key_vault_path=f"vault:{secret_name}",
            # Auto-detection configuration
            auto_detection_enabled=obj_in.auto_detection_enabled or False,
            monitoring_frequency_minutes=obj_in.monitoring_frequency_minutes or 30,
            ticket_batch_size=obj_in.ticket_batch_size or 10,
            re_ingest_updated_tickets=obj_in.re_ingest_updated_tickets or False,
        )
        db_session.add(groove_record)
        await db_session.flush()
        await db_session.refresh(groove_record)

        logger.info("Groove source record created successfully.")
        return source

    async def create_with_sql_db(
        self,
        *,
        workspace_id: UUID,
        obj_in: ISQLSource,  # One unified input model
        db_session: AsyncSession = None,
    ) -> Source:
        """
        Creates a SQL source connector (PostgreSQL or MySQL).
        Stores credentials in Vault, metadata in DB.
        """
        secret_client: SecretClient = deps.get_secret_client()
        db_session = db_session or super().get_db().session

        # Step 1: Create the Source entry
        source = Source(
            workspace_id=workspace_id,
            source_type=obj_in.source_type,  # pg_db or mysql_db
            created_at=datetime.now(),
        )
        db_session.add(source)
        await db_session.flush()

        # Step 2: Create and store credentials in Vault
        # secret_name = f"source-{source.id}-{obj_in.source_type}-credentials"
        secret_name = make_safe_secret_name(source.id, obj_in.source_type)
        credentials = {
            "username": obj_in.username,
            "password": obj_in.password,
        }

        try:
            if settings.DEPLOYED_ENV.startswith("azure_"):
                secret_client.set_secret(secret_name, json.dumps(credentials))
                logger.info(
                    f"{obj_in.source_type} credentials stored in Azure Key Vault."
                )
            else:
                vault_client.secrets.kv.v2.create_or_update_secret(
                    path=secret_name,
                    secret=credentials,
                )
                logger.info(
                    f"{obj_in.source_type} credentials stored in HashiCorp Vault."
                )

        except Exception as e:
            logger.error(f"Failed to store credentials in vault: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to store credentials in the vault.",
            )

        # Step 3: Dynamically create the database-specific record
        if obj_in.source_type == "pg_db":
            db_record = PGVector(
                source_id=source.id,
                host=obj_in.host,
                port=obj_in.port,
                database_name=obj_in.database_name,
                username=f"vault:{secret_name}:username",
                password=f"vault:{secret_name}:password",
            )
        elif obj_in.source_type == "mysql_db":
            db_record = MySQLModel(
                source_id=source.id,
                host=obj_in.host,
                port=obj_in.port,
                database_name=obj_in.database_name,
                ssl_mode=obj_in.ssl_mode,
                username=f"vault:{secret_name}:username",
                password=f"vault:{secret_name}:password",
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported SQL source type: {obj_in.source_type}",
            )

        db_session.add(db_record)
        await db_session.flush()
        await db_session.refresh(db_record)

        logger.info(f"{obj_in.source_type.upper()} record created successfully.")
        return source

    async def update_source_with_azure(
        self,
        *,
        workspace_id: UUID,
        source_id: UUID,
        obj_in: ISourceUpdate,
        db_session: AsyncSession = None,
    ) -> None:
        secret_client: SecretClient = deps.get_secret_client()
        source_db = await db_session.get(Source, source_id)
        if not source_db or source_db.workspace_id != workspace_id:
            raise HTTPException(status_code=404, detail="Source not found.")

        azure_storage = await db_session.execute(
            select(AzureModel).where(AzureModel.source_id == source_id)
        )
        azure_storage = azure_storage.scalar_one_or_none()

        if azure_storage:
            try:
                decoded_sas_url = base64.b64decode(obj_in.sas_url).decode("utf-8")

                if settings.DEPLOYED_ENV.startswith("azure_"):
                    vault_reference = azure_storage.sas_url.replace("vault:", "")
                    secret_client.set_secret(vault_reference, decoded_sas_url)
                else:
                    vault_reference = azure_storage.sas_url.replace("vault:", "")
                    vault_client.secrets.kv.v2.create_or_update_secret(
                        path=vault_reference, secret={"sas_url": decoded_sas_url}
                    )
                print(f"Successfully updated SAS URL in {settings.DEPLOYED_ENV} Vault.")
            except Exception as e:
                logger.error(f"Failed to update the Vault: {str(e)}")
                raise HTTPException(
                    status_code=500, detail="Failed to update the Vault."
                )

            azure_storage.container_name = obj_in.container_name
            await db_session.commit()
            await db_session.refresh(azure_storage)

        source_db.updated_at = datetime.now()
        await db_session.commit()
        await db_session.refresh(source_db)

    async def update_source_with_azure_fabric(
        self,
        *,
        workspace_id: UUID,
        source_id: UUID,
        obj_in: ISourceUpdate,
        db_session: AsyncSession = None,
    ) -> None:
        secret_client: SecretClient = deps.get_secret_client()
        source_db = await db_session.get(Source, source_id)
        if not source_db or source_db.workspace_id != workspace_id:
            raise HTTPException(status_code=404, detail="Source not found.")

        azure_fabric = await db_session.execute(
            select(AzureFabric).where(AzureFabric.source_id == source_id)
        )
        azure_fabric = azure_fabric.scalar_one_or_none()

        if azure_fabric:
            try:
                decoded_sas_url = base64.b64decode(obj_in.sas_url).decode("utf-8")

                if settings.DEPLOYED_ENV.startswith("azure_"):
                    vault_reference = azure_fabric.sas_url.replace("vault:", "")
                    secret_client.set_secret(vault_reference, decoded_sas_url)
                else:
                    vault_reference = azure_fabric.sas_url.replace("vault:", "")
                    vault_client.secrets.kv.v2.create_or_update_secret(
                        path=vault_reference, secret={"sas_url": decoded_sas_url}
                    )
                print(f"Successfully updated SAS URL in {settings.DEPLOYED_ENV} Vault.")
            except Exception as e:
                logger.error(f"Failed to update the Vault: {str(e)}")
                raise HTTPException(
                    status_code=500, detail="Failed to update the Vault."
                )

            azure_fabric.container_name = obj_in.container_name
            await db_session.commit()
            await db_session.refresh(azure_fabric)

        source_db.updated_at = datetime.now()
        await db_session.commit()
        await db_session.refresh(source_db)

    async def update_source_with_s3(
        self,
        *,
        workspace_id: UUID,
        source_id: UUID,
        obj_in: ISourceUpdate,
        db_session: AsyncSession = None,
    ) -> None:
        secret_client: SecretClient = deps.get_secret_client()
        source_db = await db_session.get(Source, source_id)
        if not source_db or source_db.workspace_id != workspace_id:
            raise HTTPException(status_code=404, detail="Source not found.")

        s3_storage = await db_session.execute(
            select(S3Model).where(S3Model.source_id == source_id)
        )
        s3_storage = s3_storage.scalar_one_or_none()

        if s3_storage:
            try:
                decoded_access_id = base64.b64decode(obj_in.access_id).decode("utf-8")
                decoded_access_secret = base64.b64decode(obj_in.access_secret).decode(
                    "utf-8"
                )

                secret_name = f"source-{source_id}-credentials"
                if settings.DEPLOYED_ENV.startswith("azure_"):
                    secret_client.set_secret(f"{secret_name}-access", decoded_access_id)
                    secret_client.set_secret(
                        f"{secret_name}-secret", decoded_access_secret
                    )
                else:
                    vault_client.secrets.kv.v2.create_or_update_secret(
                        path=f"{secret_name}-access",
                        secret={"access_id": decoded_access_id},
                    )
                    vault_client.secrets.kv.v2.create_or_update_secret(
                        path=f"{secret_name}-secret",
                        secret={"access_secret": decoded_access_secret},
                    )
                print(
                    f"Successfully updated access credentials in {settings.DEPLOYED_ENV} Vault."
                )
            except Exception as e:
                logger.error(f"Failed to update the Vault: {str(e)}")
                raise HTTPException(
                    status_code=500, detail="Failed to update the Vault."
                )

            s3_storage.bucket_name = obj_in.bucket_name
            await db_session.commit()
            await db_session.refresh(s3_storage)

        source_db.updated_at = datetime.now()
        await db_session.commit()
        await db_session.refresh(source_db)

    ### ADDED FOR GROOVE ###
    async def update_source_with_groove(
        self,
        *,
        workspace_id: UUID,
        source_id: UUID,
        obj_in: IGrooveSource,
        db_session: AsyncSession = None,
    ) -> None:
        """
        Updates an existing Groove source connector, primarily for updating the API key in Vault.
        """
        db_session = db_session or super().get_db().session
        secret_client: SecretClient = deps.get_secret_client()

        # Step 1: Validate source exists and belongs to the workspace
        source_db = await self._get_validated_source(
            source_id, workspace_id, db_session
        )

        # Step 2: Get the GrooveSource model
        groove_db = (
            await db_session.execute(
                select(GrooveModel).where(GrooveModel.source_id == source_id)
            )
        ).scalar_one_or_none()

        if not groove_db:
            raise HTTPException(
                status_code=404, detail="Groove source details not found."
            )

        # Step 3: Update the API key in Vault if a new one is provided
        if obj_in.groove_api_key:
            try:
                decoded_api_key = base64.b64decode(obj_in.groove_api_key).decode(
                    "utf-8"
                )
                credentials = {"api_key": decoded_api_key}

                secret_path = groove_db.api_key_vault_path.replace("vault:", "")

                if settings.DEPLOYED_ENV.startswith("azure_"):
                    secret_client.set_secret(secret_path, json.dumps(credentials))
                else:
                    vault_client.secrets.kv.v2.create_or_update_secret(
                        path=secret_path,
                        secret=credentials,
                    )
                logger.info(f"Groove API key updated in Vault at path: {secret_path}")

            except Exception as e:
                logger.error(
                    f"Failed to update Groove API key in vault: {str(e)}", exc_info=True
                )
                raise HTTPException(
                    status_code=500, detail="Failed to update API key in the vault."
                )

        # Step 4: Update the timestamp and commit
        source_db.updated_at = datetime.now()
        await db_session.commit()
        await db_session.refresh(source_db)

    async def update_sql_db(
        self,
        *,
        workspace_id: UUID,
        source_id: UUID,
        obj_in: ISourceUpdate,
        source_type: str,  # "pg_db" or "mysql_db"
        db_session: AsyncSession = None,
    ) -> None:
        db_session = db_session or super().get_db().session

        logger.info(
            f"[{source_type}] Update requested for source_id={source_id} in workspace_id={workspace_id}"
        )

        # Validate source belongs to workspace
        logger.info(
            "befor the get validated source not in the function testing purpose"
        )
        source = await self._get_validated_source(source_id, workspace_id, db_session)
        logger.info("outside the get_Validation_source")
        logger.debug(f"[{source_type}] Source validated: {source}")

        # Load the corresponding PGVector or MySQLModel entry
        logger.info("inside the load db model")
        db_model = await self._load_db_model(source_type, source_id, db_session)
        logger.info("out of the load db model")

        # Prepare and store updated credentials in Vault
        secret_client: SecretClient = deps.get_secret_client()
        secret_name = make_safe_secret_name(source_id, source_type)

        credentials = {}
        if hasattr(obj_in, "username") and obj_in.username:
            credentials["username"] = obj_in.username
        if hasattr(obj_in, "password") and obj_in.password:
            credentials["password"] = obj_in.password

        try:
            if credentials:  # Only update Vault if there is something to store
                if settings.DEPLOYED_ENV.startswith("azure_"):
                    secret_client.set_secret(secret_name, json.dumps(credentials))
                    logger.info("Updated credentials stored in Azure Key Vault.")
                else:
                    vault_client.secrets.kv.v2.create_or_update_secret(
                        path=secret_name,
                        secret=credentials,
                    )
                    logger.info("Updated credentials stored in HashiCorp Vault.")
        except Exception as e:
            logger.error(f"Failed to update credentials in Vault: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to update credentials in the vault.",
            )

        # Update vault reference in DB model if new username/password are provided
        if "username" in credentials:
            try:
                db_model.username = f"vault:{secret_name}:username"
            except Exception as e:
                logger.error(f"Failed to assign username: {e}")
                raise

        if "password" in credentials:
            try:
                db_model.password = f"vault:{secret_name}:password"
            except Exception as e:
                logger.error(f"Failed to assign password: {e}")
                raise

        # Update other fields from the request body
        logger.info("before the update_model_field")
        self._update_model_fields(source_type, db_model, obj_in)

        # Timestamp update
        source.updated_at = datetime.now()

        # Commit and refresh
        await self._commit_update(db_session, db_model, source, source_type, source_id)

    async def _get_validated_source(
        self, source_id: UUID, workspace_id: UUID, db_session: AsyncSession
    ):
        logger.info("inside the get_Validation_source")
        source = await db_session.get(Source, source_id)
        if not source or source.workspace_id != workspace_id:
            raise HTTPException(status_code=404, detail="Source not found.")
        return source

    async def _get_pg_db(self, source_id: UUID, db_session: AsyncSession):
        pg_db = (
            await db_session.execute(
                select(PGVector).where(PGVector.source_id == source_id)
            )
        ).scalar_one_or_none()

        if not pg_db:
            raise HTTPException(status_code=404, detail="PGVector source not found.")
        return pg_db

    async def _prepare_updated_credentials(
        self, obj_in: ISourceUpdate, source_id: UUID, source_type: str
    ) -> dict:
        # secret_path = f"source-{source_id}-pg-vector-credentials"
        secret_path = self._get_secret_path(source_id, source_type)

        try:
            if settings.DEPLOYED_ENV.startswith("azure_"):
                secret_client: SecretClient = deps.get_secret_client()
                secret_bundle = secret_client.get_secret(secret_path)
                existing_credentials = json.loads(secret_bundle.value)
            else:
                result = vault_client.secrets.kv.v2.read_secret_version(
                    path=secret_path
                )
                existing_credentials = result["data"]["data"]
        except Exception as e:
            logger.error(f"Failed to read existing Vault secrets: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to read Vault secrets.")

        return {
            "username": obj_in.username or existing_credentials.get("username"),
            "password": obj_in.password or existing_credentials.get("password"),
        }

    async def _store_credentials(
        self, source_id: UUID, source_type: str, updated_credentials: dict
    ):
        # secret_path = f"source-{source_id}-pg-vector-credentials"
        secret_path = self._get_secret_path(source_id, source_type)
        try:
            if settings.DEPLOYED_ENV.startswith("azure_"):
                secret_client: SecretClient = deps.get_secret_client()
                secret_client.set_secret(secret_path, json.dumps(updated_credentials))
            else:
                vault_client.secrets.kv.v2.create_or_update_secret(
                    path=secret_path,
                    secret=updated_credentials,
                )
            logger.info(f"Successfully stored credentials at {secret_path}")
        except Exception as e:
            logger.error(f"Failed to update Vault secrets: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Failed to update Vault secrets."
            )

    # NEW METHOD: Generate correct secret path based on database type
    def _get_secret_path(self, source_id: UUID, source_type: str) -> str:
        """Generate the correct secret path based on source type"""
        type_mapping = {
            "mysql_db": "mysql_db-credentials",
            "pg_db": "pg_db-credentials",
            # "pg_vector": "pg-vector-credentials",
            # "mssql_db": "mssql-credentials",
            # Add other database types as needed
        }

        credential_suffix = type_mapping.get(source_type, "credentials")
        return f"source-{source_id}-{credential_suffix}"

    def _update_pg_db_fields(self, pg_db: PGVector, obj_in: ISourceUpdate):
        pg_db.host = obj_in.host or pg_db.host
        pg_db.port = obj_in.port or pg_db.port
        pg_db.database_name = obj_in.database_name or pg_db.database_name

    async def check_valid_file_type(
        self, *, file_extension: str, content_type: str
    ) -> bool:
        valid_extensions = settings.ALLOWED_FILE_EXTENSIONS
        valid_mimetypes = settings.ALLOWED_MIME_TYPES

        # Check if file extension is allowed
        if file_extension not in valid_extensions:
            return False
        if content_type not in valid_mimetypes:
            return False

        # Additional check for video files - require video ingestion to be enabled
        video_extensions = {"mp4", "avi", "mov", "wmv", "flv", "webm", "mkv"}
        video_mimetypes = {
            "video/mp4",
            "video/x-msvideo",
            "video/quicktime",
            "video/x-ms-wmv",
            "video/x-flv",
            "video/webm",
            "video/x-matroska",
        }

        is_video_file = (
            file_extension in video_extensions or content_type in video_mimetypes
        )

        if is_video_file and not is_video_ingestion_enabled():
            logger.warning(
                f"Video file upload blocked - video ingestion is disabled: {file_extension}"
            )
            return False

        return True

    def sync_check_valid_file_type(self, *, file_extension: str) -> bool:
        valid_extensions = settings.ALLOWED_FILE_EXTENSIONS
        # valid_mimetypes = settings.ALLOWED_MIME_TYPES
        if file_extension not in valid_extensions:
            return False
        # if content_type not in valid_mimetypes:
        #     return False

        # Additional check for video files - require video ingestion to be enabled
        video_extensions = {"mp4", "avi", "mov", "wmv", "flv", "webm", "mkv"}

        if file_extension in video_extensions and not is_video_ingestion_enabled():
            logger.warning(
                f"Video file upload blocked - video ingestion is disabled: {file_extension}"
            )
            return False

        return True

    async def compress_pdf_if_needed(
        self,
        original_path: str,
        compressed_path: str,
        pixel_threshold: int = 2000,
        pages_to_check: int = 3,
    ) -> bool:
        """
        Checks if PDF needs compression and compresses it if necessary.
        Returns True if compression was performed, False otherwise.
        """
        logger.info(f"Starting PDF compression check for: {original_path}")

        try:
            doc = fitz.open(original_path)
            logger.debug(f"Successfully opened PDF with {len(doc)} pages")

            # Check for high resolution pages
            for i in range(min(pages_to_check, len(doc))):
                page = doc.load_page(i)
                pix = page.get_pixmap(dpi=100)
                if pix.width > pixel_threshold or pix.height > pixel_threshold:
                    logger.info(
                        f"High resolution page found: {pix.width}x{pix.height} px"
                    )
                    break
            else:
                logger.info("No high-resolution pages found, skipping compression")
                return False

            logger.info("Starting PDF compression process")
            images = []
            for page_num in range(len(doc)):
                logger.debug(f"Processing page {page_num + 1}/{len(doc)}")
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=100)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Optional resizing for very large images
                max_width = 1200
                if img.width > max_width:
                    w_percent = max_width / float(img.width)
                    h_size = int((float(img.height) * float(w_percent)))
                    logger.debug(
                        f"Resizing image from {img.width}x{img.height} to {max_width}x{h_size}"
                    )
                    img = img.resize((max_width, h_size), Image.LANCZOS)

                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format="JPEG", quality=50)
                img_byte_arr.seek(0)
                images.append(ImageReader(img_byte_arr))

            # Create compressed PDF
            logger.info(f"Creating compressed PDF: {compressed_path}")
            c = canvas.Canvas(compressed_path, pagesize=A4)
            for i, img in enumerate(images, 1):
                logger.debug(f"Adding compressed page {i}/{len(images)}")
                c.drawImage(img, 0, 0, width=A4[0], height=A4[1])
                c.showPage()
            c.save()

            # Compare file sizes
            original_size = os.path.getsize(original_path)
            compressed_size = os.path.getsize(compressed_path)
            compression_ratio = (1 - compressed_size / original_size) * 100

            logger.info(
                f"Compression complete. Original: {original_size/1024:.2f}KB, "
                f"Compressed: {compressed_size/1024:.2f}KB, "
                f"Reduction: {compression_ratio:.1f}%"
            )

            return True

        except Exception as e:
            logger.error(f"PDF compression failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"PDF compression failed: {str(e)}"
            )

    async def upload_file(
        self,
        *,
        workspace_id: UUID,
        file: UploadFile,
        db_session: AsyncSession = None,
        target_path: str,
        user_id: UUID = None,
    ) -> File:
        """Uploads a file and queues PDF processing as a background task if needed."""
        db_session = db_session or super().get_db().session
        logger.info(f"Starting file upload process for workspace: {workspace_id}")

        try:
            file_extension = file.filename.split(".")[-1].lower()
            if not await self._validate_file_type(file_extension, file.content_type):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type. Allowed: {settings.ALLOWED_FILE_EXTENSIONS}",
                )

            file_id = uuid7()
            if not os.path.exists(target_path):
                os.makedirs(target_path)
                logger.debug(f"Created target directory: {target_path}")

            final_filename = await self._generate_unique_filename(
                workspace_id, file.filename, db_session
            )
            file_location = os.path.join(target_path, final_filename)
            await self._write_file(file, file_location)

            file_size = os.path.getsize(file_location)
            initial_status = (
                FileStatusEnum.Processing
                if file_extension == "pdf"
                else FileStatusEnum.Uploaded
            )

            original_file = await self._create_file_record(
                db_session,
                file_id,
                workspace_id,
                final_filename,
                file.content_type,
                file_size,
                file_location,
                initial_status,
            )

            if file_extension == "pdf":
                await self._queue_pdf_task(
                    file_id, workspace_id, user_id, db_session, original_file
                )

            return original_file

        except Exception as e:
            logger.error(f"File upload failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

    async def _validate_file_type(self, file_extension: str, content_type: str) -> bool:
        logger.debug(f"Validating file type: {file_extension}, {content_type}")
        valid = await self.check_valid_file_type(
            file_extension=file_extension, content_type=content_type
        )
        if not valid:
            logger.warning(f"Invalid file type attempted: {file_extension}")
        return valid

    async def _generate_unique_filename(
        self, workspace_id, original_filename, db_session
    ):
        base_name, ext = os.path.splitext(original_filename)
        query = await db_session.execute(
            select(File).where(
                File.workspace_id == workspace_id,
                File.deleted_at.is_(None),
                File.source_id.is_(None),
            )
        )
        existing = query.scalars().all()
        existing_names = [os.path.splitext(f.filename)[0] for f in existing]

        unique_name = base_name
        counter = 1
        while unique_name in existing_names:
            unique_name = f"{base_name}({counter})"
            counter += 1
        return f"{unique_name}{ext}"

    async def _write_file(self, file: UploadFile, file_location: str):
        logger.info(f"Saving file as: {os.path.basename(file_location)}")
        async with aiofiles.open(file_location, "wb") as f:
            while True:
                content = await file.read(1024 * 1024)
                if not content:
                    break
                await f.write(content)
        logger.info(
            f"Original file saved: {file_location} ({os.path.getsize(file_location)/1024:.2f}KB)"
        )

    async def _create_file_record(
        self, db_session, file_id, workspace_id, filename, mimetype, size, path, status
    ) -> File:
        file_record = File(
            id=file_id,
            workspace_id=workspace_id,
            filename=filename,
            mimetype=mimetype,
            size=size,
            file_path=path,
            status=status,
            created_at=datetime.utcnow(),
        )
        db_session.add(file_record)
        await db_session.commit()
        await db_session.refresh(file_record)
        logger.debug(f"File record created with {status.value} status, ID: {file_id}")
        return file_record

    async def _queue_pdf_task(
        self, file_id, workspace_id, user_id, db_session, file_obj: File
    ):
        try:
            logger.info("PDF file detected, queuing processing task")
            task_kwargs = {
                "file_id": file_id,
                "workspace_id": workspace_id,
                "pixel_threshold": 2000,
                "pages_to_check": 3,
            }
            if user_id:
                task_kwargs["user_id"] = user_id

            celery.signature("tasks.check_pdf_task", kwargs=task_kwargs).apply_async()
            logger.info(f"PDF processing task queued for file: {file_id}")
        except Exception as e:
            logger.error(
                f"Failed to queue PDF processing task: {str(e)}", exc_info=True
            )
            file_obj.status = FileStatusEnum.Uploaded
            await db_session.commit()
            logger.warning(
                f"Set file status to Uploaded due to task queue failure: {file_id}"
            )

    async def delete_source(
        self,
        *,
        source_id: UUID,
        workspace_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> Source:
        """Soft delete a source and its related entries."""

        db_session = db_session or super().get_db().session
        logger.info(
            f"Attempting to soft delete source {source_id} for workspace {workspace_id}"
        )

        source = await self.get_by_id(
            source_id=source_id, workspace_id=workspace_id, db_session=db_session
        )
        if not source:
            logger.warning(f"Source with ID {source_id} not found.")
            raise HTTPException(status_code=404, detail="Source not found")

        try:
            source.deleted_at = datetime.now()
            await db_session.commit()

            source_models = {
                "azure_storage": AzureModel,
                "aws_s3": S3Model,
                "azure_fabric": AzureFabric,
                "pg_db": PGVector,
                "mysql_db": MySQLModel,
                "groove_source": GrooveModel,  # ### MODIFIED FOR GROOVE ###
            }

            if source.source_type in source_models:
                entry = (
                    (
                        await db_session.execute(
                            select(source_models[source.source_type]).where(
                                source_models[source.source_type].source_id == source_id
                            )
                        )
                    )
                    .scalars()
                    .first()
                )

                if entry:
                    entry.deleted_at = datetime.now()
                    await db_session.commit()
                    logger.info(
                        f"{source.source_type} entry for source {source_id} marked as deleted."
                    )

            # Get all files linked to this source before marking them as deleted
            # so we can clean up video segments for video files
            files_result = await db_session.execute(
                select(File).where(
                    File.source_id == source_id, File.deleted_at.is_(None)
                )
            )
            files_to_delete = files_result.scalars().all()

            # Clean up video segments for any video files in this source
            from app.utils.video_cleanup_utils import (
                clear_reference_cache,
                delete_video_segments,
                is_video_file,
            )

            video_files_processed = 0
            video_files_cleaned = 0
            video_files_preserved = 0

            for file in files_to_delete:
                if is_video_file(file.filename, file.mimetype):
                    video_files_processed += 1

                    # Clear cache for this specific file to ensure fresh reference check
                    clear_reference_cache(workspace_id=workspace_id, file_id=file.id)
                    logger.debug(f"Cleared reference cache for file {file.id}")

                    try:
                        segments_deleted = await delete_video_segments(
                            workspace_id, file.id, db_session=db_session
                        )
                        if segments_deleted:
                            video_files_cleaned += 1
                            logger.info(
                                f"Video segments deleted for file {file.id} ({file.filename})"
                            )
                        else:
                            video_files_preserved += 1
                            logger.info(
                                f"Video segments preserved for file {file.id} ({file.filename}) - still referenced by active documents"
                            )
                    except Exception as cleanup_error:
                        logger.error(
                            f"Error cleaning up video segments for file {file.id}: {cleanup_error}",
                            exc_info=True,
                        )

            if video_files_processed > 0:
                logger.info(
                    f"Processed {video_files_processed} video files from source {source_id}: "
                    f"{video_files_cleaned} cleaned up, {video_files_preserved} preserved"
                )

                # Try to clean up empty workspace directory and orphaned segments
                try:
                    from app.utils.video_cleanup_utils import (
                        cleanup_empty_workspace_video_dir,
                        cleanup_orphaned_video_segments,
                    )

                    # Clean up any orphaned segments that might exist
                    orphaned_count = await cleanup_orphaned_video_segments(
                        workspace_id, db_session=db_session
                    )
                    if orphaned_count > 0:
                        logger.info(
                            f"Cleaned up {orphaned_count} additional orphaned video segment directories"
                        )

                    # Clean up empty workspace directory
                    cleanup_empty_workspace_video_dir(workspace_id)
                except Exception as e:
                    logger.warning(
                        f"Failed to cleanup workspace video directories: {e}"
                    )

            # Mark all files as deleted
            await db_session.execute(
                update(File)
                .where(File.source_id == source_id)
                .values(deleted_at=datetime.now())
            )
            await db_session.commit()
            logger.info(f"All files linked to source {source_id} marked as deleted.")
            logger.info(f"Source {source_id} marked as deleted.")
            return source

        except Exception as e:
            logger.error(f"Failed to soft delete source {source_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal Server Error")

    async def delete_file(
        self,
        *,
        file_id: UUID,
        workspace_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> File:
        """Soft delete a file by setting its deleted_at timestamp."""

        db_session = db_session or super().get_db().session
        try:
            result = await db_session.execute(
                select(File).where(
                    File.id == file_id,
                    File.workspace_id == workspace_id,
                    File.deleted_at.is_(None),
                )
            )
            file = result.scalar_one_or_none()

            if not file:
                logger.warning(f"File with ID {file_id} not found or already deleted.")
                raise HTTPException(status_code=404, detail="File not found")

            # Check if this is a video file and clean up segments if needed
            from app.utils.video_cleanup_utils import (
                cleanup_empty_workspace_video_dir,
                clear_reference_cache,
                delete_video_segments,
                is_video_file,
            )

            if is_video_file(file.filename, file.mimetype):
                logger.info(
                    f"Video file detected for deletion: {file.filename} (file_id: {file_id})"
                )

                # Clear cache for this specific file to ensure fresh reference check
                clear_reference_cache(workspace_id=workspace_id, file_id=file_id)
                logger.debug(f"Cleared reference cache for file {file_id}")

                # Try to delete video segments (will only delete if no active references)
                try:
                    segments_deleted = await delete_video_segments(
                        workspace_id, file_id, db_session=db_session
                    )
                    if segments_deleted:
                        logger.info(
                            f"Video segments successfully deleted for file {file_id}"
                        )
                        # Try to clean up empty workspace directory
                        cleanup_empty_workspace_video_dir(workspace_id)
                    else:
                        logger.info(
                            f"Video segments for file {file_id} are still referenced by active documents - keeping segments"
                        )
                except Exception as cleanup_error:
                    logger.error(
                        f"Error during video segments cleanup for file {file_id}: {cleanup_error}",
                        exc_info=True,
                    )

            file.deleted_at = datetime.utcnow()
            await db_session.commit()
            logger.info(f"File {file_id} marked as deleted.")

            return file

        except Exception as e:
            logger.error(
                f"Failed to soft delete file {file_id}: {str(e)}", exc_info=True
            )
            raise HTTPException(status_code=500, detail="Internal Server Error")

    async def get_sources_paginated_ordered(
        self,
        *,
        workspace_id: UUID,
        params: Params = Params(),
        order_by: str = "created_at",
        order: IOrderEnum = IOrderEnum.ascendent,
        db_session: AsyncSession | None = None,
    ) -> Page[Source]:
        db_session = db_session or super().get_db().session
        columns = Source.__table__.columns

        if order_by not in columns:
            order_by = "created_at"

        order_clause = (
            asc(columns[order_by])
            if order == IOrderEnum.ascendent
            else desc(columns[order_by])
        )

        query = (
            select(Source)
            .where(Source.workspace_id == workspace_id, Source.deleted_at.is_(None))
            .options(
                selectinload(Source.azure_storage),
                selectinload(Source.aws_s3_storage),
                selectinload(Source.azure_fabric),
                selectinload(Source.source_pgvector),
                selectinload(Source.source_mysql),
                selectinload(Source.groove_source),  ### MODIFIED FOR GROOVE ###
            )
            .order_by(order_clause)
        )

        results = await db_session.execute(query)
        sources = results.scalars().all()

        for source in sources:
            if source.source_type == "pg_db":
                if not source.source_pgvector:
                    logger.error(f"PGVector entry MISSING for source {source.id}")

        return paginate(sources, params)

    async def get_files_by_workspace(
        self,
        *,
        workspace_id: UUID,
        params: OptionalParams = OptionalParams(),
        order_by: str = "created_at",
        order: IOrderEnum = IOrderEnum.ascendent,
        only_uploaded: bool = True,
        search: Optional[str] = None,
        db_session: AsyncSession | None = None,
    ) -> Page[File] | list[File]:
        db_session = db_session or self.get_db().session
        columns = File.__table__.columns

        if order_by not in columns:
            order_by = "created_at"

        order_clause = (
            asc(columns[order_by])
            if order == IOrderEnum.ascendent
            else desc(columns[order_by])
        )

        filters = [File.workspace_id == workspace_id, File.deleted_at.is_(None)]

        if only_uploaded:
            filters.append(File.source_id.is_(None))

        # Add search functionality
        if search:
            search_pattern = f"%{search.lower()}%"
            filters.append(func.lower(File.filename).like(search_pattern))

        query = (
            select(File)
            .where(*filters)
            .options(selectinload(File.workspace))
            .order_by(order_clause)
        )

        results = await db_session.execute(query)
        files = results.scalars().all()

        if params.page is None and params.size is None:
            return files

        return paginate(files, Params(page=params.page, size=params.size))

    async def get_file_by_id_in_workspace(
        self,
        *,
        file_id: UUID,
        workspace_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> File | None:
        db_session = db_session or super().get_db().session

        query = (
            select(File)
            .where(
                File.id == file_id,
                File.workspace_id == workspace_id,
                File.deleted_at.is_(None),
            )
            .options(selectinload(File.workspace))
        )

        result = await db_session.execute(query)
        return result.scalar_one_or_none()

    async def get_file_names_by_workspace(
        self,
        *,
        workspace_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> List[File]:
        db_session = db_session or super().get_db().session
        query = (
            select(File.filename)
            .where(File.workspace_id == workspace_id)
            .options(selectinload(File.workspace))
        )
        results = await db_session.execute(query)
        return results.scalars().all()

    async def get_file_names_by_file_ids(
        self,
        *,
        file_ids: List[UUID],
        db_session: AsyncSession | None = None,
    ) -> List[File]:
        db_session = db_session or super().get_db().session
        query = select(File.filename).where(File.id.in_(file_ids))
        results = await db_session.execute(query)
        return results.scalars().all()

    async def get_file_by_fileid(
        self,
        *,
        file_id: UUID,
        db_session: AsyncSession | None = None,
    ):
        db_session = db_session or super().get_db().session
        query = select(File).where(File.id == file_id)
        result = await db_session.execute(query)
        return result.scalars().first()

    async def fetch_secret_value(self, vault_reference: str, secret_type: str) -> str:
        """Fetches secret value from Azure Key Vault or HashiCorp Vault."""
        secret_client: SecretClient = deps.get_secret_client()
        try:
            vault_path = unquote(
                vault_reference.replace("vault:", "").split(":")[0].strip()
            )

            if settings.DEPLOYED_ENV.startswith("azure_"):
                raw_secret = secret_client.get_secret(vault_path).value
            else:
                secret = vault_client.secrets.kv.v2.read_secret_version(path=vault_path)
                raw_secret = secret["data"]["data"].get(secret_type)

            if not raw_secret:
                logger.warning(
                    f"Secret '{secret_type}' not found or empty in Vault: {vault_path}"
                )
                return None

            try:
                secret_data = json.loads(raw_secret)
                secret_value = secret_data.get(secret_type)
                if not secret_value:
                    return None
            except json.JSONDecodeError:
                secret_value = raw_secret

            return secret_value

        except Exception as e:
            logger.error(f"Failed to fetch secret {vault_reference}: {str(e)}")
            return None

    async def process_source(self, source) -> Optional[ISourceConnectorResponse]:
        """Processes a single source and returns a response model."""
        logger.debug(f"Processing source: {source.id} of type: {source.source_type}")

        ### MODIFIED FOR GROOVE ###
        handler_map = {
            "azure_storage": self._handle_azure_storage,
            "azure_fabric": self._handle_azure_fabric,
            "aws_s3": self._handle_aws_s3,
            "pg_db": lambda source: self._handle_sql_db(
                source, "source_pgvector", PGVectorSourceResponse
            ),
            "mysql_db": lambda source: self._handle_sql_db(
                source, "source_mysql", MySQLSourceResponse
            ),
            "groove_source": self._handle_groove_source,
        }

        handler = handler_map.get(source.source_type)
        if not handler:
            logger.error(
                f"Unknown source_type: {source.source_type} for source {source.id}"
            )
            return None

        try:
            return await handler(source)
        except Exception as e:
            logger.error(
                f"Failed to process {source.source_type} source {source.id}: {str(e)}"
            )
            return None

    ### ADDED FOR GROOVE ###
    async def _handle_groove_source(self, source) -> ISourceConnectorResponse:
        try:
            logger.debug(
                f"Processing Groove source: {source.id}, source_type: {source.source_type}"
            )
            logger.debug(f"Source attributes: {dir(source)}")

            storage = getattr(source, "groove_source", None)
            logger.debug(f"Groove storage object: {storage}")

            if not storage:
                logger.error(
                    f"Groove source relationship not found for source {source.id}"
                )
                logger.error(
                    f"Available relationships: {[attr for attr in dir(source) if not attr.startswith('_')]}"
                )
                self._log_missing_storage(source)
                return None

            logger.debug(f"Processing Groove source: {source.id}, storage: {storage}")
            logger.debug(f"Storage attributes: {dir(storage)}")

            return ISourceConnectorResponse(
                sources=GrooveSourceResponse(
                    source_id=source.id,
                    source_type=source.source_type,
                    source_name=storage.source_name,
                    auto_detection_enabled=storage.auto_detection_enabled or False,
                    monitoring_frequency_minutes=storage.monitoring_frequency_minutes
                    or 30,
                    ticket_batch_size=storage.ticket_batch_size or 10,
                    re_ingest_updated_tickets=storage.re_ingest_updated_tickets
                    or False,
                )
            )
        except Exception as e:
            logger.error(
                f"Error processing Groove source {source.id}: {str(e)}", exc_info=True
            )
            return None

    async def _handle_azure_storage(self, source) -> ISourceConnectorResponse:
        storage = getattr(source, "azure_storage", None)
        if not storage:
            self._log_missing_storage(source)
            return None

        return ISourceConnectorResponse(
            sources=AzureStorageSourceResponse(
                source_id=source.id,
                source_type=source.source_type,
                container_name=storage.container_name,
            )
        )

    async def _handle_azure_fabric(self, source) -> ISourceConnectorResponse:
        storage = getattr(source, "azure_fabric", None)
        if not storage:
            self._log_missing_storage(source)
            return None

        return ISourceConnectorResponse(
            sources=AzureFabricSourceResponse(
                source_id=source.id,
                source_type=source.source_type,
                container_name=storage.container_name,
            )
        )

    async def _handle_aws_s3(self, source) -> Optional[ISourceConnectorResponse]:
        storage = source.aws_s3_storage
        if not storage:
            self._log_missing_storage(source)
            return None

        access_key = await self.fetch_secret_value(storage.access_id, "access_key")
        secret_key = await self.fetch_secret_value(storage.access_secret, "secret_key")

        if not access_key or not secret_key:
            logger.warning(f"Missing credentials for AWS S3 source {source.id}")
            return None

        return ISourceConnectorResponse(
            sources=AWSS3SourceResponse(
                source_id=source.id,
                source_type=source.source_type,
                bucket_name=storage.bucket_name,
            )
        )

    async def _handle_sql_db(
        self,
        source,
        storage_attr: str,
        response_model: Type[ISourceConnectorResponse],
    ) -> Optional[ISourceConnectorResponse]:
        """Generic handler for SQL DB sources (e.g., pg_db, mysql_db)"""

        storage = getattr(source, storage_attr, None)
        if not storage:
            self._log_missing_storage(source)
            return None

        username = await self.fetch_secret_value(storage.username, "username")
        password = await self.fetch_secret_value(storage.password, "password")

        if not username or not password:
            logger.warning(
                f"Missing credentials for {source.source_type} source {source.id}"
            )
            return None

        return ISourceConnectorResponse(
            sources=response_model(
                source_id=source.id,
                source_type=source.source_type,
                host=storage.host,
                port=storage.port,
                database_name=storage.database_name,
            )
        )

    def _log_missing_storage(self, source):
        logger.error(
            f"Storage details not found for source: {source.id} of type: {source.source_type}"
        )

    async def get_azure_file_status(
        self,
        *,
        source_id: UUID,
        filename: str,
        db_session: AsyncSession | None = None,
    ) -> FileStatusEnum | None:
        db_session = db_session or super().get_db().session
        query = select(File.status).where(
            func.concat(source_id, "_", filename) == File.filename
        )
        result = await db_session.execute(query)
        file_status = result.scalars().first()
        if file_status is None:
            return None
        return file_status

    async def get_file_metadata_by_source_id(
        self,
        *,
        source_id: UUID,
        params: Params = Params(),
        search: Optional[str] = None,
        order_by: str = "created_at",
        order: IOrderEnum = IOrderEnum.ascendent,
        db_session: AsyncSession | None = None,
    ) -> Page[File]:
        db_session = db_session or super().get_db().session

        filters = [File.source_id == source_id, File.deleted_at.is_(None)]

        # Add search functionality
        if search:
            search_pattern = f"%{search.lower()}%"
            filters.append(func.lower(File.filename).like(search_pattern))

        # Add ordering support
        columns = File.__table__.columns

        if order_by not in columns:
            order_by = "created_at"

        order_clause = (
            asc(columns[order_by])
            if order == IOrderEnum.ascendent
            else desc(columns[order_by])
        )

        query = select(File).where(*filters).order_by(order_clause)

        results = await db_session.execute(query)
        file_metadata = results.scalars().all()

        return paginate(file_metadata, params)

    async def get_source_info(
        self, workspace_id: UUID, source_id: UUID, db_session: AsyncSession
    ):
        result = await db_session.execute(
            select(Source)
            .options(
                joinedload(Source.azure_storage),
                joinedload(Source.aws_s3_storage),
                joinedload(Source.azure_fabric),
                joinedload(Source.source_pgvector),
                joinedload(Source.source_mysql),
                joinedload(Source.groove_source),  ### MODIFIED FOR GROOVE ###
            )
            .where(
                Source.id == source_id,
                Source.workspace_id == workspace_id,
                Source.deleted_at.is_(None),
            )
        )
        source = result.scalar_one_or_none()

        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        logger.debug(f"Loaded source: {source.id} with type: {source.source_type}")

        if source.source_type == "aws_s3":
            logger.debug(f"aws_s3_storage: {source.aws_s3_storage}")

        return source

    async def check_azure_connection(
        self, *, container_name: str, sas_url: str
    ) -> tuple[bool, str]:
        try:
            blob_service_client = BlobServiceClient(account_url=sas_url)
            container_client = blob_service_client.get_container_client(container_name)
            await container_client.get_container_properties()
            return True, "Azure connection successful"
        except ResourceNotFoundError:
            return False, "Azure container or blob not found"
        except HttpResponseError as e:
            return False, f"Azure connection error: {str(e)}"
        except Exception as e:
            return False, f"An unexpected Azure connection error occurred: {str(e)}"

    async def check_s3_connection(
        self,
        bucket_name: str,
        access_id: str,
        access_secret: str,
    ) -> tuple[bool, str]:
        try:
            session = aioboto3.Session()
            async with session.client(
                "s3",
                aws_access_key_id=access_id,
                aws_secret_access_key=access_secret,
            ) as s3:
                await s3.head_bucket(Bucket=bucket_name)
                return True, "AWS S3 connection successful"
        except NoCredentialsError:
            return False, "Invalid AWS credentials"
        except ClientError as e:
            return False, f"AWS connection error: {str(e)}"
        except Exception as e:
            return False, f"An unexpected AWS connection error occurred: {str(e)}"

    async def check_azure_fabric_connection(
        self, container_name: str, sas_url: str
    ) -> tuple[bool, str]:
        try:
            fabric_service_client = BlobServiceClient(account_url=sas_url)
            container_client = fabric_service_client.get_container_client(
                container_name
            )
            await container_client.get_container_properties()
            return True, "Azure Fabric connection successful"
        except ResourceNotFoundError:
            return False, "Azure Fabric container or blob not found"
        except HttpResponseError as e:
            return False, f"Azure Fabric connection error: {str(e)}"
        except Exception as e:
            return (
                False,
                f"An unexpected Azure Fabric connection error occurred: {str(e)}",
            )

    async def check_sql_db_connection(
        self,
        db_type: str,
        *,
        host: str,
        port: int,
        database_name: str,
        username: str,
        password: str,
        ssl_mode: Optional[str] = None,
    ) -> tuple[bool, str]:
        """
        Check database connection with SSL support.

        Args:
            db_type: Type of database ("pg_db" or "mysql_db")
            host: Database host
            port: Database port
            database_name: Database name
            username: Database username
            password: Database password
            ssl_mode: SSL mode ("disabled", "required"")

        Returns:
            Tuple of (success: bool, message: str)
        """
        logger.info("Inside the function check sql db connection")
        try:
            if db_type == "pg_db":
                dsn = (
                    f"postgresql://{username}:{password}@{host}:{port}/{database_name}"
                )
                return await asyncio.wait_for(try_postgres_connection(dsn), timeout=20)

            elif db_type == "mysql_db":
                ssl_config = build_mysql_ssl_config(ssl_mode)
                return await asyncio.wait_for(
                    try_mysql_connection(
                        host, port, username, password, database_name, ssl_config
                    ),
                    timeout=20,
                )

            else:
                return False, f"Unsupported db_type: {db_type}"

        except asyncio.TimeoutError:
            return False, "Connection timed out."
        except (OSError, ConnectionError) as e:
            return False, f"Network error: {str(e)}"
        except Exception as e:
            return False, f"{db_type.upper()} connection error: {str(e)}"

    async def pull_files(
        self,
        *,
        workspace_id: UUID,
        user_id: UUID,
        source_id: UUID,
        db_session: AsyncSession,
    ):
        """Pulls files from various source types asynchronously."""

        logger.info(f"Pulling files from source: {source_id}")
        source = await self.get_source_info(workspace_id, source_id, db_session)
        logger.debug(f"Source details for {source_id}: {source}")
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        ### MODIFIED FOR GROOVE ###
        task_mapping = {
            # "pg_db": (
            #     "tasks.pull_tables_from_pg_db_task",
            #     self.get_sql_db_details,
            #     ["username", "password", "host", "port", "database_name"],
            # ),
            "pg_db": (
                "tasks.pull_tables_from_pg_db_task",
                lambda source_id, db_session: self.get_sql_db_details(
                    source_id, "pg_db", db_session
                ),
                ["username", "password", "host", "port", "database_name"],
                lambda **kwargs: self.check_sql_db_connection("pg_db", **kwargs),
            ),
            "azure_storage": (
                "tasks.pull_files_from_azure_source_task",
                self.get_azure_storage_details,
                ["container_name", "sas_url"],
                self.check_azure_connection,
            ),
            "azure_fabric": (
                "tasks.pull_files_from_azure_source_task",
                self.get_azure_fabric_details,
                ["container_name", "sas_url"],
                self.check_azure_fabric_connection,
            ),
            "aws_s3": (
                "tasks.pull_files_from_aws_s3_task",
                self.get_s3_storage_details,
                ["bucket_name", "access_key", "secret_key"],
                self.check_s3_connection,
            ),
            # "mysql_db": (
            #     "tasks.pull_tables_from_mysql_db_task",
            #     self.get_mysql_db_details,
            #     ["username", "password", "host", "port", "database_name"],
            #     self. check_sql_db_connection,
            # ),
            "mysql_db": (
                "tasks.pull_tables_from_mysql_db_task",
                lambda source_id, db_session: self.get_sql_db_details(
                    source_id, "mysql_db", db_session
                ),
                [
                    "username",
                    "password",
                    "host",
                    "port",
                    "database_name",
                    "ssl_mode",
                ],
                lambda **kwargs: self.check_sql_db_connection("mysql_db", **kwargs),
            ),
            "groove_source": (
                "tasks.pull_files_from_groove_source_task",
                self.get_groove_details,
                ["api_key"],
                self.check_groove_connection,
            ),
        }

        if source.source_type not in task_mapping:
            raise HTTPException(status_code=400, detail="Unsupported source type")

        task_name, details_func, required_keys, *connection_func = task_mapping[
            source.source_type
        ]

        try:
            details = await details_func(source_id, db_session)
            kwargs = {key: details.get(key) for key in required_keys}
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to fetch details: {str(e)}"
            )

        # Connection check (if applicable)
        if connection_func:
            is_connected, message = await connection_func[0](**kwargs)
            if not is_connected:
                logger.warning(f"Failed connection to {source.source_type}: {message}")
                return

        # Filter out SSL parameters for database tasks as they might not be expected by the task
        ssl_params = ["ssl_mode"]
        if source.source_type in ["mysql_db"]:
            task_kwargs = {k: v for k, v in kwargs.items() if k not in ssl_params}
        else:
            task_kwargs = kwargs

        logger.info(f"Submitting {source.source_type} pull task to Celery")
        celery.signature(
            task_name,
            kwargs={
                **task_kwargs,  # Changed from **kwargs to **task_kwargs
                "workspace_id": workspace_id,
                "user_id": user_id,
                "source_id": source_id,
            },
        ).apply_async()
        logger.debug(f"Task submitted: {task_name}")

    async def clear_old_files(
        self,
        *,
        workspace_id: UUID,
        source_id: UUID,
        db_session: AsyncSession,
    ):
        """
        Deletes previously pulled files from the File table and the local disk.
        This is typically called before re-pulling files for a given source.
        """
        try:
            logger.info(f"Clearing old files for source: {source_id}")

            # Fetch file paths before deleting from DB
            result = await db_session.execute(
                select(File).where(
                    File.workspace_id == workspace_id,
                    File.source_id == source_id,
                )
            )
            files = result.scalars().all()

            # Delete files from disk
            for file in files:
                if file.file_path and os.path.exists(file.file_path):
                    try:
                        os.remove(file.file_path)
                        logger.debug(f"Deleted file from disk: {file.file_path}")
                    except Exception as e:
                        logger.warning(
                            f"Could not delete file {file.file_path}: {str(e)}"
                        )

            # Delete records from database
            await db_session.execute(
                delete(File).where(
                    File.workspace_id == workspace_id,
                    File.source_id == source_id,
                )
            )
            await db_session.commit()

            logger.info(f"Old files cleared from DB and disk for source: {source_id}")

        except Exception as e:
            logger.error(f"Failed to clear old files for source {source_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to clear old files")

    async def needs_pdf_compression(
        self,
        file_path: str,
        pixel_threshold: int = 2000,
        pages_to_check: int = 3,
    ) -> bool:
        """
        Quick check if PDF needs compression based on image resolution.
        Returns True if compression is needed, False otherwise.
        """
        logger.info(f"Checking if PDF needs compression: {file_path}")

        try:
            doc = fitz.open(file_path)
            logger.debug(
                f"Checking first {min(pages_to_check, len(doc))} pages of {len(doc)} total pages"
            )

            for i in range(min(pages_to_check, len(doc))):
                page = doc.load_page(i)
                pix = page.get_pixmap(dpi=100)
                if pix.width > pixel_threshold or pix.height > pixel_threshold:
                    logger.info(
                        f"High resolution page found: {pix.width}x{pix.height} px"
                    )
                    doc.close()
                    return True

            doc.close()
            logger.info("No compression needed")
            return False

        except Exception as e:
            logger.error(f"Error checking PDF compression need: {str(e)}")
            return False

    def get_azure_storage_details_sync(self, source_id, db_session):
        """Synchronous version for use in Celery tasks and monitoring."""
        source = (
            db_session.query(Source)
            .filter(Source.id == source_id, Source.deleted_at.is_(None))
            .first()
        )
        if not source:
            raise Exception("Source not found or deleted.")
        azure_storage = (
            db_session.query(AzureModel)
            .filter(AzureModel.source_id == source_id)
            .first()
        )
        if not azure_storage:
            return None
        sas_url = azure_storage.sas_url
        try:
            if sas_url.startswith("vault:"):
                vault_path = sas_url.replace("vault:", "")
                if settings.DEPLOYED_ENV.startswith("azure"):
                    secret_client: SecretClient = deps.get_secret_client()
                    secret = secret_client.get_secret(vault_path)
                    actual_sas_url = secret.value
                else:
                    secret = vault_client.secrets.kv.v2.read_secret_version(
                        path=vault_path
                    )
                    actual_sas_url = secret["data"]["data"]["sas_url"]
            else:
                actual_sas_url = sas_url
        except Exception as e:
            raise Exception(f"Failed to fetch SAS URL: {str(e)}")
        return {
            "container_name": azure_storage.container_name,
            "sas_url": actual_sas_url,
        }


crud_source = CRUDSource(Source)
