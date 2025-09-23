import asyncio
import socket
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Optional, Union
from urllib.parse import quote
from uuid import UUID

import aiohttp
import asyncpg
import hvac
import urllib3
from azure.keyvault.secrets import SecretClient
from fastapi import HTTPException
from fastapi_pagination import Params
from sqlalchemy.orm import Session
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api import deps
from app.be_core.config import settings
from app.be_core.logger import logger
from app.crud.base_crud import CRUDBase
from app.models.databricks_info import DatabricksInfo  # Import Databricks model
from app.models.destination_model import Destination
from app.models.pg_vector_info import PgVectorInfo
from app.schemas.destination_schema import (
    IDestinationCreate,
)

executor = ThreadPoolExecutor()

vault_client = hvac.Client(url=settings.VAULT_ADDR, token=settings.VAULT_TOKEN)


class CRUDDestination(CRUDBase[Destination, IDestinationCreate, None]):

    async def _get_pg_vector_details(
        self,
        destination_id: UUID,
        db_session: AsyncSession,
    ) -> dict:
        secret_client: SecretClient = deps.get_secret_client()
        """
        Fetches PgVector details, including credentials from Azure Key Vault or HashiCorp Vault.
        """
        pg_vector_result = await db_session.execute(
            select(PgVectorInfo).where(PgVectorInfo.destination_id == destination_id)
        )
        pg_vector = pg_vector_result.scalar_one_or_none()

        if not pg_vector:
            logger.warning(
                f"PgVector details not found for destination {destination_id}."
            )
            return {}

        secret_path_username = f"pgvector-{quote(str(destination_id))}-username"
        secret_path_password = f"pgvector-{quote(str(destination_id))}-password"

        try:
            if settings.DEPLOYED_ENV == "azure_dev":
                secret_username_response = secret_client.get_secret(
                    secret_path_username
                )
                secret_password_response = secret_client.get_secret(
                    secret_path_password
                )

                username = secret_username_response.value
                password = secret_password_response.value

            else:
                secret_response = vault_client.secrets.kv.v2.read_secret(
                    path=secret_path_username
                )
                username = secret_response["data"]["data"].get("username")

                secret_response = vault_client.secrets.kv.v2.read_secret(
                    path=secret_path_password
                )
                password = secret_response["data"]["data"].get("password")

            return {
                "host": pg_vector.host,
                "port": pg_vector.port,
                "database_name": pg_vector.database_name,
                "username": username,
                "password": password,
            }

        except Exception as e:
            logger.error(
                f"Failed to retrieve PgVector secrets for destination {destination_id}. Error: {str(e)}"
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve PgVector secrets from the vault",
            )

    async def _get_databricks_details(
        self,
        destination_id: UUID,
        db_session: AsyncSession,
    ) -> dict:
        secret_client: SecretClient = deps.get_secret_client()
        """
        Fetches Databricks details, including the token from Azure Key Vault or HashiCorp Vault.
        """
        databricks_result = await db_session.execute(
            select(DatabricksInfo).where(
                DatabricksInfo.destination_id == destination_id
            )
        )
        databricks = databricks_result.scalar_one_or_none()

        if not databricks:
            logger.warning(
                f"Databricks details not found for destination {destination_id}."
            )
            return {}

        secret_path = f"databricks-{quote(str(destination_id))}-token"
        logger.info(f"Fetching secret from path: {secret_path}")

        try:
            if settings.DEPLOYED_ENV == "azure_dev":
                secret_response = secret_client.get_secret(secret_path)
                token = secret_response.value

            else:
                secret_response = vault_client.secrets.kv.v2.read_secret(
                    path=secret_path
                )
                token = secret_response["data"]["data"].get("token")

            return {
                "workspace_url": databricks.workspace_url,
                "token": token,
                "warehouse_id": databricks.warehouse_id,
                "database_name": databricks.database_name,
                "table_name": databricks.table_name,
            }

        except Exception as e:
            logger.error(
                f"Failed to retrieve Databricks secrets for destination {destination_id}. Error: {str(e)}"
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve Databricks secrets from the vault",
            )

    def _get_pg_vector_details_sync(
        self,
        destination_id: UUID,
        db_session: Session,
    ) -> dict:
        secret_client: SecretClient = deps.get_secret_client()
        """
        Fetches PgVector details, including credentials from Azure Key Vault or HashiCorp Vault.
        """
        pg_vector_result = db_session.execute(
            select(PgVectorInfo).where(PgVectorInfo.destination_id == destination_id)
        )
        pg_vector = pg_vector_result.scalar_one_or_none()

        if not pg_vector:
            return {}

        secret_path_username = f"pgvector-{quote(str(destination_id))}-username"
        secret_path_password = f"pgvector-{quote(str(destination_id))}-password"

        try:
            if settings.DEPLOYED_ENV == "azure_dev":

                secret_username_response = secret_client.get_secret(
                    secret_path_username
                )
                secret_password_response = secret_client.get_secret(
                    secret_path_password
                )

                username = secret_username_response.value
                password = secret_password_response.value

            else:
                secret_response = vault_client.secrets.kv.v2.read_secret(
                    path=secret_path_username
                )
                username = secret_response["data"]["data"].get("username")

                secret_response = vault_client.secrets.kv.v2.read_secret(
                    path=secret_path_password
                )
                password = secret_response["data"]["data"].get("password")

            return {
                "host": pg_vector.host,
                "port": pg_vector.port,
                "database_name": pg_vector.database_name,
                "table_name": pg_vector.table_name,
                "username": username,
                "password": password,
            }

        except Exception as e:
            logger.error(
                f"Failed to retrieve PgVector secrets for destination {destination_id}. Error: {str(e)}"
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve PgVector secrets from the vault",
            )

    def _get_databricks_details_sync(
        self,
        destination_id: UUID,
        db_session: Session,
    ) -> dict:
        secret_client: SecretClient = deps.get_secret_client()
        """
        Fetches Databricks details, including the token from Azure Key Vault or HashiCorp Vault.
        """
        databricks_result = db_session.execute(
            select(DatabricksInfo).where(
                DatabricksInfo.destination_id == destination_id
            )
        )
        databricks = databricks_result.scalar_one_or_none()

        if not databricks:
            return {}

        secret_path = f"databricks-{quote(str(destination_id))}-token"

        try:
            if settings.DEPLOYED_ENV == "azure_dev":
                secret_response = secret_client.get_secret(secret_path)
                token = secret_response.value

            else:
                secret_response = vault_client.secrets.kv.v2.read_secret(
                    path=secret_path
                )
                token = secret_response["data"]["data"].get("token")

            return {
                "workspace_url": databricks.workspace_url,
                "token": token,
                "warehouse_id": databricks.warehouse_id,
                "database_name": databricks.database_name,
                "table_name": databricks.table_name,
            }

        except Exception as e:
            logger.error(
                f"Failed to retrieve Databricks secrets for destination {destination_id}. Error: {str(e)}"
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve Databricks secrets from the vault",
            )

    async def create_destination(
        self,
        *,
        obj_in: IDestinationCreate,
        organization_id: UUID,
        destination_type: str,
        db_session: Optional[AsyncSession] = None,
    ) -> Destination:
        secret_client: SecretClient = deps.get_secret_client()
        db_session = db_session or super().get_db().session

        try:
            async with db_session.begin():
                destination = Destination(
                    name=obj_in.name,
                    description=obj_in.description,
                    is_active=obj_in.is_active,
                    organization_id=organization_id,
                    destination_type=destination_type,
                )
                db_session.add(destination)
                await db_session.flush()  # Ensure destination ID is generated

            if obj_in.pg_vector:
                secret_path_username = f"pgvector-{destination.id}-username"
                secret_path_password = f"pgvector-{destination.id}-password"
                destination_pgvector = PgVectorInfo(
                    destination_id=destination.id,
                    host=obj_in.pg_vector.host,
                    port=obj_in.pg_vector.port,
                    database_name=obj_in.pg_vector.database_name,
                    table_name=obj_in.pg_vector.table_name,
                    username_reference=secret_path_username,
                    password_reference=secret_path_password,
                )

                try:
                    if settings.DEPLOYED_ENV == "azure_dev":
                        secret_client.set_secret(
                            secret_path_username, obj_in.pg_vector.username
                        )
                        logger.info(
                            f"PgVector username stored in Azure Key Vault at {secret_path_username}"
                        )
                        secret_client.set_secret(
                            secret_path_password, obj_in.pg_vector.password
                        )
                        logger.info(
                            f"PgVector password stored in Azure Key Vault at {secret_path_password}"
                        )
                    else:
                        vault_client.secrets.kv.v2.create_or_update_secret(
                            path=secret_path_username,
                            secret={"username": obj_in.pg_vector.username},
                        )
                        logger.info(
                            f"PgVector username stored in HashiCorp Vault at {secret_path_username}"
                        )
                        vault_client.secrets.kv.v2.create_or_update_secret(
                            path=secret_path_password,
                            secret={"password": obj_in.pg_vector.password},
                        )
                        logger.info(
                            f"PgVector password stored in HashiCorp Vault at {secret_path_password}"
                        )
                except Exception as e:
                    logger.error(f"Failed to store PgVector credentials: {str(e)}")
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to store PgVector credentials in the vault.",
                    )

                async with db_session.begin():
                    db_session.add(destination_pgvector)
                    await db_session.commit()

            if obj_in.databricks:
                secret_path_token = f"databricks-{destination.id}-token"
                # secret_path_token = f"databricks-{quote(str(destination.id))}-token"
                destination_databricks = DatabricksInfo(
                    destination_id=destination.id,
                    workspace_url=obj_in.databricks.workspace_url,
                    token=secret_path_token,
                    warehouse_id=obj_in.databricks.warehouse_id,
                    database_name=obj_in.databricks.database_name,
                    table_name=obj_in.databricks.table_name,
                )

                try:
                    if settings.DEPLOYED_ENV == "azure_dev":
                        secret_client.set_secret(
                            secret_path_token, obj_in.databricks.token
                        )
                        logger.info(
                            f"Databricks token stored in Azure Key Vault at {secret_path_token}"
                        )
                    else:
                        vault_client.secrets.kv.v2.create_or_update_secret(
                            path=secret_path_token,
                            secret={"token": obj_in.databricks.token},
                        )
                        logger.info(
                            f"Databricks token stored in HashiCorp Vault at {secret_path_token}"
                        )
                except Exception as e:
                    logger.error(f"Failed to store Databricks token: {str(e)}")
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to store Databricks token in the vault.",
                    )

                async with db_session.begin():
                    db_session.add(destination_databricks)
                    await db_session.commit()

            await db_session.refresh(destination)

        except Exception as e:
            logger.error(f"Failed to create destination: {str(e)}")
            await db_session.rollback()
            raise

        return destination

    async def delete_destination(
        self,
        *,
        destination_id: UUID,
        organization_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> Destination:
        db_session = db_session or super().get_db().session
        logger.info(
            f"Attempting to soft delete destination {destination_id} for organization {organization_id}"
        )

        destination = await self.get_destination_by_id(
            destination_id=destination_id,
            organization_id=organization_id,
            db_session=db_session,
        )

        if not destination:
            logger.warning(f"Destination with ID {destination_id} not found.")
            raise HTTPException(status_code=404, detail="Destination not found")

        try:
            destination.deleted_at = datetime.now()
            await db_session.commit()

            if destination.destination_type == "pg_vector":
                destination_pgvector = await db_session.execute(
                    select(PgVectorInfo).where(
                        PgVectorInfo.destination_id == destination_id
                    )
                )
                destination_pgvector_instance = destination_pgvector.scalars().first()

                if destination_pgvector_instance:
                    destination_pgvector_instance.deleted_at = datetime.utcnow()
                    await db_session.commit()
                    logger.info(
                        f"PgVectorInfo entry for destination {destination_id} marked as deleted."
                    )

            elif destination.destination_type == "databricks":
                destination_databricks = await db_session.execute(
                    select(DatabricksInfo).where(
                        DatabricksInfo.destination_id == destination_id
                    )
                )
                destination_databricks_instance = (
                    destination_databricks.scalars().first()
                )

                if destination_databricks_instance:
                    destination_databricks_instance.deleted_at = datetime.utcnow()
                    await db_session.commit()
                    logger.info(
                        f"DatabricksInfo entry for destination {destination_id} marked as deleted."
                    )

            logger.info(f"Destination {destination_id} marked as deleted.")
            return destination

        except Exception as e:
            logger.error(
                f"Failed to soft delete destination {destination_id}: {str(e)}"
            )
            raise HTTPException(status_code=500, detail="Internal server error")

    async def get_destination_by_id(
        self,
        *,
        destination_id: UUID,
        organization_id: UUID,
        db_session: AsyncSession | None = None,
    ) -> Destination:
        db_session = db_session or super().get_db().session

        result = await db_session.execute(
            select(Destination).where(
                Destination.id == destination_id,
                Destination.organization_id == organization_id,
                Destination.deleted_at.is_(None),
            )
        )
        destination = result.scalar_one_or_none()

        if not destination:
            logger.warning(f"Destination with ID {destination_id} not found.")
            raise HTTPException(status_code=404, detail="Destination not found")

        return destination

    # def get_destination_connection_details_by_id_sync(
    #     self,
    #     *,
    #     destination_id: UUID,
    #     organization_id: UUID,
    #     db_session: Session | None = None,
    # ) -> Destination:
    #     db_session = db_session or super().get_db().session

    #     pg_vector = self._get_pg_vector_details_sync(destination_id, db_session)

    #     # Fetch DatabricksInfo if it exists
    #     databricks = self._get_databricks_details_sync(destination_id, db_session)
    #     # databricks = databricks_result.scalar_one_or_none()

    #     # Return the dictionary that is not null
    #     if pg_vector is not None:
    #         pg_vector["destination_type"] = "pg_vector"
    #         return pg_vector
    #     elif databricks is not None:
    #         databricks["destination_type"] = "databricks"
    #         return databricks
    #     else:
    #         raise HTTPException(status_code=404, detail="No connection details found")

    def get_destination_connection_details_by_id_sync(
        self,
        *,
        destination_id: UUID,
        organization_id: UUID,
        db_session: Session | None = None,
    ) -> Destination:
        db_session = db_session or super().get_db().session

        pg_vector = self._get_pg_vector_details_sync(destination_id, db_session)
        databricks = self._get_databricks_details_sync(destination_id, db_session)

        logger.debug(f"Destination ID: {destination_id}")
        logger.debug("Retrieving PG Vector and Databricks connection details")

        if pg_vector:
            pg_vector["destination_type"] = "pg_vector"
            return pg_vector

        elif databricks:
            databricks["destination_type"] = "databricks"
            return databricks

        else:
            raise HTTPException(
                status_code=404,
                detail="No connection details found for the given destination ID",
            )

    async def get_destinations(
        self,
        *,
        organization_id: UUID,
        params: Params,
        db_session: AsyncSession | None = None,
    ) -> List[Destination]:
        db_session = db_session or super().get_db().session

        query = select(Destination).where(
            Destination.organization_id == organization_id,
            Destination.deleted_at.is_(None),
        )

        results = await db_session.execute(query)
        return results.scalars().all()

    async def get_destination_info(
        self, organization_id: UUID, destination_id: UUID, db_session: AsyncSession
    ) -> Optional[Destination]:
        """
        Fetch destination details from the database.
        """
        result = await db_session.execute(
            select(Destination).where(
                Destination.id == destination_id,
                Destination.organization_id == organization_id,
                Destination.deleted_at.is_(None),
            )
        )
        return result.scalar_one_or_none()

    async def check_destination_connection(
        self, connection_details: Union[str, dict], destination_type: str
    ) -> tuple[bool, str]:
        try:
            if destination_type == "pg_vector":
                return await self._check_pg_vector_connection(connection_details)
            elif destination_type == "databricks":
                return await self._check_databricks_connection(connection_details)
            else:
                return False, "Unsupported destination type"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    async def _check_pg_vector_connection(self, db_url: str) -> tuple[bool, str]:
        """Checks the connection to a PostgreSQL database."""
        try:
            connection = await asyncpg.connect(dsn=db_url)
            await connection.close()
            return True, "PostgreSQL connection successful"
        except asyncpg.exceptions.InvalidCatalogNameError as e:
            return False, f"Invalid database name: {str(e)}"
        except asyncpg.exceptions.InvalidPasswordError as e:
            return False, f"Authentication failed: {str(e)}"
        except asyncpg.exceptions.ConnectionDoesNotExistError as e:
            return False, f"Connection to PostgreSQL server failed: {str(e)}"
        except asyncpg.exceptions.InvalidAuthorizationSpecificationError as e:
            return False, f"Invalid authorization: {str(e)}"
        except Exception as e:
            return False, f"PostgreSQL connection error: {str(e)}"

    async def _check_databricks_connection(
        self, connection_details: dict
    ) -> tuple[bool, str]:
        """Checks the connection to a Databricks SQL Warehouse using `warehouse_id` instead of `cluster_id`."""
        try:
            workspace_url = connection_details["workspace_url"].replace("https://", "")
            access_token = connection_details["token"]
            warehouse_id = connection_details["warehouse_id"]

            api_url = f"https://{workspace_url}/api/2.0/sql/warehouses/{warehouse_id}"

            urllib3.disable_warnings()

            try:
                socket.gethostbyname(workspace_url)
            except socket.gaierror:
                logger.error(
                    f"Databricks hostname '{workspace_url}' is invalid or unreachable."
                )
                return (
                    False,
                    "Invalid Databricks Credentials. Please check your connection details.",
                )

            async def _connect():
                try:
                    headers = {
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                    }
                    async with aiohttp.ClientSession() as session:
                        async with session.get(api_url, headers=headers) as response:
                            if response.status == 200:
                                return (
                                    True,
                                    "Databricks SQL Warehouse connection successful",
                                )
                            elif response.status == 403:
                                return (
                                    False,
                                    "Invalid Databricks credentials or insufficient permissions.",
                                )
                            else:
                                return (
                                    False,
                                    f"Databricks API error: {response.status} - {await response.text()}",
                                )
                except aiohttp.ClientError as e:
                    return False, f"Databricks connection error: {str(e)}"
                except Exception as e:
                    return False, f"Unexpected error: {str(e)}"

            result = await asyncio.wait_for(_connect(), timeout=10)
            return result

        except asyncio.TimeoutError:
            logger.error("Databricks connection timeout.")
            return False, "Databricks connection failed due to timeout."


destination_crud = CRUDDestination(Destination)
