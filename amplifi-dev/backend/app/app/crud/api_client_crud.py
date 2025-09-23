import re
import secrets
import string
from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import HTTPException
from fastapi_pagination import Params, paginate
from sqlalchemy import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.api import deps
from app.be_core.config import settings
from app.be_core.logger import logger
from app.crud.base_crud import CRUDBase
from app.models.api_client_model import ApiClient
from app.schemas.api_client_schema import ApiClientCreate, ApiClientUpdate
from app.utils.datetime_utils import ensure_naive_datetime


class CRUDApiClient(CRUDBase[ApiClient, ApiClientCreate, ApiClientUpdate]):
    """CRUD operations for API clients"""

    def _generate_client_id(self) -> str:
        """Generate a unique client ID"""
        return f"client_{secrets.token_urlsafe(16)}"

    def _generate_client_secret(self) -> str:
        """Generate a secure client secret"""
        # Generate a 32-character random string with letters, digits, and special characters
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return "".join(secrets.choice(alphabet) for _ in range(32))

    def _get_secret_name(self, client_id: str) -> str:
        """Generate the secret name for Azure Key Vault"""
        # Only allow alphanumeric and dashes, replace others with dash
        safe_client_id = re.sub(r"[^a-zA-Z0-9-]", "-", client_id)
        # Remove leading/trailing dashes (if any)
        safe_client_id = safe_client_id.strip("-")
        return f"api-client-{safe_client_id}-secret"

    async def create(
        self,
        *,
        obj_in: ApiClientCreate,
        organization_id: UUID,
        created_by: UUID,
        db_session: AsyncSession,
    ) -> ApiClient:
        """Create a new API client with client_secret stored in Azure Key Vault"""
        # Generate client credentials
        client_id = self._generate_client_id()
        client_secret = self._generate_client_secret()

        # Create the API client record
        db_obj = ApiClient(
            client_id=client_id,
            name=obj_in.name,
            description=obj_in.description,
            organization_id=organization_id,
            expires_at=obj_in.expires_at,
        )

        db_session.add(db_obj)
        await db_session.flush()

        # Store client_secret in Azure Key Vault
        try:
            secret_client = deps.get_secret_client()
            secret_name = self._get_secret_name(client_id)

            if settings.DEPLOYED_ENV.startswith("azure"):
                secret_client.set_secret(secret_name, client_secret)
                logger.info(f"Client secret stored in Azure Key Vault: {secret_name}")
            else:
                # For local development, use HashiCorp Vault
                import hvac

                vault_client = hvac.Client(
                    url=settings.VAULT_ADDR, token=settings.VAULT_TOKEN
                )
                vault_client.secrets.kv.v2.create_or_update_secret(
                    path=secret_name,
                    secret={"client_secret": client_secret},
                )
                logger.info(f"Client secret stored in HashiCorp Vault: {secret_name}")

        except Exception as e:
            logger.error(f"Failed to store client secret in vault: {str(e)}")
            await db_session.rollback()
            raise HTTPException(
                status_code=500, detail="Failed to store client secret securely"
            )

        await db_session.commit()
        await db_session.refresh(db_obj)

        return db_obj, client_secret

    async def get_by_client_id(
        self,
        *,
        client_id: str,
        db_session: AsyncSession,
    ) -> Optional[ApiClient]:
        """Get API client by client_id"""
        result = await db_session.execute(
            select(ApiClient).where(ApiClient.client_id == client_id)
        )
        return result.scalar_one_or_none()

    async def get_by_organization(
        self,
        *,
        organization_id: UUID,
        pagination_params: Params = Params(),
        db_session: AsyncSession,
    ) -> list[ApiClient]:
        """Get all API clients for an organization"""
        result = await db_session.execute(
            select(ApiClient).where(
                ApiClient.organization_id == organization_id,
                ApiClient.deleted_at.is_(None),
            )
        )
        api_clients = result.scalars().all()
        return paginate(api_clients, pagination_params)

    async def authenticate_client(
        self,
        *,
        client_id: str,
        client_secret: str,
        db_session: AsyncSession,
    ) -> Optional[ApiClient]:
        """Authenticate an API client using client_id and client_secret"""
        # Get the API client
        api_client = await self.get_by_client_id(
            client_id=client_id, db_session=db_session
        )

        if not api_client:
            return None

        # Check if client has expired
        if api_client.expires_at and ensure_naive_datetime(
            api_client.expires_at
        ) < ensure_naive_datetime(datetime.utcnow()):
            logger.info(f"Client {client_id} has expired")
            return None

        # Verify client_secret from Azure Key Vault
        try:
            secret_client = deps.get_secret_client()
            secret_name = self._get_secret_name(client_id)

            if settings.DEPLOYED_ENV.startswith("azure"):
                stored_secret = secret_client.get_secret(secret_name).value
            else:
                # For local development, use HashiCorp Vault
                import hvac

                vault_client = hvac.Client(
                    url=settings.VAULT_ADDR, token=settings.VAULT_TOKEN
                )
                secret_response = vault_client.secrets.kv.v2.read_secret_version(
                    path=secret_name
                )
                stored_secret = secret_response["data"]["data"]["client_secret"]

            if stored_secret != client_secret:
                logger.info(f"Client {client_id} has invalid secret")
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve client secret from vault: {str(e)}")
            return None

        # Update last_used_at
        api_client.last_used_at = ensure_naive_datetime(datetime.utcnow())
        await db_session.commit()

        return api_client

    async def delete(
        self,
        *,
        id: UUID,
        db_session: AsyncSession,
    ) -> ApiClient:
        """Delete an API client and remove its secret from Azure Key Vault"""
        api_client = await db_session.get(ApiClient, id)
        if not api_client:
            raise HTTPException(status_code=404, detail="API client not found")

        # Remove client_secret from Azure Key Vault
        try:
            secret_client = deps.get_secret_client()
            secret_name = self._get_secret_name(api_client.client_id)

            if settings.DEPLOYED_ENV.startswith("azure"):
                secret_client.begin_delete_secret(secret_name)
                logger.info(
                    f"Client secret deleted from Azure Key Vault: {secret_name}"
                )
            else:
                # For local development, use HashiCorp Vault
                import hvac

                vault_client = hvac.Client(
                    url=settings.VAULT_ADDR, token=settings.VAULT_TOKEN
                )
                vault_client.secrets.kv.v2.delete_metadata_and_all_versions(
                    path=secret_name
                )
                logger.info(
                    f"Client secret deleted from HashiCorp Vault: {secret_name}"
                )

        except Exception as e:
            logger.error(f"Failed to delete client secret from vault: {str(e)}")
            # Continue with deletion even if vault cleanup fails

        # Delete the database record
        return await super().remove(id=id, db_session=db_session)

    async def regenerate_secret(
        self,
        *,
        id: UUID,
        db_session: AsyncSession,
    ) -> str:
        """Regenerate client_secret for an API client"""
        api_client = await db_session.get(ApiClient, id)
        if not api_client:
            raise HTTPException(status_code=404, detail="API client not found")

        # Generate new client_secret
        new_client_secret = self._generate_client_secret()

        # Update client_secret in Azure Key Vault
        try:
            secret_client = deps.get_secret_client()
            secret_name = self._get_secret_name(api_client.client_id)

            if settings.DEPLOYED_ENV.startswith("azure"):
                secret_client.set_secret(secret_name, new_client_secret)
                logger.info(
                    f"Client secret regenerated in Azure Key Vault: {secret_name}"
                )
            else:
                # For local development, use HashiCorp Vault
                import hvac

                vault_client = hvac.Client(
                    url=settings.VAULT_ADDR, token=settings.VAULT_TOKEN
                )
                vault_client.secrets.kv.v2.create_or_update_secret(
                    path=secret_name,
                    secret={"client_secret": new_client_secret},
                )
                logger.info(
                    f"Client secret regenerated in HashiCorp Vault: {secret_name}"
                )

        except Exception as e:
            logger.error(f"Failed to regenerate client secret in vault: {str(e)}")
            raise HTTPException(
                status_code=500, detail="Failed to regenerate client secret securely"
            )

        return new_client_secret

    async def update(
        self,
        *,
        db_obj: ApiClient,
        obj_in: ApiClientUpdate,
        db_session: AsyncSession,
    ) -> ApiClient:
        update_data = obj_in.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_obj, field, value)
        db_session.add(db_obj)
        await db_session.commit()
        await db_session.refresh(db_obj)
        return db_obj


# Create CRUD instance
api_client = CRUDApiClient(ApiClient)
