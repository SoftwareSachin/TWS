# API Client Authentication System

This document describes the implementation of JWT-based API client authentication for the entities_extractor endpoint and other API endpoints.

## Overview

The API client authentication system allows backend clients to authenticate using `client_id` and `client_secret` credentials, receiving JWT tokens for subsequent API calls. This is designed for server-to-server communication where traditional user authentication is not suitable.

## Architecture

### Components

1. **ApiClient Model** (`app/models/api_client_model.py`)
   - Stores client metadata in the database
   - Links clients to organizations
   - Tracks usage and expiration

2. **API Client Security** (`app/be_core/api_client_security.py`)
   - JWT token creation and validation
   - API client authentication logic
   - UserData conversion for compatibility

3. **API Client CRUD** (`app/crud/api_client_crud.py`)
   - Database operations for API clients
   - Azure Key Vault integration for secret storage
   - Client authentication and secret management

4. **API Endpoints** (`app/api/v1/endpoints/api_client.py`)
   - CRUD operations for managing API clients
   - Token generation endpoint
   - Secret regeneration functionality

5. **Authentication Middleware** (`app/api/deps.py`)
   - Enhanced authentication dependencies
   - Support for both user and API client authentication
   - Role-based access control

## Database Schema

### ApiClient Table

```sql
CREATE TABLE api_clients (
    id UUID PRIMARY KEY,
    client_id VARCHAR UNIQUE NOT NULL,
    name VARCHAR NOT NULL,
    description VARCHAR,
    organization_id UUID NOT NULL REFERENCES organizations(id),
    is_active BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMP,
    created_by UUID REFERENCES users(id),
    last_used_at TIMESTAMP,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP
);

CREATE INDEX ix_api_clients_client_id ON api_clients(client_id);
```

### Secret Storage

Client secrets are stored securely in Azure Key Vault (or HashiCorp Vault for local development) using the naming convention:
- Secret name: `api-client-{client_id}-secret`
- Secret value: The actual client secret

## API Endpoints

### Authentication

#### POST `/api/v1/api-clients/token`

Get JWT token for API client authentication.

**Request:**
```json
{
    "client_id": "client_abc123",
    "client_secret": "secret_xyz789"
}
```

**Response:**
```json
{
    "data": {
        "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
        "token_type": "bearer",
        "expires_in": 3600,
        "client_id": "client_abc123"
    },
    "message": "API client authenticated successfully"
}
```

### CRUD Operations

#### POST `/api/v1/api-clients`

Create a new API client (requires admin/developer role).

**Request:**
```json
{
    "name": "My API Client",
    "description": "Client for external service integration",
    "organization_id": "uuid-here",
    "is_active": true,
    "expires_at": null
}
```

**Response:**
```json
{
    "data": {
        "id": "uuid-here",
        "client_id": "client_abc123",
        "client_secret": "secret_xyz789",
        "name": "My API Client",
        "description": "Client for external service integration",
        "organization_id": "uuid-here",
        "is_active": true,
        "expires_at": null,
        "created_at": "2025-07-10T05:00:00Z"
    },
    "message": "API client created successfully"
}
```

#### GET `/api/v1/api-clients`

List all API clients for the current organization.

#### GET `/api/v1/api-clients/{api_client_id}`

Get a specific API client by ID.

#### PUT `/api/v1/api-clients/{api_client_id}`

Update an API client.

#### DELETE `/api/v1/api-clients/{api_client_id}`

Delete an API client.

#### POST `/api/v1/api-clients/{api_client_id}/regenerate-secret`

Regenerate the client secret for an API client.

## Usage Examples

### 1. Creating an API Client

```python
import requests

# Create API client (requires admin/developer authentication)
response = requests.post(
    "http://localhost:8000/api/v1/api-clients",
    headers={"Authorization": "Bearer <user_token>"},
    json={
        "name": "External Service Client",
        "description": "Client for external service integration",
        "organization_id": "your-org-uuid",
        "is_active": True
    }
)

client_data = response.json()["data"]
client_id = client_data["client_id"]
client_secret = client_data["client_secret"]  # Only available during creation
```

### 2. Getting an API Token

```python
# Get JWT token for API client
response = requests.post(
    "http://localhost:8000/api/v1/api-clients/token",
    json={
        "client_id": client_id,
        "client_secret": client_secret
    }
)

token_data = response.json()["data"]
api_token = token_data["access_token"]
```

### 3. Using the API with Client Authentication

```python
# Use the API with client authentication
response = requests.post(
    "http://localhost:8000/api/v1/entities_extractor/extract",
    headers={"Authorization": f"Bearer {api_token}"},
    files={"file": open("document.pdf", "rb")},
    data={
        "schemas": json.dumps({"person": {"name": "string", "age": "integer"}}),
        "example_format": json.dumps({"single_entity": {"person": {"name": "John", "age": 30}}})
    }
)

result = response.json()
```

## Security Features

### 1. Secure Secret Storage
- Client secrets are never stored in the database
- Secrets are encrypted and stored in Azure Key Vault
- Automatic secret rotation and regeneration

### 2. JWT Token Security
- Tokens include client ID and organization ID
- Tokens have configurable expiration times
- Token type validation prevents misuse

### 3. Access Control
- API clients are scoped to organizations
- Role-based access control for management endpoints
- Client expiration and deactivation support

### 4. Audit Trail
- Last used timestamp tracking
- Creation and modification tracking
- Integration with existing audit system

## Configuration

### Environment Variables

The system uses existing configuration for Azure Key Vault:

```bash
AZURE_KEY_VAULT_NAME=your-key-vault-name
DEPLOYED_ENV=azure_dev  # or local for HashiCorp Vault
VAULT_ADDR=http://vault:8200  # for local development
VAULT_TOKEN=your-vault-token  # for local development
```

### JWT Settings

JWT tokens use the same configuration as user tokens:

```python
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1 hour default
SECRET_KEY = "your-secret-key"
```

## Migration

To set up the API client authentication system:

1. **Run the database migration:**
   ```bash
   alembic upgrade head
   ```

2. **Verify the migration:**
   ```bash
   alembic current
   ```

3. **Test the system:**
   ```bash
   python test_api_client_auth.py
   ```

## Integration with Existing Endpoints

The entities_extractor endpoint now supports both user and API client authentication:

```python
@router.post("/entities_extractor/extract")
async def extract_entities(
    file: UploadFile,
    schemas: str = Form(...),
    example_format: Optional[str] = Form(None),
    current_user: UserData = Depends(deps.get_current_user_or_api_client()),
    db: AsyncSession = Depends(deps.get_db),
):
    # This endpoint now works with both user tokens and API client tokens
    pass
```

## Best Practices

### 1. Secret Management
- Store client secrets securely and never log them
- Rotate secrets regularly using the regenerate endpoint
- Use different clients for different services

### 2. Token Management
- Cache tokens appropriately to avoid frequent authentication
- Handle token expiration gracefully
- Use HTTPS for all API communications

### 3. Monitoring
- Monitor API client usage patterns
- Set up alerts for unusual activity
- Regularly review and clean up unused clients

### 4. Security
- Use strong, randomly generated client secrets
- Set appropriate expiration dates for clients
- Regularly audit client permissions and usage

## Troubleshooting

### Common Issues

1. **Authentication Failed**
   - Verify client_id and client_secret are correct
   - Check if the client is active and not expired
   - Ensure the client belongs to the correct organization

2. **Token Validation Failed**
   - Check if the token has expired
   - Verify the token type is "api_client"
   - Ensure the client still exists and is active

3. **Azure Key Vault Errors**
   - Verify Azure Key Vault configuration
   - Check permissions for secret access
   - Ensure the vault name is correct

### Debug Mode

Enable debug logging to troubleshoot authentication issues:

```python
import logging
logging.getLogger("app.be_core.api_client_security").setLevel(logging.DEBUG)
logging.getLogger("app.crud.api_client_crud").setLevel(logging.DEBUG)
```

## Future Enhancements

1. **Rate Limiting**: Implement rate limiting for API client requests
2. **Scopes**: Add fine-grained permission scopes for API clients
3. **Webhooks**: Support webhook-based authentication
4. **Analytics**: Enhanced usage analytics and reporting
5. **Multi-tenancy**: Support for cross-organization API clients 