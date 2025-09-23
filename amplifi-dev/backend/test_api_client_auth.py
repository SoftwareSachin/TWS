#!/usr/bin/env python3
"""
Test script for API client authentication functionality.
This script demonstrates how to use the new API client authentication system.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent / "app"))

from app.api.deps import get_db
from app.crud.api_client_crud import api_client
from app.schemas.api_client_schema import ApiClientCreate
from app.be_core.api_client_security import create_api_client_token, validate_api_client_token
from app.be_core.config import settings
from uuid import uuid4


async def test_api_client_creation():
    """Test creating an API client"""
    print("=== Testing API Client Creation ===")
    
    # Get database session
    async for db_session in get_db():
        try:
            # Create a test API client
            api_client_data = ApiClientCreate(
                name="Test API Client",
                description="A test API client for demonstration",
                organization_id=uuid4(),  # You'll need a real organization ID
                expires_at=None
            )
            
            # Create the API client
            created_client, client_secret = await api_client.create(
                obj_in=api_client_data,
                created_by=uuid4(),  # You'll need a real user ID
                db_session=db_session
            )
            
            print(f"✅ API Client created successfully!")
            print(f"   Client ID: {created_client.client_id}")
            print(f"   Client Secret: {getattr(created_client, 'client_secret', 'N/A')}")
            print(f"   Name: {created_client.name}")
            print(f"   Organization ID: {created_client.organization_id}")
            
            return created_client
            
        except Exception as e:
            print(f"❌ Error creating API client: {e}")
            return None


async def test_api_client_authentication():
    """Test API client authentication"""
    print("\n=== Testing API Client Authentication ===")
    
    # Get database session
    async for db_session in get_db():
        try:
            # First, create a test client
            test_client = await test_api_client_creation()
            if not test_client:
                print("❌ Cannot test authentication without a test client")
                return
            
            # Test authentication with the created client
            client_secret = getattr(test_client, 'client_secret', None)
            if not client_secret:
                print("❌ No client secret available for testing")
                return
            
            # Authenticate the client
            authenticated_client = await api_client.authenticate_client(
                client_id=test_client.client_id,
                client_secret=client_secret,
                db_session=db_session
            )
            
            if authenticated_client:
                print("✅ API Client authenticated successfully!")
                print(f"   Client ID: {authenticated_client.client_id}")
                print(f"   Name: {authenticated_client.name}")
            else:
                print("❌ API Client authentication failed")
                
        except Exception as e:
            print(f"❌ Error during authentication test: {e}")


async def test_jwt_token_creation():
    """Test JWT token creation for API clients"""
    print("\n=== Testing JWT Token Creation ===")
    
    try:
        # Create a test token
        test_client_id = "test_client_123"
        test_organization_id = uuid4()
        
        # Create JWT token
        token = create_api_client_token(
            client_id=test_client_id,
            organization_id=test_organization_id
        )
        
        print(f"✅ JWT Token created successfully!")
        print(f"   Token: {token[:50]}...")
        
        # Validate the token
        payload = validate_api_client_token(token)
        print(f"✅ JWT Token validated successfully!")
        print(f"   Client ID: {payload.get('sub')}")
        print(f"   Organization ID: {payload.get('organization_id')}")
        print(f"   Token Type: {payload.get('type')}")
        
    except Exception as e:
        print(f"❌ Error during JWT token test: {e}")


async def test_api_client_crud_operations():
    """Test CRUD operations for API clients"""
    print("\n=== Testing API Client CRUD Operations ===")
    
    # Get database session
    async for db_session in get_db():
        try:
            # Create a test organization ID (you'll need a real one in practice)
            test_org_id = uuid4()
            test_user_id = uuid4()
            
            # Test 1: Create API client
            api_client_data = ApiClientCreate(
                name="CRUD Test Client",
                description="Testing CRUD operations",
                organization_id=test_org_id
            )
            
            created_client = await api_client.create(
                obj_in=api_client_data,
                created_by=test_user_id,
                db_session=db_session
            )
            print("✅ Create operation successful")
            
            # Test 2: Get by client ID
            retrieved_client = await api_client.get_by_client_id(
                client_id=created_client.client_id,
                db_session=db_session
            )
            if retrieved_client:
                print("✅ Get by client ID operation successful")
            else:
                print("❌ Get by client ID operation failed")
            
            # Test 3: Get by organization
            org_clients = await api_client.get_by_organization(
                organization_id=test_org_id,
                db_session=db_session
            )
            print(f"✅ Get by organization operation successful - Found {len(org_clients)} clients")
            
            # Test 4: Regenerate secret
            new_secret = await api_client.regenerate_secret(
                id=created_client.id,
                db_session=db_session
            )
            print(f"✅ Regenerate secret operation successful - New secret: {new_secret[:10]}...")
            
            # Test 5: Delete client
            await api_client.delete(
                id=created_client.id,
                db_session=db_session
            )
            print("✅ Delete operation successful")
            
        except Exception as e:
            print(f"❌ Error during CRUD operations test: {e}")


async def main():
    """Main test function"""
    print("🚀 Starting API Client Authentication Tests\n")
    
    # Test JWT token creation (doesn't require database)
    await test_jwt_token_creation()
    
    # Test CRUD operations
    await test_api_client_crud_operations()
    
    # Test authentication flow
    await test_api_client_authentication()
    
    print("\n🎉 All tests completed!")


if __name__ == "__main__":
    asyncio.run(main()) 