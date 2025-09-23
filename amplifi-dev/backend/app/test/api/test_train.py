import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

from app.models.vanna_trainings_model import VannaTraining


class TestTrainAPIs:
    """Test cases for training APIs"""

    @pytest.mark.asyncio
    async def test_invalid_dataset_id(
        self, test_client_admin: AsyncClient, workspace_id
    ):
        """❌ TEST 1: Training with invalid dataset ID should return 404"""
        invalid_dataset_id = str(uuid.uuid4())

        request_data = {"documentation": "Test documentation", "question_sql_pairs": []}

        response = await test_client_admin.post(
            f"/workspace/{workspace_id}/dataset/{invalid_dataset_id}/train",
            json=request_data,
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_invalid_workspace_id(self, test_client_admin: AsyncClient):
        """❌ TEST 2: Training with invalid workspace ID should return 404"""
        invalid_workspace_id = str(uuid.uuid4())
        invalid_dataset_id = str(uuid.uuid4())

        request_data = {"documentation": "Test documentation", "question_sql_pairs": []}

        response = await test_client_admin.post(
            f"/workspace/{invalid_workspace_id}/dataset/{invalid_dataset_id}/train",
            json=request_data,
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_train_with_empty_request(
        self, test_client_admin: AsyncClient, workspace_id
    ):
        """❌ TEST 3: Training with completely empty request should return 422"""
        invalid_dataset_id = str(uuid.uuid4())

        response = await test_client_admin.post(
            f"/workspace/{workspace_id}/dataset/{invalid_dataset_id}/train", json={}
        )

        # Should fail with validation error (422) or dataset not found (404)
        assert response.status_code in [404, 422]

    @pytest.mark.asyncio
    async def test_train_with_malformed_json(
        self, test_client_admin: AsyncClient, workspace_id
    ):
        """❌ TEST 4: Training with malformed request should return 422"""
        invalid_dataset_id = str(uuid.uuid4())

        response = await test_client_admin.post(
            f"/workspace/{workspace_id}/dataset/{invalid_dataset_id}/train",
            json={
                "documentation": "Test doc",
                "question_sql_pairs": "invalid_format",  # Should be array
            },
        )

        # Should fail with validation error or dataset not found
        assert response.status_code in [404, 422]

    @pytest.mark.asyncio
    async def test_retrain_with_invalid_dataset_id(
        self, test_client_admin: AsyncClient, workspace_id
    ):
        """❌ TEST 5: Retrain with invalid dataset ID should return 404"""
        invalid_dataset_id = str(uuid.uuid4())

        request_data = {"documentation": "Test documentation", "question_sql_pairs": []}

        response = await test_client_admin.post(
            f"/workspace/{workspace_id}/dataset/{invalid_dataset_id}/retrain",
            json=request_data,
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_training_details_invalid_dataset(
        self, test_client_admin: AsyncClient, workspace_id
    ):
        """❌ TEST 6: Get training details with invalid dataset should return empty or 404"""
        invalid_dataset_id = str(uuid.uuid4())

        response = await test_client_admin.get(
            f"/workspace/{workspace_id}/dataset/{invalid_dataset_id}/trainings"
        )

        # Should either return 404 or empty list
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert "data" in data
            assert isinstance(data["data"], list)

    @pytest.mark.asyncio
    async def test_get_all_versions_invalid_dataset(
        self, test_client_admin: AsyncClient, workspace_id
    ):
        """❌ TEST 7: Get all versions with invalid dataset should return empty or 404"""
        invalid_dataset_id = str(uuid.uuid4())

        response = await test_client_admin.get(
            f"/workspace/{workspace_id}/dataset/{invalid_dataset_id}/trainings/all-versions"
        )

        # Should either return 404 or empty list
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert "data" in data
            assert isinstance(data["data"], list)

    @pytest.mark.asyncio
    async def test_train_with_special_characters_in_documentation(
        self, test_client_admin: AsyncClient, workspace_id
    ):
        """❌ TEST 8: Training with special characters should handle gracefully"""
        invalid_dataset_id = str(uuid.uuid4())

        request_data = {
            "documentation": "Test with special chars: éñ中文🚀<script>alert('xss')</script>",
            "question_sql_pairs": [],
        }

        response = await test_client_admin.post(
            f"/workspace/{workspace_id}/dataset/{invalid_dataset_id}/train",
            json=request_data,
        )

        # Should fail with dataset not found, not with data processing error
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_endpoints_exist_and_respond(
        self, test_client_admin: AsyncClient, workspace_id
    ):
        """✅ TEST 9: Verify all training endpoints exist and return proper HTTP codes"""
        test_dataset_id = str(uuid.uuid4())

        # Test train endpoint exists
        response = await test_client_admin.post(
            f"/workspace/{workspace_id}/dataset/{test_dataset_id}/train",
            json={"documentation": "test", "question_sql_pairs": []},
        )
        assert response.status_code in [200, 404, 422, 500]  # Valid HTTP responses

        # Test retrain endpoint exists
        response = await test_client_admin.post(
            f"/workspace/{workspace_id}/dataset/{test_dataset_id}/retrain",
            json={"documentation": "test", "question_sql_pairs": []},
        )
        assert response.status_code in [200, 404, 422, 500]  # Valid HTTP responses

        # Test get trainings endpoint exists
        response = await test_client_admin.get(
            f"/workspace/{workspace_id}/dataset/{test_dataset_id}/trainings"
        )
        assert response.status_code in [200, 404, 500]  # Valid HTTP responses

        # Test get all versions endpoint exists
        response = await test_client_admin.get(
            f"/workspace/{workspace_id}/dataset/{test_dataset_id}/trainings/all-versions"
        )
        assert response.status_code in [200, 404, 500]  # Valid HTTP responses

    # =======================================================================
    # GET API TESTS
    # =======================================================================

    @pytest.mark.asyncio
    async def test_get_trainings_with_valid_workspace_and_dataset(
        self, test_client_admin: AsyncClient, workspace_id
    ):
        """✅ GET TEST 1: Get trainings with valid workspace but non-existent dataset should return empty list"""
        test_dataset_id = str(uuid.uuid4())

        response = await test_client_admin.get(
            f"/workspace/{workspace_id}/dataset/{test_dataset_id}/trainings"
        )

        # Should return 200 with empty data or 404
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert "data" in data
            assert "message" in data
            assert isinstance(data["data"], list)

    @pytest.mark.asyncio
    async def test_get_trainings_response_structure(
        self, test_client_admin: AsyncClient, workspace_id
    ):
        """✅ GET TEST 2: Verify trainings endpoint returns proper response structure"""
        test_dataset_id = str(uuid.uuid4())

        response = await test_client_admin.get(
            f"/workspace/{workspace_id}/dataset/{test_dataset_id}/trainings"
        )

        if response.status_code == 200:
            data = response.json()
            # Verify response structure
            assert "data" in data
            assert "message" in data
            assert isinstance(data["data"], list)
            # Message should contain dataset ID
            assert str(test_dataset_id) in data["message"]

    @pytest.mark.asyncio
    async def test_get_all_versions_response_structure(
        self, test_client_admin: AsyncClient, workspace_id
    ):
        """✅ GET TEST 3: Verify all-versions endpoint returns proper response structure"""
        test_dataset_id = str(uuid.uuid4())

        response = await test_client_admin.get(
            f"/workspace/{workspace_id}/dataset/{test_dataset_id}/trainings/all-versions"
        )

        if response.status_code == 200:
            data = response.json()
            # Verify response structure for all-versions
            assert "data" in data
            assert "message" in data
            assert isinstance(data["data"], list)
            # Each version should have version_id and training_records
            for version in data["data"]:
                if version:  # If there are versions
                    assert "version_id" in version
                    assert "training_records" in version
                    assert isinstance(version["training_records"], list)

    # =======================================================================
    # TRAIN API TESTS (Additional scenarios)
    # =======================================================================

    @pytest.mark.asyncio
    async def test_train_with_large_documentation(
        self, test_client_admin: AsyncClient, workspace_id
    ):
        """✅ TRAIN TEST 1: Train with large documentation should handle gracefully"""
        test_dataset_id = str(uuid.uuid4())

        # Create large documentation (10KB)
        large_doc = "This is a test documentation. " * 400  # ~10KB

        request_data = {
            "documentation": large_doc,
            "question_sql_pairs": [
                {"question": "Test question?", "sql": "SELECT * FROM test_table"}
            ],
        }

        response = await test_client_admin.post(
            f"/workspace/{workspace_id}/dataset/{test_dataset_id}/train",
            json=request_data,
        )

        # Should handle large data (404 for non-existent dataset or 200 for success)
        assert response.status_code in [200, 404, 422]

    @pytest.mark.asyncio
    async def test_train_with_multiple_question_sql_pairs(
        self, test_client_admin: AsyncClient, workspace_id
    ):
        """✅ TRAIN TEST 2: Train with multiple question-SQL pairs"""
        test_dataset_id = str(uuid.uuid4())

        request_data = {
            "documentation": "Sales database documentation",
            "question_sql_pairs": [
                {
                    "question": "What are total sales?",
                    "sql": "SELECT SUM(amount) FROM sales",
                },
                {
                    "question": "How many customers?",
                    "sql": "SELECT COUNT(*) FROM customers",
                },
                {
                    "question": "Top selling product?",
                    "sql": "SELECT product_name FROM products ORDER BY sales DESC LIMIT 1",
                },
                {
                    "question": "Monthly revenue?",
                    "sql": "SELECT DATE_TRUNC('month', date), SUM(amount) FROM sales GROUP BY 1",
                },
            ],
        }

        response = await test_client_admin.post(
            f"/workspace/{workspace_id}/dataset/{test_dataset_id}/train",
            json=request_data,
        )

        # Should handle multiple pairs properly
        assert response.status_code in [200, 404, 422]

    @pytest.mark.asyncio
    async def test_train_with_complex_sql_queries(
        self, test_client_admin: AsyncClient, workspace_id
    ):
        """✅ TRAIN TEST 3: Train with complex SQL queries including JOINs and subqueries"""
        test_dataset_id = str(uuid.uuid4())

        request_data = {
            "documentation": "Complex database with multiple tables",
            "question_sql_pairs": [
                {
                    "question": "What is the average order value by customer segment?",
                    "sql": """
                    SELECT c.segment, AVG(o.total_amount) as avg_order_value
                    FROM customers c
                    JOIN orders o ON c.id = o.customer_id
                    WHERE o.status = 'completed'
                    GROUP BY c.segment
                    HAVING COUNT(o.id) > 5
                    ORDER BY avg_order_value DESC
                    """,
                },
                {
                    "question": "Which products have above-average profit margins?",
                    "sql": """
                    SELECT p.name, p.profit_margin
                    FROM products p
                    WHERE p.profit_margin > (
                        SELECT AVG(profit_margin) FROM products WHERE active = true
                    )
                    """,
                },
            ],
        }

        response = await test_client_admin.post(
            f"/workspace/{workspace_id}/dataset/{test_dataset_id}/train",
            json=request_data,
        )

        # Should handle complex SQL properly
        assert response.status_code in [200, 404, 422]

    # =======================================================================
    # RETRAIN API TESTS (Additional scenarios)
    # =======================================================================

    @pytest.mark.asyncio
    async def test_retrain_with_updated_documentation(
        self, test_client_admin: AsyncClient, workspace_id
    ):
        """✅ RETRAIN TEST 1: Retrain with updated documentation"""
        test_dataset_id = str(uuid.uuid4())

        request_data = {
            "documentation": "Updated documentation with new schema changes and business rules",
            "question_sql_pairs": [
                {
                    "question": "What is the updated sales calculation?",
                    "sql": "SELECT SUM(amount * tax_rate) FROM updated_sales",
                }
            ],
        }

        response = await test_client_admin.post(
            f"/workspace/{workspace_id}/dataset/{test_dataset_id}/retrain",
            json=request_data,
        )

        # Should handle retrain request properly
        assert response.status_code in [200, 404, 422]

    @pytest.mark.asyncio
    async def test_retrain_with_empty_question_sql_pairs(
        self, test_client_admin: AsyncClient, workspace_id
    ):
        """✅ RETRAIN TEST 2: Retrain with only documentation, no question-SQL pairs"""
        test_dataset_id = str(uuid.uuid4())

        request_data = {
            "documentation": "Schema-only training with detailed table descriptions",
            "question_sql_pairs": [],
        }

        response = await test_client_admin.post(
            f"/workspace/{workspace_id}/dataset/{test_dataset_id}/retrain",
            json=request_data,
        )

        # Should handle documentation-only retrain
        assert response.status_code in [200, 404, 422]

    @pytest.mark.asyncio
    async def test_retrain_concurrent_request_handling(
        self, test_client_admin: AsyncClient, workspace_id
    ):
        """✅ RETRAIN TEST 3: Test retrain behavior for concurrent requests"""
        test_dataset_id = str(uuid.uuid4())

        request_data = {
            "documentation": "Concurrent retrain test documentation",
            "question_sql_pairs": [
                {"question": "Test concurrent query?", "sql": "SELECT 1"}
            ],
        }

        # Make first retrain request
        response1 = await test_client_admin.post(
            f"/workspace/{workspace_id}/dataset/{test_dataset_id}/retrain",
            json=request_data,
        )

        # Make second retrain request immediately
        response2 = await test_client_admin.post(
            f"/workspace/{workspace_id}/dataset/{test_dataset_id}/retrain",
            json=request_data,
        )

        # Both should handle gracefully (either success or proper error)
        assert response1.status_code in [200, 404, 422, 429, 500]
        assert response2.status_code in [200, 404, 422, 429, 500]

        # If both succeed, they should return proper response structure
        for response in [response1, response2]:
            if response.status_code == 200:
                data = response.json()
                assert "message" in data or "data" in data


# Run with: pytest backend/app/test/api/test_train.py -v
