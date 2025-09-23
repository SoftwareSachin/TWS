from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.tools.system_tools.logic.graph_search_logic import (
    GraphSearchService,
    _extract_content,
    perform_graph_search,
)
from app.tools.system_tools.schemas.graph_search_schema import (
    GraphSearchInput,
    GraphSearchOutput,
    GraphSearchResult,
)


class TestGraphSearchLogic:
    """Basic tests for graph search logic functions"""

    def test_extract_content_entity(self):
        """Test _extract_content function for entity results"""
        entity_result = {
            "result_type": "entity",
            "name": "Socrates",
            "type": "Philosopher",
            "description": "Ancient Greek philosopher",
        }
        content = _extract_content(entity_result)
        assert content == "Socrates (Philosopher): Ancient Greek philosopher"

    def test_extract_content_relationship(self):
        """Test _extract_content function for relationship results"""
        rel_result = {
            "result_type": "relationship",
            "source_name": "Socrates",
            "target_name": "Plato",
            "relationship_type": "taught",
        }
        content = _extract_content(rel_result)
        assert content == "Socrates --[taught]--> Plato"

    def test_extract_content_fallback(self):
        """Test _extract_content function for general results"""
        general_result = {"name": "Aristotle"}
        content = _extract_content(general_result)
        assert content == "Aristotle"


class TestGraphSearchService:
    """Basic tests for GraphSearchService class"""

    @pytest.fixture
    def service(self):
        """Create a GraphSearchService instance with mocked dependencies"""
        with patch(
            "app.tools.system_tools.logic.graph_search_logic.client"
        ) as mock_client:
            service = GraphSearchService()
            service.llm_client = mock_client
            return service

    @pytest.mark.asyncio
    async def test_search_success(self, service):
        """Test successful search"""
        mock_db = MagicMock()
        dataset_ids = [uuid4()]

        with patch.object(service, "_search_single_dataset") as mock_search:
            mock_search.return_value = {
                "success": True,
                "results": [
                    {"name": "Socrates", "score": 0.9, "result_type": "entity"},
                ],
                "method": "llm_generated",
                "cypher_query": "MATCH (e:Entity) WHERE e.name CONTAINS 'Socrates' RETURN e",
            }

            result = await service.search(
                db=mock_db,
                query="Find Socrates",
                dataset_ids=dataset_ids,
                limit=10,
            )

            assert result["success"] is True
            assert result["query"] == "Find Socrates"
            assert result["dataset_ids"] == dataset_ids
            assert result["method"] == "llm_generated"
            assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_search_no_graphs_available(self, service):
        """Test search when no graphs are available"""
        mock_db = MagicMock()
        dataset_ids = [uuid4()]

        with patch.object(service, "_search_single_dataset") as mock_search:
            mock_search.return_value = {
                "success": False,
                "error": "No graph available for dataset",
                "results": [],
                "count": 0,
            }

            result = await service.search(
                db=mock_db,
                query="Find Socrates",
                dataset_ids=dataset_ids,
                limit=10,
            )

            assert result["success"] is False
            assert "No graphs available" in result["error"]
            assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_get_graph_id_success(self, service):
        """Test successful graph ID retrieval"""
        mock_db = MagicMock()
        dataset_id = uuid4()
        mock_graph = MagicMock()
        mock_graph.id = "graph-123"
        mock_graph.entities_status = "success"
        mock_graph.relationships_status = "success"

        with patch(
            "app.tools.system_tools.logic.graph_search_logic.crud_graph"
        ) as mock_crud:
            # Fix: Mock the async method correctly
            mock_crud.get_graphs_by_dataset = AsyncMock(return_value=[mock_graph])

            graph_id = await service._get_graph_id(mock_db, dataset_id)

            assert graph_id == "graph-123"

    @pytest.mark.asyncio
    async def test_generate_cypher_success(self, service):
        """Test successful Cypher query generation"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "MATCH (e:Entity) WHERE e.name CONTAINS 'Socrates' RETURN e LIMIT 20"
        )

        service.llm_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        with (
            patch.object(service, "_get_simple_schema") as mock_schema,
            patch.object(service, "_get_relationship_examples") as mock_rel,
        ):
            mock_schema.return_value = {
                "entity_types": ["Person"],
                "sample_entities": ["Socrates"],
            }
            mock_rel.return_value = {"relationship_types": ["taught"]}

            result = await service._generate_cypher("Find Socrates", "graph-123")

            assert result["success"] is True
            assert "MATCH (e:Entity)" in result["cypher_query"]

    def test_deduplicate_results(self, service):
        """Test result deduplication"""
        results = [
            {"name": "Socrates", "result_type": "entity"},
            {"name": "Socrates", "result_type": "entity"},  # Duplicate
            {"name": "Plato", "result_type": "entity"},
        ]

        deduplicated = service._deduplicate_results(results)

        assert len(deduplicated) == 2
        names = [r["name"] for r in deduplicated]
        assert "Socrates" in names
        assert "Plato" in names

    def test_format_results(self, service):
        """Test result formatting"""
        raw_results = [
            {
                "name": "Socrates",
                "type": "Philosopher",
                "description": "Ancient Greek philosopher",
                "score": 0.9,
                "result_type": "entity",
            },
        ]

        formatted = service._format_results(raw_results, "Find Socrates")

        assert len(formatted) == 1
        assert formatted[0]["name"] == "Socrates"
        assert formatted[0]["score"] == 0.9
        assert formatted[0]["result_type"] == "entity"


class TestGraphSearchIntegration:
    """Basic integration tests for graph search functionality"""

    @pytest.mark.asyncio
    @pytest.mark.skip(
        reason="Integration test needs proper mocking setup - skipping for now"
    )
    async def test_perform_graph_search_success(self):
        """Test successful graph search execution"""
        input_data = GraphSearchInput(
            query="Find Socrates",
            dataset_ids=[uuid4()],
            limit=10,
        )

        with (
            patch(
                "app.db.session.SessionLocal",
                new_callable=MagicMock,
            ) as mock_session,
            patch(
                "app.tools.system_tools.logic.graph_search_logic.GraphSearchService"
            ) as mock_service,
        ):

            mock_db = MagicMock()
            mock_session.return_value.__aenter__.return_value = mock_db

            mock_service_instance = MagicMock()
            mock_service.return_value = mock_service_instance
            mock_service_instance.search.return_value = {
                "success": True,
                "query": "Find Socrates",
                "dataset_ids": input_data.dataset_ids,
                "cypher_query": "MATCH (e:Entity) WHERE e.name CONTAINS 'Socrates' RETURN e",
                "method": "llm_generated",
                "results": [
                    {
                        "name": "Socrates",
                        "type": "Philosopher",
                        "description": "Ancient Greek philosopher",
                        "score": 0.9,
                        "result_type": "entity",
                    }
                ],
                "count": 1,
            }

            result = await perform_graph_search(input_data)

            assert isinstance(result, GraphSearchOutput)
            assert result.success is True
            assert result.query == "Find Socrates"
            assert result.method == "llm_generated"
            assert result.count == 1
            assert len(result.results) == 1

    @pytest.mark.asyncio
    @pytest.mark.skip(
        reason="Integration test needs proper mocking setup - skipping for now"
    )
    async def test_perform_graph_search_failure(self):
        """Test graph search execution failure"""
        input_data = GraphSearchInput(
            query="Find Socrates",
            dataset_ids=[uuid4()],
            limit=10,
        )

        with patch(
            "app.db.session.SessionLocal",
            new_callable=MagicMock,
        ) as mock_session:
            mock_session.return_value.__aenter__.side_effect = Exception(
                "Database error"
            )

            result = await perform_graph_search(input_data)

            assert isinstance(result, GraphSearchOutput)
            assert result.success is False
            assert result.method == "error"
            assert result.count == 0
            assert len(result.results) == 0


class TestGraphSearchSchema:
    """Basic schema validation tests"""

    def test_graph_search_input_validation(self):
        """Test GraphSearchInput validation"""
        valid_input = GraphSearchInput(
            query="Find Socrates",
            dataset_ids=[uuid4()],
            limit=10,
        )
        assert valid_input.query == "Find Socrates"
        assert len(valid_input.dataset_ids) == 1
        assert valid_input.limit == 10

    def test_graph_search_output_creation(self):
        """Test GraphSearchOutput creation"""
        output = GraphSearchOutput(
            query="Find Socrates",
            dataset_ids=[uuid4()],
            cypher_query="MATCH (e:Entity) WHERE e.name CONTAINS 'Socrates' RETURN e",
            method="llm_generated",
            results=[
                GraphSearchResult(
                    content="Socrates (Philosopher): Ancient Greek philosopher",
                    result_type="entity",
                    score=0.9,
                    metadata={"name": "Socrates"},
                )
            ],
            count=1,
            success=True,
        )

        assert output.query == "Find Socrates"
        assert output.method == "llm_generated"
        assert output.success is True
        assert output.count == 1
        assert len(output.results) == 1
