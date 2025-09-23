from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.tools.system_tools.logic.graph_search_logic import (
    GraphSearchService,
    _extract_content,
)
from app.tools.system_tools.schemas.graph_search_schema import (
    GraphSearchInput,
    GraphSearchOutput,
    GraphSearchResult,
)


class TestGraphSearchUnit:
    """Basic unit tests for graph search functionality"""

    def test_extract_content_various_formats(self):
        """Test _extract_content function with various result formats"""
        # Test entity result
        entity_result = {
            "result_type": "entity",
            "name": "Socrates",
            "type": "Philosopher",
            "description": "Ancient Greek philosopher",
        }
        content = _extract_content(entity_result)
        assert content == "Socrates (Philosopher): Ancient Greek philosopher"

        # Test relationship result
        rel_result = {
            "result_type": "relationship",
            "source_name": "Socrates",
            "target_name": "Plato",
            "relationship_type": "taught",
        }
        content = _extract_content(rel_result)
        assert content == "Socrates --[taught]--> Plato"

        # Test general result
        general_result = {"name": "Aristotle"}
        content = _extract_content(general_result)
        assert content == "Aristotle"

    @pytest.mark.asyncio
    async def test_graph_search_service_initialization(self):
        """Test GraphSearchService initialization"""
        with patch(
            "app.tools.system_tools.logic.graph_search_logic.client"
        ) as mock_client:
            service = GraphSearchService()
            assert service.llm_client == mock_client

    @pytest.mark.asyncio
    async def test_search_single_dataset_success(self):
        """Test successful single dataset search"""
        service = GraphSearchService()
        mock_db = MagicMock()
        dataset_id = uuid4()

        with (
            patch.object(service, "_get_graph_id") as mock_get_graph,
            patch.object(service, "_vector_search_entities") as mock_vector,
            patch.object(service, "_generate_cypher") as mock_cypher,
            patch.object(service, "_execute_cypher") as mock_execute,
            patch.object(service, "_fallback_search") as mock_fallback,
            patch.object(service, "_deduplicate_results") as mock_dedup,
            patch.object(service, "_format_results") as mock_format,
        ):

            mock_get_graph.return_value = "graph-123"
            mock_vector.return_value = [
                {"name": "Socrates", "score": 0.9, "result_type": "entity"}
            ]
            mock_cypher.return_value = {
                "success": True,
                "cypher_query": "MATCH (e:Entity) WHERE e.name CONTAINS 'Socrates' RETURN e",
            }
            mock_execute.return_value = []
            mock_fallback.return_value = []
            mock_dedup.return_value = [
                {"name": "Socrates", "score": 0.9, "result_type": "entity"}
            ]
            mock_format.return_value = [
                {"name": "Socrates", "score": 0.9, "result_type": "entity"}
            ]

            result = await service._search_single_dataset(
                mock_db, "Find Socrates", dataset_id, 10
            )

            assert result["success"] is True
            assert result["query"] == "Find Socrates"
            assert result["dataset_id"] == dataset_id
            assert (
                result["method"] == "vector_search"
            )  # Fixed: should be vector_search since only vector results returned
            assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_search_single_dataset_no_graph(self):
        """Test single dataset search when no graph is available"""
        service = GraphSearchService()
        mock_db = MagicMock()
        dataset_id = uuid4()

        with patch.object(service, "_get_graph_id") as mock_get_graph:
            mock_get_graph.return_value = None

            result = await service._search_single_dataset(
                mock_db, "Find Socrates", dataset_id, 10
            )

            assert result["success"] is False
            assert "No graph available" in result["error"]
            assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_generate_cypher_success(self):
        """Test successful Cypher query generation"""
        service = GraphSearchService()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "MATCH (e:Entity) WHERE e.name CONTAINS 'Socrates' RETURN e LIMIT 20"
        )

        service.llm_client = MagicMock()
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
            assert "LIMIT 20" in result["cypher_query"]

    def test_deduplicate_results(self):
        """Test result deduplication"""
        service = GraphSearchService()

        results = [
            {"name": "Socrates", "result_type": "entity", "score": 0.9},
            {"name": "Socrates", "result_type": "entity", "score": 0.8},  # Duplicate
            {"name": "Plato", "result_type": "entity", "score": 0.7},
        ]

        deduplicated = service._deduplicate_results(results)

        assert len(deduplicated) == 2
        names = [r["name"] for r in deduplicated]
        assert "Socrates" in names
        assert "Plato" in names
        # Should keep the higher score for Socrates
        socrates_result = next(r for r in deduplicated if r["name"] == "Socrates")
        assert socrates_result["score"] == 0.9

    def test_format_results(self):
        """Test result formatting"""
        service = GraphSearchService()

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
        assert formatted[0]["type"] == "Philosopher"
        assert formatted[0]["description"] == "Ancient Greek philosopher"
        assert formatted[0]["score"] == 0.9
        assert formatted[0]["result_type"] == "entity"


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

        # Test default limit
        input_with_default = GraphSearchInput(
            query="Find Socrates",
            dataset_ids=[uuid4()],
        )
        assert input_with_default.limit == 20  # Default value

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
        assert (
            output.results[0].content
            == "Socrates (Philosopher): Ancient Greek philosopher"
        )
        assert output.results[0].score == 0.9

    def test_graph_search_result_creation(self):
        """Test GraphSearchResult creation"""
        result = GraphSearchResult(
            content="Socrates (Philosopher): Ancient Greek philosopher",
            result_type="entity",
            score=0.9,
            metadata={"name": "Socrates", "type": "Philosopher"},
        )

        assert result.content == "Socrates (Philosopher): Ancient Greek philosopher"
        assert result.result_type == "entity"
        assert result.score == 0.9
        assert result.metadata["name"] == "Socrates"
        assert result.metadata["type"] == "Philosopher"
