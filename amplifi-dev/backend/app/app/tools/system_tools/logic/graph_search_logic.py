import re
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.api.deps import get_async_gpt4o_client
from app.be_core.logger import logger
from app.crud.graph_crud import crud_graph
from app.tools.system_tools.schemas.graph_search_schema import (
    GraphSearchInput,
    GraphSearchOutput,
    GraphSearchResult,
)
from app.utils.graph.extract import GraphReader
from app.utils.openai_utils import generate_embedding_async

client = get_async_gpt4o_client()


async def perform_graph_search(input_data: GraphSearchInput) -> GraphSearchOutput:
    """Main function to perform graph search - entry point for the tool"""
    logger.info(f"Performing graph search with input: {input_data}")

    # Note: Since we can't easily inject db session in this context,
    # we'll create a direct connection. In production, consider using dependency injection.
    from app.db.session import SessionLocal

    try:
        async with SessionLocal() as db:
            service = GraphSearchService()
            result = await service.search(
                db=db,
                query=input_data.query,
                dataset_ids=input_data.dataset_ids,
                limit=input_data.limit,
            )

            return GraphSearchOutput(
                query=result["query"],
                dataset_ids=result["dataset_ids"],
                cypher_query=result.get("cypher_query"),
                method=result["method"],
                results=[
                    GraphSearchResult(
                        content=_extract_content(r),
                        result_type=r.get("result_type", "unknown"),
                        score=r.get("score", 0.0),
                        metadata=r,
                    )
                    for r in result["results"]
                ],
                count=result["count"],
                success=result["success"],
                error=result.get("error"),
            )
    except Exception as e:
        logger.error(f"Graph search failed: {str(e)}")
        return GraphSearchOutput(
            query=input_data.query,
            dataset_ids=input_data.dataset_ids,
            cypher_query=None,
            method="error",
            results=[],
            count=0,
            success=False,
            error=str(e),
        )


def _extract_content(result_dict: Dict[str, Any]) -> str:
    """Extract meaningful content from result dictionary for display"""
    if result_dict.get("result_type") == "entity":
        name = result_dict.get("name", "")
        entity_type = result_dict.get("type", "")
        description = result_dict.get("description", "")
        return f"{name} ({entity_type}): {description}"
    elif result_dict.get("result_type") == "relationship":
        source = result_dict.get("source_name", "")
        target = result_dict.get("target_name", "")
        rel_type = result_dict.get("relationship_type", "")
        return f"{source} --[{rel_type}]--> {target}"
    elif result_dict.get("result_type") == "count":
        # Handle count results
        if "totalEntities" in result_dict and "totalRelationships" in result_dict:
            entities = result_dict.get("totalEntities", 0)
            relationships = result_dict.get("totalRelationships", 0)
            return f"Entities: {entities}, Relationships: {relationships}"
        elif "totalEntities" in result_dict:
            return f"Total Entities: {result_dict.get('totalEntities', 0)}"
        elif "totalRelationships" in result_dict:
            return f"Total Relationships: {result_dict.get('totalRelationships', 0)}"
        else:
            return str(result_dict.get("count", "Unknown count"))
    else:
        # Fallback: try to extract meaningful content
        return str(
            result_dict.get("name", result_dict.get("content", str(result_dict)))
        )


class GraphSearchService:
    """Graph search service for converting natural language to Cypher queries"""

    def __init__(self):
        self.llm_client = client

    async def search(
        self, db: Session, query: str, dataset_ids: List[UUID], limit: int = 20
    ) -> Dict[str, Any]:
        """Main search method that processes multiple datasets"""
        try:
            all_results = []
            successful_searches = 0
            cypher_queries = []
            methods_used = []

            # Use higher limit for listing queries
            effective_limit = 1000 if self._is_listing_query(query) else limit

            for dataset_id in dataset_ids:
                dataset_result = await self._search_single_dataset(
                    db, query, dataset_id, effective_limit
                )

                if dataset_result["success"]:
                    successful_searches += 1
                    all_results.extend(dataset_result["results"])
                    if dataset_result.get("cypher_query"):
                        cypher_queries.append(dataset_result["cypher_query"])
                    methods_used.append(dataset_result["method"])
                else:
                    logger.warning(
                        f"Search failed for dataset {dataset_id}: {dataset_result.get('error')}"
                    )

            if successful_searches == 0:
                return {
                    "success": False,
                    "error": "No graphs available for the provided datasets",
                    "results": [],
                    "count": 0,
                    "query": query,
                    "dataset_ids": dataset_ids,
                    "cypher_query": None,
                    "method": "none",
                }

            # Sort results by score and limit
            all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

            # Check if this is a listing query that should return all results
            if self._is_listing_query(query):
                limited_results = all_results  # Return all results for listing queries
            else:
                limited_results = all_results[:limit]  # Apply limit for search queries

            # Determine primary method and cypher query
            primary_method = (
                "llm_generated"
                if "llm_generated" in methods_used
                else methods_used[0] if methods_used else "fallback"
            )
            primary_cypher = cypher_queries[0] if cypher_queries else None

            return {
                "success": True,
                "query": query,
                "dataset_ids": dataset_ids,
                "cypher_query": primary_cypher,
                "method": primary_method,
                "results": limited_results,
                "count": len(limited_results),
            }

        except Exception as e:
            logger.error(f"Graph search failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "count": 0,
                "query": query,
                "dataset_ids": dataset_ids,
                "cypher_query": None,
                "method": "error",
            }

    async def _search_single_dataset(
        self, db: Session, query: str, dataset_id: UUID, limit: int = 20
    ) -> Dict[str, Any]:
        """Search method for a single dataset"""
        try:
            # 1. Get graph ID for dataset
            graph_id = await self._get_graph_id(db, dataset_id)
            if not graph_id:
                return {
                    "success": False,
                    "error": f"No graph available for dataset {dataset_id}",
                    "results": [],
                    "count": 0,
                }

            # 2. Try vector search first (most semantic)
            # For listing queries, use higher limit for vector search
            vector_limit = limit if self._is_listing_query(query) else limit // 3
            vector_results = await self._vector_search_entities(
                graph_id, query, vector_limit
            )

            # 3. Convert natural language to Cypher using LLM
            cypher_result = await self._generate_cypher(query, graph_id)
            cypher_results = []

            if cypher_result["success"]:
                # 4. Execute Cypher query using GraphReader
                # For listing queries, use higher limit or no limit
                cypher_limit = limit if self._is_listing_query(query) else limit // 3
                cypher_results = await self._execute_cypher(
                    graph_id, cypher_result["cypher_query"], cypher_limit
                )

            # 5. Fallback to simple search if needed
            fallback_results = []
            if len(vector_results) + len(cypher_results) < limit:
                remaining_limit = limit - len(vector_results) - len(cypher_results)
                # For listing queries, use higher limit for fallback search
                fallback_limit = (
                    remaining_limit if not self._is_listing_query(query) else limit
                )
                fallback_results = await self._fallback_search(
                    graph_id, query, fallback_limit
                )

            # 6. Combine and deduplicate results
            all_results = vector_results + cypher_results + fallback_results
            deduplicated_results = self._deduplicate_results(all_results)

            # 7. Format results
            formatted_results = self._format_results(deduplicated_results, query)

            # Determine method used
            method = "hybrid"
            if vector_results and not cypher_results and not fallback_results:
                method = "vector_search"
            elif cypher_results and not vector_results and not fallback_results:
                method = "llm_generated"
            elif fallback_results and not vector_results and not cypher_results:
                method = "fallback"

            # For listing queries, return all results; for search queries, apply limit
            if self._is_listing_query(query):
                final_results = formatted_results
            else:
                final_results = formatted_results[:limit]

            return {
                "success": True,
                "query": query,
                "dataset_id": dataset_id,
                "cypher_query": cypher_result.get("cypher_query"),
                "method": method,
                "results": final_results,
                "count": len(final_results),
            }

        except Exception as e:
            logger.error(f"Graph search failed for dataset {dataset_id}: {str(e)}")
            return {"success": False, "error": str(e), "results": [], "count": 0}

    async def _get_graph_id(self, db: Session, dataset_id: UUID) -> Optional[str]:
        """Get most recent successful graph for dataset"""
        try:
            graphs = await crud_graph.get_graphs_by_dataset(
                db_session=db, dataset_id=dataset_id
            )

            # Filter successful graphs and sort by creation date (most recent first)
            # A graph is successful if both entities and relationships are successfully extracted
            successful_graphs = [
                g
                for g in graphs
                if g.entities_status == "success"
                and g.relationships_status == "success"
            ]

            if not successful_graphs:
                logger.warning(f"No successful graphs found for dataset {dataset_id}")
                return None

            # Sort by id (UUID7 contains timestamp) to get most recent - fallback approach
            # Since GraphRead doesn't have created_at, we use id sorting as UUID7 embeds timestamp
            most_recent_graph = sorted(
                successful_graphs, key=lambda g: g.id, reverse=True
            )[0]

            logger.debug(
                f"Selected most recent successful graph {most_recent_graph.id} "
                f"for dataset {dataset_id}"
            )

            return str(most_recent_graph.id)
        except Exception as e:
            logger.error(f"Error getting graph ID for dataset {dataset_id}: {str(e)}")
            return None

    async def _generate_cypher(self, query: str, graph_id: str) -> Dict[str, Any]:
        """Generate Cypher query using LLM"""
        try:
            # Get graph schema for context
            schema = await self._get_simple_schema(graph_id)

            # Get sample relationships for better context
            relationship_context = await self._get_relationship_examples(graph_id)

            # Create prompt
            prompt = f"""
            Convert this natural language query to Cypher for KuzuDB:
            Query: {query}

            Schema Context:
            - Available Entity Types: {', '.join(schema.get('entity_types', []))}
            - Sample Entity Names: {', '.join(schema.get('sample_entities', [])[:10])}
            - Sample Relationship Types: {', '.join(relationship_context.get('relationship_types', []))}

            Cypher Query Rules:
            - Use 'Entity' as the main node label: MATCH (e:Entity)
            - Use 'Relationship' as the edge label: -[r:Relationship]-
            - Entity properties: name, type, description
            - Relationship properties: type (stored as r.type), description
            - For COUNT queries: DO NOT add LIMIT (e.g., "count entities" → MATCH (e:Entity) RETURN count(e))
            - For LIST ALL queries: DO NOT add LIMIT (e.g., "list all entities" → MATCH (e:Entity) RETURN e.name, e.type, e.description)
            - For SEARCH queries: Add LIMIT 20 at the end
            - Use case-insensitive search: toLower(e.name) CONTAINS toLower('search_term')
            - Use the EXACT search terms from the query, do not modify them
            - Do not add words like "Schema" or other modifications to search terms

            Examples:
            - "Count entities" → MATCH (e:Entity) RETURN count(e) as totalEntities
            - "Total number of entities" → MATCH (e:Entity) RETURN count(e) as totalEntities
            - "Total number of entities and relationships" → MATCH (e:Entity) RETURN 'entities' as type, count(e) as count UNION ALL MATCH ()-[r:Relationship]->() RETURN 'relationships' as type, count(r) as count
            - "List all entities" → MATCH (e:Entity) RETURN e.name, e.type, e.description
            - "List all entity types" → MATCH (e:Entity) RETURN DISTINCT e.type ORDER BY e.type
            - "List all relationship types" → MATCH ()-[r:Relationship]-() RETURN DISTINCT r.type ORDER BY r.type
            - "Find John" → MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower('John') RETURN e.name, e.type, e.description LIMIT 20
            - "Show relationships of Apple" → MATCH (e:Entity)-[r:Relationship]-(connected:Entity) WHERE toLower(e.name) CONTAINS toLower('Apple') RETURN e.name, r.type, connected.name, connected.type LIMIT 20

            Return ONLY the Cypher query, no explanation or formatting.
            """

            response = await self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Cypher query generator for KuzuDB. Return only valid Cypher queries without any formatting or explanation.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.1,
            )

            # Clean the response
            cypher_query = response.choices[0].message.content.strip()
            cypher_query = cypher_query.replace("```", "").replace("cypher", "").strip()

            # Add LIMIT only for non-aggregation queries AND non-listing queries
            if (
                not self._is_aggregation_query(cypher_query)
                and not self._is_listing_query(query)
                and "LIMIT" not in cypher_query.upper()
            ):
                cypher_query += " LIMIT 20"

            logger.info(f"Generated Cypher query: {cypher_query}")

            return {"success": True, "cypher_query": cypher_query}

        except Exception as e:
            logger.error(f"Cypher generation failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def _is_aggregation_query(self, cypher_query: str) -> bool:
        """Check if the Cypher query is an aggregation query (COUNT, SUM, etc.) or a listing query that shouldn't have LIMIT"""
        query_upper = cypher_query.upper()
        aggregation_functions = [
            "COUNT(",
            "SUM(",
            "AVG(",
            "MIN(",
            "MAX(",
            "COLLECT(",
            "DISTINCT",
        ]

        # Check for aggregation functions
        for func in aggregation_functions:
            if func in query_upper:
                return True

        # Check for GROUP BY clauses
        if "GROUP BY" in query_upper:
            return True

        # Check for queries that return counts or totals
        count_patterns = [
            "RETURN.*COUNT",
            "RETURN.*TOTAL",
            "RETURN.*NUMBER",
            "RETURN.*AMOUNT",
        ]

        for pattern in count_patterns:
            if re.search(pattern, query_upper):
                return True

        # Check for listing queries that should return all results
        listing_patterns = [
            "RETURN DISTINCT.*TYPE",
            "RETURN DISTINCT.*RELATIONSHIP_TYPE",
            "LIST ALL.*TYPES",
            "SHOW ALL.*TYPES",
            "RETURN.*NAME.*TYPE.*DESCRIPTION",
            "LIST ALL.*ENTITIES",
            "SHOW ALL.*ENTITIES",
            "RETURN.*NAME.*TYPE.*DESCRIPTION$",  # Match end of query for complete entity listing
        ]

        for pattern in listing_patterns:
            if re.search(pattern, query_upper):
                return True

        return False

    def _is_listing_query(self, user_query: str) -> bool:
        """Check if the user query is asking for a complete list of entities/types"""
        query_lower = user_query.lower()

        # First check if this is a filtered query (has "who is", "assigned to", "responsible for", etc.)
        # These should NOT be treated as listing queries
        filtered_patterns = [
            "who is",
            "assigned to",
            "responsible for",
            "who are",
            "assigned for",
            "responsible to",
            "works on",
            "handles",
            "manages",
        ]

        for pattern in filtered_patterns:
            if pattern in query_lower:
                return False  # This is a filtered search, not a general listing

        # Now check for true listing patterns
        listing_patterns = [
            "list all entities",
            "show all entities",
            "get all entities",
            "display all entities",
            "find all entities",
            "give me all entities",
            "all entities",
            "all entity types",
            "all relationship types",
            "all relationships",
            "complete list of entities",
            "entire list of entities",
            "full list of entities",
            "list of all entities",
            "show me all entities",
            "what are all entities",
            "what are the entities",
            "entity types",
            "relationship types",
            "relationships present",
            "all the relationships",
            "all the entities",
            "list all entity types",
            "show all entity types",
            "list all relationship types",
            "show all relationship types",
            "list all the persons",
        ]

        for pattern in listing_patterns:
            if pattern in query_lower:
                return True

        return False

    def _is_labeled_count_query(self, cypher_query: str) -> bool:
        """Check if the Cypher query is a labeled count query with UNION ALL"""
        query_upper = cypher_query.upper()
        return (
            "UNION ALL" in query_upper
            and "COUNT(" in query_upper
            and "'entities'" in cypher_query
            and "'relationships'" in cypher_query
        )

    async def _get_simple_schema(self, graph_id: str) -> Dict[str, Any]:
        """Get basic schema info from the graph"""
        try:
            with GraphReader(graph_id=graph_id) as reader:
                # Get entity types
                entity_types_query = """
                    MATCH (e:Entity)
                    RETURN e.type as type, count(e) as count
                    ORDER BY count DESC
                """
                entity_types_result = reader.execute_query(entity_types_query)

                # Convert entity types result to list
                entity_types = []
                while entity_types_result.has_next():
                    row = entity_types_result.get_next()
                    entity_types.append(row[0])  # Get the type value

                # Get sample entities
                sample_entities_query = """
                    MATCH (e:Entity)
                    RETURN e.name as name
                    ORDER BY e.name LIMIT 15
                """
                sample_entities_result = reader.execute_query(sample_entities_query)

                # Convert sample entities result to list
                sample_entities = []
                while sample_entities_result.has_next():
                    row = sample_entities_result.get_next()
                    sample_entities.append(row[0])  # Get the name value

                return {
                    "entity_types": [et for et in entity_types if et and et.strip()],
                    "sample_entities": [
                        se for se in sample_entities if se and se.strip()
                    ],
                }
        except Exception as e:
            logger.error(f"Error getting schema for graph {graph_id}: {str(e)}")
            return {"entity_types": [], "sample_entities": []}

    async def _execute_cypher(
        self, graph_id: str, cypher_query: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Execute Cypher query using GraphReader"""
        try:
            with GraphReader(graph_id=graph_id) as reader:
                result = reader.execute_query(cypher_query)

                # Convert Kuzu QueryResult to list of dictionaries
                results = []
                count = 0

                # Check if this is a DISTINCT query that should return all results
                is_distinct_query = "DISTINCT" in cypher_query.upper()
                effective_limit = limit if not is_distinct_query else float("inf")

                while result.has_next() and count < effective_limit:
                    row = result.get_next()
                    # Convert row to dictionary based on column names
                    row_dict = {}
                    for i, column_name in enumerate(result.get_column_names()):
                        row_dict[column_name] = row[i]

                    # Special handling for labeled count queries
                    if (
                        self._is_labeled_count_query(cypher_query)
                        and "type" in row_dict
                        and "count" in row_dict
                    ):
                        # Transform labeled count results to match expected format
                        count_type = row_dict["type"]
                        count_value = row_dict["count"]

                        if count_type == "entities":
                            row_dict = {
                                "totalEntities": count_value,
                                "result_type": "count",
                                "count_type": "entities",
                            }
                        elif count_type == "relationships":
                            row_dict = {
                                "totalRelationships": count_value,
                                "result_type": "count",
                                "count_type": "relationships",
                            }

                    results.append(row_dict)
                    count += 1

                return results
        except Exception as e:
            logger.error(f"Cypher execution failed: {str(e)}")
            logger.error(f"Query was: {cypher_query}")
            return []

    async def _fallback_search(
        self, graph_id: str, query: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Enhanced fallback search when Cypher generation fails"""
        try:
            with GraphReader(graph_id=graph_id) as reader:
                results = []

                # Search 1: Entity name and description search
                entity_search_query = f"""
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower('{query}') OR toLower(e.description) CONTAINS toLower('{query}')
                RETURN e.name as name, e.type as type, e.description as description, 'entity' as result_type
                ORDER BY e.name
                LIMIT {limit // 2}
                """
                entity_result = reader.execute_query(entity_search_query)

                # Convert entity results
                while entity_result.has_next():
                    row = entity_result.get_next()
                    row_dict = {}
                    for i, column_name in enumerate(entity_result.get_column_names()):
                        row_dict[column_name] = row[i]
                    results.append(row_dict)

                # Search 2: Relationship search
                relationship_search_query = f"""
                MATCH (source:Entity)-[r:Relationship]->(target:Entity)
                WHERE toLower(source.name) CONTAINS toLower('{query}')
                   OR toLower(target.name) CONTAINS toLower('{query}')
                   OR toLower(r.type) CONTAINS toLower('{query}')
                   OR toLower(r.description) CONTAINS toLower('{query}')
                RETURN source.name as source_name, target.name as target_name,
                       r.type as relationship_type, r.description as relationship_description,
                       'relationship' as result_type
                ORDER BY source.name
                LIMIT {limit // 2}
                """
                rel_result = reader.execute_query(relationship_search_query)

                # Convert relationship results
                while rel_result.has_next():
                    row = rel_result.get_next()
                    row_dict = {}
                    for i, column_name in enumerate(rel_result.get_column_names()):
                        row_dict[column_name] = row[i]
                    results.append(row_dict)

                return results[:limit]
        except Exception as e:
            logger.error(f"Fallback search failed: {str(e)}")
            return []

    async def _get_relationship_examples(self, graph_id: str) -> Dict[str, Any]:
        """Get sample relationship types from the graph"""
        try:
            with GraphReader(graph_id=graph_id) as reader:
                # Get relationship types
                relationship_types_query = """
                    MATCH ()-[r:Relationship]-()
                    RETURN r.type as relationship_type, count(r) as count
                    ORDER BY count DESC LIMIT 10
                """
                rel_types_result = reader.execute_query(relationship_types_query)

                # Convert relationship types result to list
                relationship_types = []
                while rel_types_result.has_next():
                    row = rel_types_result.get_next()
                    if row[0]:  # Only add non-null relationship types
                        relationship_types.append(row[0])

                return {"relationship_types": relationship_types}
        except Exception as e:
            logger.error(
                f"Error getting relationship examples for graph {graph_id}: {str(e)}"
            )
            return {"relationship_types": []}

    async def _vector_search_entities(
        self, graph_id: str, query: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search on entities using query embeddings
        TODO: For production, consider using dedicated vector database (Pinecone/Qdrant)
        instead of computing similarities in Python
        """
        try:
            # Generate embedding for the query
            query_embedding = await generate_embedding_async(query)

            # For better performance, consider pagination and early termination
            batch_size = 1000  # Process entities in batches

            with GraphReader(graph_id=graph_id) as reader:
                # Get count first to avoid loading everything
                count_query = """
                    MATCH (e:Entity)
                    WHERE e.name_embedding IS NOT NULL OR e.description_embedding IS NOT NULL
                    RETURN count(e) as total
                """
                count_result = reader.execute_query(count_query)
                total_entities = (
                    count_result.get_next()[0] if count_result.has_next() else 0
                )

                if total_entities > 10000:
                    logger.warning(
                        f"Large graph with {total_entities} entities. Consider using dedicated vector database."
                    )

                # Retrieve entities with embeddings
                entities_query = """
                    MATCH (e:Entity)
                    WHERE e.name_embedding IS NOT NULL OR e.description_embedding IS NOT NULL
                    RETURN e.name, e.type, e.description, e.name_embedding, e.description_embedding
                    LIMIT $batch_size
                """
                entities_result = reader.execute_query(
                    entities_query, {"batch_size": min(batch_size, total_entities)}
                )

                # Convert to list for processing
                entities = []
                while entities_result.has_next():
                    row = entities_result.get_next()
                    entities.append(
                        {
                            "name": row[0],
                            "type": row[1],
                            "description": row[2],
                            "name_embedding": row[3],
                            "description_embedding": row[4],
                        }
                    )

                # Calculate similarities with early termination for performance
                from app.utils.vector_utils import (
                    is_valid_vector,
                    top_k_similar,
                )

                similarity_threshold = 0.5  # Skip very low similarity scores

                # Prepare vectors for batch processing
                entity_vectors = []
                entity_metadata = []

                for entity in entities:
                    # Validate and collect embeddings
                    name_emb = (
                        entity["name_embedding"]
                        if is_valid_vector(entity["name_embedding"], 1536)
                        else None
                    )
                    desc_emb = (
                        entity["description_embedding"]
                        if is_valid_vector(entity["description_embedding"], 1536)
                        else None
                    )

                    if name_emb:
                        entity_vectors.append(name_emb)
                        entity_metadata.append(
                            {"entity": entity, "embedding_type": "name"}
                        )

                    if desc_emb:
                        entity_vectors.append(desc_emb)
                        entity_metadata.append(
                            {"entity": entity, "embedding_type": "description"}
                        )

                # Use batch similarity calculation for better performance
                if entity_vectors:
                    top_results = top_k_similar(
                        query_embedding,
                        entity_vectors,
                        k=limit * 2,  # Get more candidates for deduplication
                        threshold=similarity_threshold,
                    )

                    # Convert results back to entity format with deduplication
                    entity_scores = {}  # Track best score per entity

                    for idx, similarity_score in top_results:
                        metadata = entity_metadata[idx]
                        entity = metadata["entity"]
                        entity_name = entity["name"]

                        # Keep only the highest score per entity
                        if (
                            entity_name not in entity_scores
                            or similarity_score > entity_scores[entity_name]["score"]
                        ):
                            entity_scores[entity_name] = {
                                "name": entity["name"],
                                "type": entity["type"],
                                "description": entity["description"],
                                "score": similarity_score,
                                "match_type": metadata["embedding_type"],
                                "result_type": "entity",
                            }

                    # Convert to list and sort
                    scored_entities = list(entity_scores.values())
                    scored_entities.sort(key=lambda x: x["score"], reverse=True)
                else:
                    scored_entities = []

                logger.info(
                    f"Vector search found {len(scored_entities)} relevant entities out of {total_entities}"
                )
                return scored_entities[:limit]

        except Exception as e:
            logger.error(f"Vector search failed for graph {graph_id}: {str(e)}")
            return []

    def _deduplicate_results(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate results based on entity names or relationship patterns
        """
        seen = set()
        deduplicated = []

        for result in results:
            # Create a unique key based on the result content
            if result.get("result_type") == "entity":
                key = f"entity:{result.get('name', '')}"
            elif result.get("result_type") == "relationship":
                source = result.get("source_name", result.get("source", ""))
                target = result.get("target_name", result.get("target", ""))
                rel_type = result.get("relationship_type", "")
                key = f"relationship:{source}:{target}:{rel_type}"
            else:
                # For general results, use a hash of the content
                content = str(result.get("content", result))
                key = f"general:{hash(content)}"

            if key not in seen:
                seen.add(key)
                deduplicated.append(result)

        return deduplicated

    def _format_results(
        self, results: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """
        Format and enrich results for consistent output structure

        Args:
            results: Raw results from different search methods
            query: Original user query for context

        Returns:
            List of formatted result dictionaries
        """
        formatted = []

        for result in results:
            # Ensure all results have consistent fields
            formatted_result = {
                "name": result.get("name", ""),
                "type": result.get("type", ""),
                "description": result.get("description", ""),
                "score": result.get("score", 0.0),
                "result_type": result.get("result_type", "unknown"),
                "match_type": result.get("match_type", "unknown"),
            }

            # Add relationship-specific fields if present
            if result.get("result_type") == "relationship":
                formatted_result.update(
                    {
                        "source_name": result.get("source_name", ""),
                        "target_name": result.get("target_name", ""),
                        "relationship_type": result.get("relationship_type", ""),
                        "relationship_description": result.get(
                            "relationship_description", ""
                        ),
                    }
                )

            # Add count-specific fields if present
            if result.get("result_type") == "count":
                if "totalEntities" in result:
                    formatted_result["totalEntities"] = result["totalEntities"]
                if "totalRelationships" in result:
                    formatted_result["totalRelationships"] = result[
                        "totalRelationships"
                    ]
                if "count_type" in result:
                    formatted_result["count_type"] = result["count_type"]

            # Add any additional fields from the original result
            for key, value in result.items():
                if key not in formatted_result:
                    formatted_result[key] = value

            formatted.append(formatted_result)

        return formatted
