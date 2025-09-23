import os
from pathlib import Path
from typing import Dict, List, Optional

import kuzu

from app.be_core.config import settings
from app.be_core.logger import logger


class KuzuManager:
    _instance: Optional["KuzuManager"] = None
    _dbs: Dict[str, kuzu.Database] = {}
    _connections: Dict[str, kuzu.Connection] = {}
    _read_only: Dict[str, bool] = {}
    DB_DIR = Path(settings.KUZU_DB_DIR)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KuzuManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.initialized = True
            if not self.DB_DIR.exists():
                logger.error(
                    f"Kuzu DB directory '{self.DB_DIR}' does not exist. The Docker volume may not be mounted correctly."
                )
                raise RuntimeError(
                    f"Kuzu DB directory '{self.DB_DIR}' does not exist. The Docker volume may not be mounted correctly."
                )

    def _initialize_db(self, graph_id: str, read_only: bool = False):
        """Initialize a KuzuDB database for a specific graph"""
        try:
            # Create database file path
            db_path = str(self.DB_DIR / f"{graph_id}.db")

            # Create database and connection
            self._dbs[graph_id] = kuzu.Database(db_path, read_only=read_only)
            self._connections[graph_id] = kuzu.Connection(self._dbs[graph_id])
            self._read_only[graph_id] = read_only
            # If read only, don't make entity tables, etc, since they've been created before and will fail.
            # TODO: make this more robust.
            if read_only:
                return
            # Create Entity table with embedding support
            try:
                self._connections[graph_id].execute(
                    """
                    CREATE NODE TABLE Entity (
                        name STRING,
                        type STRING,
                        description STRING,
                        name_embedding DOUBLE[1536],
                        description_embedding DOUBLE[1536],
                        PRIMARY KEY (name)
                    )
                """
                )
                logger.info(
                    f"Created Entity table with embeddings for graph {graph_id}"
                )
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.error(f"Failed to create Entity table: {str(e)}")
                    raise
                logger.info(f"Entity table already exists for graph {graph_id}")

            # Create Relationship table
            try:
                self._connections[graph_id].execute(
                    """
                    CREATE REL TABLE Relationship (
                        FROM Entity TO Entity,
                        description STRING,
                        type STRING
                    )
                """
                )
                logger.info(f"Created Relationship table for graph {graph_id}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.error(f"Failed to create Relationship table: {str(e)}")
                    raise
                logger.info(f"Relationship table already exists for graph {graph_id}")

            logger.info(f"Successfully initialized KuzuDB for graph {graph_id}")
        except Exception as e:
            logger.error(f"Failed to initialize KuzuDB for graph {graph_id}: {str(e)}")
            raise

    def get_connection(self, graph_id: str, read_only: bool = False) -> kuzu.Connection:
        """Get the connection instance for a specific graph"""
        if graph_id not in self._connections:
            self._initialize_db(graph_id, read_only=read_only)
        elif read_only and not self._read_only.get(graph_id, False):
            # If we need read-only but current connection is read-write, close and reopen
            self.close_connection(graph_id)
            self._initialize_db(graph_id, read_only=True)
        return self._connections[graph_id]

    def save_entity(self, graph_id: str, name: str, type: str, description: str):
        """Save an entity to the database"""
        try:
            self.get_connection(graph_id).execute(
                """
                CREATE (e:Entity {
                    name: $name,
                    type: $type,
                    description: $description
                })
            """,
                {
                    "name": name,
                    "type": type,
                    "description": description,
                },
            )
        except Exception as e:
            logger.error(f"Failed to save entity: {str(e)}")
            raise

    def save_entity_with_embeddings(
        self,
        graph_id: str,
        name: str,
        type: str,
        description: str,
        name_embedding: List[float],
        description_embedding: List[float],
    ):
        """Save an entity with embeddings to the database"""
        try:
            self.get_connection(graph_id).execute(
                """
                CREATE (e:Entity {
                    name: $name,
                    type: $type,
                    description: $description,
                    name_embedding: $name_embedding,
                    description_embedding: $description_embedding
                })
            """,
                {
                    "name": name,
                    "type": type,
                    "description": description,
                    "name_embedding": name_embedding,
                    "description_embedding": description_embedding,
                },
            )
        except Exception as e:
            logger.error(f"Failed to save entity with embeddings: {str(e)}")
            raise

    def vector_search_entities(
        self,
        graph_id: str,
        query_embedding: List[float],
        limit: int = 20,
        similarity_threshold: float = 0.7,
    ) -> List[dict]:
        """
        Perform vector similarity search on entities using both name and description embeddings
        """
        try:
            # Note: KuzuDB doesn't have built-in vector similarity functions like PostgreSQL
            # This is a simplified approach - in production, consider using dedicated vector DB
            result = self.get_connection(graph_id).execute(
                """
                MATCH (e:Entity)
                WHERE e.name_embedding IS NOT NULL OR e.description_embedding IS NOT NULL
                RETURN e.name, e.type, e.description, e.name_embedding, e.description_embedding
                LIMIT $limit
                """,
                {"limit": limit * 5},  # Get more candidates for filtering
            )

            entities = []
            while result.has_next():
                row = result.get_next()
                entities.append(
                    {
                        "name": row[0],
                        "type": row[1],
                        "description": row[2],
                        "name_embedding": row[3],
                        "description_embedding": row[4],
                    }
                )

            # Calculate cosine similarity in Python (since KuzuDB lacks vector functions)
            from app.utils.vector_utils import cosine_similarity

            scored_entities = []
            for entity in entities:
                name_sim = 0.0
                desc_sim = 0.0

                if entity["name_embedding"]:
                    name_sim = cosine_similarity(
                        query_embedding, entity["name_embedding"]
                    )

                if entity["description_embedding"]:
                    desc_sim = cosine_similarity(
                        query_embedding, entity["description_embedding"]
                    )

                # Use the higher similarity score
                max_similarity = max(name_sim, desc_sim)

                if max_similarity >= similarity_threshold:
                    scored_entities.append(
                        {
                            "name": entity["name"],
                            "type": entity["type"],
                            "description": entity["description"],
                            "similarity_score": max_similarity,
                            "match_type": (
                                "name" if name_sim > desc_sim else "description"
                            ),
                        }
                    )

            # Sort by similarity score and limit results
            scored_entities.sort(key=lambda x: x["similarity_score"], reverse=True)
            return scored_entities[:limit]

        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return []

    def save_relationship(
        self,
        graph_id: str,
        source_name: str,
        source_type: str,
        target_name: str,
        target_type: str,
        description: str,
        type: str,
    ):
        """Save a relationship to the database"""
        try:
            self.get_connection(graph_id).execute(
                """
                MATCH (source:Entity {name: $source_name, type: $source_type})
                MATCH (target:Entity {name: $target_name, type: $target_type})
                CREATE (source)-[r:Relationship {
                    description: $description,
                    type: $type
                }]->(target)
            """,
                {
                    "source_name": source_name,
                    "source_type": source_type,
                    "target_name": target_name,
                    "target_type": target_type,
                    "description": description,
                    "type": type,
                },
            )
        except Exception as e:
            logger.error(f"Failed to save relationship: {str(e)}")
            raise

    def get_entity(self, graph_id: str, entity_id: str) -> Optional[dict]:
        """Get an entity by ID"""
        try:
            result = self.get_connection(graph_id).execute(
                """
                MATCH (e:Entity {id: $id})
                RETURN e
            """,
                {"id": entity_id},
            )
            if result.has_next():
                return result.get_next()
            return None
        except Exception as e:
            logger.error(f"Failed to get entity: {str(e)}")
            raise

    def get_relationships(self, graph_id: str, entity_id: str) -> list:
        """Get all relationships for an entity"""
        try:
            result = self.get_connection(graph_id).execute(
                """
                MATCH (e:Entity {id: $id})-[r:Relationship]->(target:Entity)
                RETURN r, target
            """,
                {"id": entity_id},
            )
            relationships = []
            while result.has_next():
                relationships.append(result.get_next())
            return relationships
        except Exception as e:
            logger.error(f"Failed to get relationships: {str(e)}")
            raise

    def clear_database(self, graph_id: str):
        """Clear all data from the database"""
        try:
            self.get_connection(graph_id).execute("MATCH (n) DETACH DELETE n")
            logger.info(f"Successfully cleared database for graph {graph_id}")
        except Exception as e:
            logger.error(f"Failed to clear database: {str(e)}")
            raise

    def delete_graph(self, graph_id: str):
        """Delete a graph database"""
        try:
            if graph_id in self._connections:
                del self._connections[graph_id]
            if graph_id in self._dbs:
                del self._dbs[graph_id]
            db_path = self.DB_DIR / f"{graph_id}.db"
            if db_path.exists():
                os.remove(db_path)
            logger.info(f"Successfully deleted graph {graph_id}")
        except Exception as e:
            logger.error(f"Failed to delete graph: {str(e)}")
            raise

    def close_connection(self, graph_id: str):
        """Close the connection for a specific graph"""
        try:
            if graph_id in self._connections:
                self._connections[graph_id].close()
                del self._connections[graph_id]
                logger.info(f"Closed connection for graph {graph_id}")
        except Exception as e:
            logger.error(f"Failed to close connection for graph {graph_id}: {str(e)}")
            raise

    def close_database(self, graph_id: str):
        """Close both connection and database for a specific graph"""
        try:
            # Close connection first
            self.close_connection(graph_id)

            # Close database
            if graph_id in self._dbs:
                self._dbs[graph_id].close()
                del self._dbs[graph_id]
                if graph_id in self._read_only:
                    del self._read_only[graph_id]
                logger.info(f"Closed database for graph {graph_id}")
        except Exception as e:
            logger.error(f"Failed to close database for graph {graph_id}: {str(e)}")
            raise

    def close_all(self):
        """Close all connections and databases"""
        try:
            graph_ids = list(self._connections.keys())
            for graph_id in graph_ids:
                self.close_database(graph_id)
            logger.info("Closed all connections and databases")
        except Exception as e:
            logger.error(f"Failed to close all connections: {str(e)}")
            raise

    def is_read_only(self, graph_id: str) -> bool:
        """Check if a graph is opened in read-only mode"""
        return self._read_only.get(graph_id, False)
