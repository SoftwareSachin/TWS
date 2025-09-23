from __future__ import annotations

import json
from typing import Optional
from uuid import uuid4

import kuzu
import openai
from openai import AsyncAzureOpenAI

from app.api.deps import (
    get_async_azure_client,
    get_async_gpt4o_client,
    get_gpt5_client,
    get_gpt5_client_async,
    get_gpt41_client,
    get_gpt41_client_async,
)
from app.be_core.logger import logger
from app.schemas.graph_schema import (
    EntityRelationship,
    ExtractedEntity,
    GraphEntitiesRelationshipsResponse,
)
from app.schemas.rag_generation_schema import ChatModelEnum

from .kuzu_manager import KuzuManager

MERGE_DESCRIPTIONS_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "merge_descriptions",
        "description": "Merge multiple descriptions into one comprehensive description.",
        "schema": {
            "type": "object",
            "properties": {
                "merged_description": {
                    "type": "string",
                    "description": "A comprehensive description that combines all the information from the input descriptions without redundancy.",
                }
            },
            "required": ["merged_description"],
            "additionalProperties": False,
        },
    },
}


def get_entity_response_format(entity_types: Optional[list[str]]) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "entity_extraction",
            "description": "Extracted entities from the text with type and description.",
            "schema": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entity_name": {
                                    "type": "string",
                                    "description": "Name of the entity, capitalized",
                                },
                                "entity_type": (
                                    {"type": "string", "enum": entity_types}
                                    if entity_types
                                    else {"type": "string"}
                                ),
                                "entity_description": {
                                    "type": "string",
                                    "description": "Comprehensive description of the entity's attributes and activities",
                                },
                            },
                            "required": [
                                "entity_name",
                                "entity_type",
                                "entity_description",
                            ],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["entities"],
                "additionalProperties": False,
            },
        },
    }


async def get_entities(
    client: AsyncAzureOpenAI,
    txt: str,
    entity_types: Optional[list[str]] = None,
    validate_entity_types: bool = True,
) -> list[ExtractedEntity]:
    response = await client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": txt,
            }
        ],
        response_format=get_entity_response_format(entity_types),
    )
    entities_data = json.loads(response.choices[0].message.content)["entities"]
    entities = []
    for entity in entities_data:
        extracted_entity = ExtractedEntity(
            name=entity["entity_name"],
            type=entity["entity_type"],
            description=entity["entity_description"],
        )
        if (
            validate_entity_types
            and entity_types
            and extracted_entity.type not in entity_types
        ):
            logger.warn(
                f"invalid entity type: {extracted_entity.type} for entity: {extracted_entity.name}"
            )
            continue
        entities.append(extracted_entity)
    return entities


async def deduplicate_entities(
    entities: list[ExtractedEntity],
    client: AsyncAzureOpenAI,
) -> list[ExtractedEntity]:
    """
    Deduplicate entities with the same name and type (case insensitive).
    Uses LLM to merge descriptions of duplicate entities.
    For entities with the same name but different types, renames them to include their type.

    Returns:
        A list of deduplicated entities
    """
    # Group entities by lowercase name
    entity_groups = {}
    for entity in entities:
        key = entity.name.lower()
        if key not in entity_groups:
            entity_groups[key] = []
        entity_groups[key].append(entity)

    # Process each group
    deduplicated_entities = []
    for name_lower, group in entity_groups.items():
        if len(group) == 1:
            # No duplicates, keep as is
            deduplicated_entities.append(group[0])
            continue

        # Group by type
        type_groups = {}
        for entity in group:
            type_key = entity.type.lower()
            if type_key not in type_groups:
                type_groups[type_key] = []
            type_groups[type_key].append(entity)

        # Only add type to names if there are multiple types
        should_add_type = len(type_groups) > 1

        # Process each type group
        for type_lower, type_group in type_groups.items():
            if len(type_group) == 1:
                # Only one entity of this type
                entity = type_group[0]
                if should_add_type:
                    entity.name = f"{entity.name} ({entity.type})"
                deduplicated_entities.append(entity)
                continue

            # Multiple entities of same type, merge descriptions
            descriptions = [entity.description for entity in type_group]
            prompt = f"""Please merge these descriptions about {name_lower} (a {type_lower}) into one comprehensive description:

Descriptions to merge:
{chr(10).join(f'- {desc}' for desc in descriptions)}

Merge them into one comprehensive description that captures all unique information without redundancy."""

            response = await client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                response_format=MERGE_DESCRIPTIONS_FORMAT,
            )

            merged_description = json.loads(response.choices[0].message.content)[
                "merged_description"
            ]

            # Create new entity with merged description and renamed if needed
            new_name = (
                f"{type_group[0].name} ({type_group[0].type})"
                if should_add_type
                else type_group[0].name
            )
            deduplicated_entities.append(
                ExtractedEntity(
                    name=new_name,
                    type=type_group[0].type,
                    description=merged_description,
                )
            )
    return deduplicated_entities


def get_entities_sync(
    client: openai.AzureOpenAI,
    txt: str,
    entity_types: Optional[list[str]] = None,
    validate_entity_types: bool = True,
) -> list[ExtractedEntity]:
    """Sync version of get_entities"""
    if not entity_types:
        logger.info("No entity types provided, skipping entity type validation")
        validate_entity_types = False
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": txt,
            }
        ],
        response_format=get_entity_response_format(entity_types),
    )
    entities_data = json.loads(response.choices[0].message.content)["entities"]
    entities = []
    for entity in entities_data:
        extracted_entity = ExtractedEntity(
            name=entity["entity_name"],
            type=entity["entity_type"],
            description=entity["entity_description"],
        )
        if validate_entity_types and extracted_entity.type not in entity_types:
            logger.warn(
                f"invalid entity type: {extracted_entity.type} for entity: {extracted_entity.name}"
            )
            continue
        entities.append(extracted_entity)
    return entities


def deduplicate_entities_sync(
    entities: list[ExtractedEntity],
    client: openai.AzureOpenAI,
) -> list[ExtractedEntity]:
    """Sync version of deduplicate_entities"""
    # Group entities by lowercase name
    entity_groups = {}
    for entity in entities:
        key = entity.name.lower()
        if key not in entity_groups:
            entity_groups[key] = []
        entity_groups[key].append(entity)

    # Process each group
    deduplicated_entities = []
    for name_lower, group in entity_groups.items():
        if len(group) == 1:
            # No duplicates, keep as is
            deduplicated_entities.append(group[0])
            continue

        # Group by type
        type_groups = {}
        for entity in group:
            type_key = entity.type.lower()
            if type_key not in type_groups:
                type_groups[type_key] = []
            type_groups[type_key].append(entity)

        # Only add type to names if there are multiple types
        should_add_type = len(type_groups) > 1

        # Process each type group
        for type_lower, type_group in type_groups.items():
            if len(type_group) == 1:
                # Only one entity of this type
                entity = type_group[0]
                if should_add_type:
                    entity.name = f"{entity.name} ({entity.type})"
                deduplicated_entities.append(entity)
                continue

            # Multiple entities of same type, merge descriptions
            descriptions = [entity.description for entity in type_group]
            prompt = f"""Please merge these descriptions about {name_lower} (a {type_lower}) into one comprehensive description:

Descriptions to merge:
{chr(10).join(f'- {desc}' for desc in descriptions)}

Merge them into one comprehensive description that captures all unique information without redundancy."""

            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                response_format=MERGE_DESCRIPTIONS_FORMAT,
            )

            merged_description = json.loads(response.choices[0].message.content)[
                "merged_description"
            ]

            # Create new entity with merged description and renamed if needed
            new_name = (
                f"{type_group[0].name} ({type_group[0].type})"
                if should_add_type
                else type_group[0].name
            )
            deduplicated_entities.append(
                ExtractedEntity(
                    name=new_name,
                    type=type_group[0].type,
                    description=merged_description,
                )
            )
    return deduplicated_entities


class GraphWriter:
    # Opens read + Write connection. Use "with" as setup, unless there is explicit reason not to.
    # If used outside "with" always remember to close connection explicity, even if code errors.
    def __init__(
        self,
        entity_types: list[str] = [],
        entities: list[ExtractedEntity] = [],
        strict_entity_types: bool = False,
        llm_model: ChatModelEnum = ChatModelEnum.GPT41,
        graph_id: Optional[str] = None,
    ):
        self.strict_entity_types = strict_entity_types
        self.entity_types = entity_types
        self.entities = entities
        if strict_entity_types and not entity_types:
            raise ValueError(
                "Entity types must be provided if strict_entity_types is True"
            )
        if llm_model == ChatModelEnum.GPT4o:
            self.client = get_async_gpt4o_client()
            self.sync_client = get_gpt41_client()  # Use GPT41 as fallback for now
            self.model_name = "gpt-4o"
        elif llm_model == ChatModelEnum.GPT35:
            self.client = get_async_azure_client()
            self.sync_client = get_gpt41_client()  # Use GPT41 as fallback for now
            self.model_name = "gpt-35-turbo"
        elif llm_model == ChatModelEnum.GPT41:
            self.client = get_gpt41_client_async()
            self.sync_client = get_gpt41_client()
            self.model_name = "gpt-4.1"
        elif llm_model == ChatModelEnum.GPT5:
            self.client = get_gpt5_client_async()
            self.sync_client = get_gpt5_client()
            self.model_name = "gpt-5"
        else:
            raise ValueError(f"Model {llm_model} Not Found or Not Implemented")
        self.relationships: list[EntityRelationship] = []
        self.kuzu_manager = KuzuManager()
        self.graph_id = graph_id or str(uuid4())

    @classmethod
    def load_from_db(cls, graph_id: str) -> GraphWriter:
        """
        Load a graph from KuzuDB.

        Args:
            graph_id: The ID of the graph to load

        Returns:
            A Graph instance with loaded entities and relationships
        """
        kuzu_manager = KuzuManager()

        # Create new graph instance with the specified graph_id
        graph = cls(graph_id=graph_id)

        # Load entities
        result = kuzu_manager.get_connection(graph_id).execute(
            """
            MATCH (e:Entity)
            RETURN e.name as name, e.type as type, e.description as description
        """
        )
        while result.has_next():
            entity_data = result.get_next()
            graph.entities.append(
                ExtractedEntity(
                    name=entity_data[0],  # name
                    type=entity_data[1],  # type
                    description=entity_data[2],  # description
                )
            )

        # Load relationships
        result = kuzu_manager.get_connection(graph_id).execute(
            """
            MATCH (source:Entity)-[r:Relationship]->(target:Entity)
            RETURN source.name as source_name, target.name as target_name,
                   r.description as rel_description, r.type as rel_type
        """
        )
        while result.has_next():
            rel_data = result.get_next()
            graph.relationships.append(
                EntityRelationship(
                    source_entity=rel_data[0],  # source_name
                    target_entity=rel_data[1],  # target_name
                    relationship_description=rel_data[2],  # rel_description
                    relationship_type=rel_data[3],  # rel_type
                )
            )

        return graph

    def _save_graph(self) -> None:
        """Save the graph to database"""
        try:
            # Clear existing data first
            self.kuzu_manager.clear_database(self.graph_id)

            # Save entities with embeddings
            logger.info(
                f"Saving {len(self.entities)} entities to graph {self.graph_id}"
            )
            for entity in self.entities:
                try:
                    # Generate embeddings for entity name and description
                    import asyncio

                    from app.utils.openai_utils import generate_embedding_async

                    # Get the current event loop or create a new one
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # If we're in an async context, we need to use a different approach
                            # For now, generate embeddings synchronously as fallback
                            from app.utils.openai_utils import generate_embedding

                            name_embedding = generate_embedding(entity.name)
                            description_embedding = generate_embedding(
                                entity.description or ""
                            )
                        else:
                            # Use async version if not in running loop
                            name_embedding = loop.run_until_complete(
                                generate_embedding_async(entity.name)
                            )
                            description_embedding = loop.run_until_complete(
                                generate_embedding_async(entity.description or "")
                            )
                    except RuntimeError:
                        # Fallback to synchronous version
                        from app.utils.openai_utils import generate_embedding

                        name_embedding = generate_embedding(entity.name)
                        description_embedding = generate_embedding(
                            entity.description or ""
                        )

                    # Save entity with embeddings
                    self.kuzu_manager.save_entity_with_embeddings(
                        graph_id=self.graph_id,
                        name=entity.name,
                        type=entity.type,
                        description=entity.description or "",
                        name_embedding=name_embedding,
                        description_embedding=description_embedding,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to save entity with embeddings {entity.name}: {str(e)}"
                    )
                    # Fallback to saving without embeddings
                    try:
                        self.kuzu_manager.save_entity(
                            graph_id=self.graph_id,
                            name=entity.name,
                            type=entity.type,
                            description=entity.description or "",
                        )
                    except Exception as fallback_error:
                        logger.error(
                            f"Failed to save entity {entity.name}: {str(fallback_error)}"
                        )
                        continue

            # Save relationships
            logger.info(
                f"Saving {len(self.relationships)} relationships to graph {self.graph_id}"
            )
            for rel in self.relationships:
                # Find source and target entities
                source_entity = next(
                    (e for e in self.entities if e.name == rel.source_entity), None
                )
                target_entity = next(
                    (e for e in self.entities if e.name == rel.target_entity), None
                )
                if not (source_entity):
                    logger.warning(
                        f"Relationship Source Entity not found: {rel.source_entity}"
                    )
                    continue
                if not (target_entity):
                    logger.warning(
                        f"Relationship Target Entity not found: {rel.target_entity}"
                    )
                    continue
                try:
                    self.kuzu_manager.save_relationship(
                        graph_id=self.graph_id,
                        source_name=source_entity.name,
                        source_type=source_entity.type,
                        target_name=target_entity.name,
                        target_type=target_entity.type,
                        description=rel.relationship_description,
                        type=rel.relationship_type,
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to save relationship: "
                        f"{source_entity.name} ({source_entity.type}) -> {target_entity.name} ({target_entity.type}) "
                        f"with type '{rel.relationship_type}' and description '{rel.relationship_description}': {str(e)}",
                    )
                    continue

            logger.info(
                f"Successfully saved graph with {len(self.entities)} entities and {len(self.relationships)} relationships"
            )
        except Exception as e:
            logger.error(f"Failed to save graph: {str(e)}")
            raise

    def _get_entity_response_format(self) -> dict:
        entity_types = self.entity_types if self.strict_entity_types else None
        return get_entity_response_format(entity_types)

    async def extract_entities(self, txt: str) -> list[ExtractedEntity]:
        entities = await get_entities(
            self.client, txt, self.entity_types, self.strict_entity_types
        )
        self.entities.extend(entities)
        return entities

    def extract_entities_sync(self, txt: str) -> list[ExtractedEntity]:
        """Sync version of extract_entities"""
        entities = get_entities_sync(
            self.sync_client, txt, self.entity_types, self.strict_entity_types
        )
        self.entities.extend(entities)
        return entities

    def _get_entity_relationship_format(self) -> dict:
        entities = list({entity.name for entity in self.entities})
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "entity_relationships",
                "description": "Describes relationships between identified entities.",
                "schema": {
                    "type": "object",
                    "properties": {
                        "relationships": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source_entity": {
                                        "type": "string",
                                        "enum": entities,
                                    },
                                    "target_entity": {
                                        "type": "string",
                                        "enum": entities,
                                    },
                                    "relationship_description": {
                                        "type": "string",
                                        "description": "Explanation as to why you think the source entity and the target entity are related to each other",
                                    },
                                    "relationship_type": {
                                        "type": "string",
                                        "decription": "1-3 word description summary of the relationship, i.e. 'teacher' or 'born in'.",
                                    },
                                },
                                "required": [
                                    "source_entity",
                                    "target_entity",
                                    "relationship_description",
                                    "relationship_strength",
                                ],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["relationships"],
                    "additionalProperties": False,
                },
            },
        }

    async def extract_relationships(self, txt: str) -> list[EntityRelationship]:
        response = await self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": txt,
                }
            ],
            response_format=self._get_entity_relationship_format(),
        )
        relationships_data = json.loads(response.choices[0].message.content)[
            "relationships"
        ]
        relationships = []
        entities = list({entity.name for entity in self.entities})
        for relationship in relationships_data:
            entity_relationship = EntityRelationship(
                source_entity=relationship["source_entity"],
                target_entity=relationship["target_entity"],
                relationship_description=relationship["relationship_description"],
                relationship_type=relationship["relationship_type"],
            )

            if entity_relationship.source_entity not in entities:
                logger.warn(
                    f"invalid source entity for relationship: {entity_relationship.source_entity}"
                )
                continue
            if entity_relationship.target_entity not in entities:
                logger.warn(
                    f"invalid target entity for relationship: {entity_relationship.target_entity}"
                )
                continue
            self.relationships.append(entity_relationship)
            relationships.append(entity_relationship)
        return relationships

    def extract_relationships_sync(self, txt: str) -> list[EntityRelationship]:
        """Sync version of extract_relationships"""
        response = self.sync_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": txt,
                }
            ],
            response_format=self._get_entity_relationship_format(),
        )
        relationships_data = json.loads(response.choices[0].message.content)[
            "relationships"
        ]
        relationships = []
        entities = list({entity.name for entity in self.entities})
        for relationship in relationships_data:
            entity_relationship = EntityRelationship(
                source_entity=relationship["source_entity"],
                target_entity=relationship["target_entity"],
                relationship_description=relationship["relationship_description"],
                relationship_type=relationship["relationship_type"],
            )

            if entity_relationship.source_entity not in entities:
                logger.warn(
                    f"invalid source entity for relationship: {entity_relationship.source_entity}"
                )
                continue
            if entity_relationship.target_entity not in entities:
                logger.warn(
                    f"invalid target entity for relationship: {entity_relationship.target_entity}"
                )
                continue
            self.relationships.append(entity_relationship)
            relationships.append(entity_relationship)
        return relationships

    def _get_merge_descriptions_format(self) -> dict:
        return MERGE_DESCRIPTIONS_FORMAT

    async def deduplicate_entities(self) -> list[ExtractedEntity]:
        """
        Deduplicate entities with the same name and type (case insensitive).
        Uses LLM to merge descriptions of duplicate entities.
        For entities with the same name but different types, renames them to include their type.

        Returns:
            A list of deduplicated entities
        """
        # Group entities by lowercase name
        entity_groups = {}
        for entity in self.entities:
            key = entity.name.lower()
            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(entity)

        # Process each group
        deduplicated_entities = []
        for name_lower, group in entity_groups.items():
            if len(group) == 1:
                # No duplicates, keep as is
                deduplicated_entities.append(group[0])
                continue

            # Group by type
            type_groups = {}
            for entity in group:
                type_key = entity.type.lower()
                if type_key not in type_groups:
                    type_groups[type_key] = []
                type_groups[type_key].append(entity)

            # Only add type to names if there are multiple types
            should_add_type = len(type_groups) > 1

            # Process each type group
            for type_lower, type_group in type_groups.items():
                if len(type_group) == 1:
                    # Only one entity of this type
                    entity = type_group[0]
                    if should_add_type:
                        entity.name = f"{entity.name} ({entity.type})"
                    deduplicated_entities.append(entity)
                    continue

                # Multiple entities of same type, merge descriptions
                descriptions = [entity.description for entity in type_group]
                prompt = f"""Please merge these descriptions about {name_lower} (a {type_lower}) into one comprehensive description:

Descriptions to merge:
{chr(10).join(f'- {desc}' for desc in descriptions)}

Merge them into one comprehensive description that captures all unique information without redundancy."""

                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=self._get_merge_descriptions_format(),
                )

                merged_description = json.loads(response.choices[0].message.content)[
                    "merged_description"
                ]

                # Create new entity with merged description and renamed if needed
                new_name = (
                    f"{type_group[0].name} ({type_group[0].type})"
                    if should_add_type
                    else type_group[0].name
                )
                deduplicated_entities.append(
                    ExtractedEntity(
                        name=new_name,
                        type=type_group[0].type,
                        description=merged_description,
                    )
                )

        # Update the graph's entities
        numb_entities_original = len(self.entities)
        self.entities = deduplicated_entities
        logger.info(
            f"Deduplicated entities (from {numb_entities_original} to {len(deduplicated_entities)})"
        )
        return deduplicated_entities

    def deduplicate_entities_sync(self) -> list[ExtractedEntity]:
        """Sync version of deduplicate_entities"""
        # Group entities by lowercase name
        entity_groups = {}
        for entity in self.entities:
            key = entity.name.lower()
            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(entity)

        # Process each group
        deduplicated_entities = []
        for name_lower, group in entity_groups.items():
            if len(group) == 1:
                # No duplicates, keep as is
                deduplicated_entities.append(group[0])
                continue

            # Group by type
            type_groups = {}
            for entity in group:
                type_key = entity.type.lower()
                if type_key not in type_groups:
                    type_groups[type_key] = []
                type_groups[type_key].append(entity)

            # Only add type to names if there are multiple types
            should_add_type = len(type_groups) > 1

            # Process each type group
            for type_lower, type_group in type_groups.items():
                if len(type_group) == 1:
                    # Only one entity of this type
                    entity = type_group[0]
                    if should_add_type:
                        entity.name = f"{entity.name} ({entity.type})"
                    deduplicated_entities.append(entity)
                    continue

                # Multiple entities of same type, merge descriptions
                descriptions = [entity.description for entity in type_group]
                prompt = f"""Please merge these descriptions about {name_lower} (a {type_lower}) into one comprehensive description:

Descriptions to merge:
{chr(10).join(f'- {desc}' for desc in descriptions)}

Merge them into one comprehensive description that captures all unique information without redundancy."""

                response = self.sync_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=self._get_merge_descriptions_format(),
                )

                merged_description = json.loads(response.choices[0].message.content)[
                    "merged_description"
                ]

                # Create new entity with merged description and renamed if needed
                new_name = (
                    f"{type_group[0].name} ({type_group[0].type})"
                    if should_add_type
                    else type_group[0].name
                )
                deduplicated_entities.append(
                    ExtractedEntity(
                        name=new_name,
                        type=type_group[0].type,
                        description=merged_description,
                    )
                )

        # Update the graph's entities
        numb_entities_original = len(self.entities)
        self.entities = deduplicated_entities
        logger.info(
            f"Deduplicated entities (from {numb_entities_original} to {len(deduplicated_entities)})"
        )
        return deduplicated_entities

    def deduplicate_relationships(self) -> list[EntityRelationship]:
        """
        Deduplicate relationships in the graph.
        Removes exact and case-insensitive duplicates (by source, target, type, and description).
        Also merges reversed/symmetric duplicates with the same description and type.

        Returns:
            A list of deduplicated EntityRelationship objects.
        """
        seen = set()
        deduped_relationships = []

        for rel in self.relationships:
            # Normalize components
            src = rel.source_entity.strip().lower()
            tgt = rel.target_entity.strip().lower()
            rtype = rel.relationship_type.strip().lower()
            rdesc = rel.relationship_description.strip().lower()

            # Create keys for checking duplicates (both directions for symmetry)
            key = (src, tgt, rtype, rdesc)
            reverse_key = (tgt, src, rtype, rdesc)

            if key in seen or reverse_key in seen:
                continue  # Skip duplicate
            seen.add(key)

            # Add to deduplicated list
            deduped_relationships.append(rel)
        numb_relationships_original = len(self.relationships)
        self.relationships = deduped_relationships
        logger.info(
            f"Deduplicated relationships (from {numb_relationships_original} to {len(deduped_relationships)})"
        )
        return deduped_relationships

    def build_communities(self):
        raise NotImplementedError()

    def summarize_communities(self):
        raise NotImplementedError()

    def close_connection(self):
        """Close the connection for this graph"""
        self.kuzu_manager.close_connection(self.graph_id)

    def close_database(self):
        """Close both connection and database for this graph"""
        self.kuzu_manager.close_database(self.graph_id)

    def __enter__(self):
        """Context manager entry point"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point - automatically close database"""
        try:
            self.close_database()
        except Exception as e:
            logger.error(
                f"Error closing database in context manager: {str(e)}", exc_info=True
            )
            # Don't suppress the original exception if there was one
            if exc_type is None:
                raise


class GraphReader:
    def __init__(self, graph_id: str):
        """
        Initialize a GraphReader for a specific graph.

        Args:
            graph_id: The ID of the graph to read from
        """
        self.graph_id = graph_id
        self.kuzu_manager = KuzuManager()

    def __enter__(self):
        """Enter the runtime context for the GraphReader"""
        # Initialize connection when entering context
        try:
            self._connection = self.kuzu_manager.get_connection(
                self.graph_id, read_only=True
            )
            logger.debug(f"Opened graph reader connection for graph {self.graph_id}")
        except Exception as e:
            logger.error(
                f"Failed to open graph reader connection for {self.graph_id}: {str(e)}"
            )
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context for the GraphReader"""
        try:
            # Perform any cleanup operations
            if hasattr(self, "_connection"):
                logger.debug(
                    f"Closing graph reader connection for graph {self.graph_id}"
                )
                # Note: KuzuManager handles connection pooling, but we could add
                # specific cleanup logic here if needed in the future
                delattr(self, "_connection")
        except Exception as e:
            logger.warning(
                f"Error during GraphReader cleanup for {self.graph_id}: {str(e)}"
            )
            # Don't suppress the original exception if there was one
            if exc_type is None:
                raise

        # Return False to not suppress any exceptions that occurred in the with block
        return False

    def execute_query(
        self, cypher_query: str, parameters: Optional[dict] = None
    ) -> kuzu.QueryResult:
        """
        Execute a Cypher query on the graph.

        Args:
            cypher_query: The Cypher query to execute
            parameters: Optional parameters for the query

        Returns:
            A Kuzu QueryResult object containing the query results
        """
        try:
            connection = self.kuzu_manager.get_connection(self.graph_id, read_only=True)
            if parameters:
                result = connection.execute(cypher_query, parameters)
            else:
                result = connection.execute(cypher_query)
            return result
        except Exception as e:
            logger.error(
                f"Failed to execute query on graph {self.graph_id}: {str(e)}",
                exc_info=True,
            )
            raise

    def get_connection(self) -> kuzu.Connection:
        """
        Get the direct connection to the graph database.

        Returns:
            A Kuzu Connection object for direct database access
        """
        return self.kuzu_manager.get_connection(self.graph_id, read_only=True)

    def get_entities_and_relationships(
        self, limit: Optional[int] = None
    ) -> GraphEntitiesRelationshipsResponse:
        """
        Fetch entities and relationships for this graph in serializable format.
        Returns at most 'limit' entities and relationships, or all if limit is None.
        Args:
            limit (Optional[int]): Maximum number of entities and relationships to return,
                                 if None returns all entities and relationships
        Returns:
            GraphEntitiesRelationshipsResponse: Object containing entities, relationships, and total counts
        """
        # Get total count of entities
        entity_count_query = """
            MATCH (e:Entity)
            RETURN count(e) as total_entities
            """
        entity_count_result = self.execute_query(entity_count_query)
        total_entities = 0
        if entity_count_result.has_next():
            total_entities = entity_count_result.get_next()[0]

        # Get total count of relationships
        relationship_count_query = """
            MATCH (source:Entity)-[r:Relationship]->(target:Entity)
            RETURN count(r) as total_relationships
            """
        relationship_count_result = self.execute_query(relationship_count_query)
        total_relationships = 0
        if relationship_count_result.has_next():
            total_relationships = relationship_count_result.get_next()[0]

        # Fetch entities with optional limit
        if limit is not None:
            entity_query = f"""
                MATCH (e:Entity)
                RETURN e.name as name, e.type as type, e.description as description
                LIMIT {limit}
                """
        else:
            entity_query = """
                MATCH (e:Entity)
                RETURN e.name as name, e.type as type, e.description as description
                """
        entity_result = self.execute_query(entity_query)
        entities = []
        while entity_result.has_next():
            row = entity_result.get_next()
            entities.append(
                {
                    "name": row[0],
                    "type": row[1],
                    "description": row[2],
                }
            )

        # Fetch relationships with optional limit
        if limit is not None:
            relationship_query = f"""
                MATCH (source:Entity)-[r:Relationship]->(target:Entity)
                RETURN source.name as source_entity, target.name as target_entity, r.type as relationship_type, r.description as relationship_description
                LIMIT {limit}
                """
        else:
            relationship_query = """
                MATCH (source:Entity)-[r:Relationship]->(target:Entity)
                RETURN source.name as source_entity, target.name as target_entity, r.type as relationship_type, r.description as relationship_description
                """
        relationship_result = self.execute_query(relationship_query)
        relationships = []
        while relationship_result.has_next():
            row = relationship_result.get_next()
            relationships.append(
                {
                    "source_entity": row[0],
                    "target_entity": row[1],
                    "relationship_type": row[2],
                    "relationship_description": row[3],
                }
            )

        return GraphEntitiesRelationshipsResponse(
            entities=entities,
            relationships=relationships,
            total_entities=total_entities,
            total_relationships=total_relationships,
        )

    def close_connection(self):
        """Close the connection for this graph"""
        self.kuzu_manager.close_connection(self.graph_id)

    def close_database(self):
        """Close both connection and database for this graph"""
        self.kuzu_manager.close_database(self.graph_id)

    def is_read_only(self) -> bool:
        """Check if this graph is opened in read-only mode (always True for GraphReader)"""
        return True
