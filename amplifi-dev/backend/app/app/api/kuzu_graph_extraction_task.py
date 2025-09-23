from typing import Optional

from app.api.deps import get_gpt41_client
from app.be_core.celery import celery
from app.be_core.logger import logger
from app.crud.document_crud import document as document_crud
from app.crud.extracted_entity_crud import extracted_entity_crud
from app.crud.graph_crud import crud_graph
from app.db.session import SyncSessionLocal
from app.schemas.graph_schema import ExtractedEntity
from app.utils.graph.extract import (
    GraphWriter,
    deduplicate_entities_sync,
    get_entities_sync,
)


@celery.task(name="tasks.kuzu_relationships_extraction_task", bind=True, acks_late=True)
def kuzu_relationships_extraction_task(
    self,
    dataset_id: str,
    graph_id: str,
):
    """
    Celery task for extracting relationships from Postgres and processing it into the Kuzu graph database.
    Args:
        dataset_id (str): The dataset UUID
        graph_id (str): The graph UUID (newly created)
    """
    logger.info(
        f"[KUZU_RELATIONSHIPS_EXTRACTION_TASK] Called for dataset_id={dataset_id}, graph_id={graph_id}"
    )

    try:
        # Get entities info with sync session
        with SyncSessionLocal() as db_session:
            entities: list[ExtractedEntity] = (
                extracted_entity_crud.get_entities_info_sync(
                    db_session=db_session, graph_id=graph_id
                )
            )

            with GraphWriter(graph_id=graph_id, entities=entities) as graph:
                # Can optimize this using db cursor and streaming chunks instead of loading everything into list
                document_ids = document_crud.get_document_ids_by_dataset_sync(
                    db=db_session, dataset_id=dataset_id
                )
                for id in document_ids:
                    # can force garbage collection here if memory usesage very high (and not streaming)
                    chunks = document_crud.get_chunks_text_by_document_sync(
                        db=db_session, document_id=id
                    )
                    for chunk in chunks:
                        # Use sync GraphWriter operations
                        graph.extract_relationships_sync(chunk)

                graph.deduplicate_relationships()
                graph._save_graph()
                numb_relationships = len(graph.relationships)

        # Update graph status with sync session
        with SyncSessionLocal() as db_session:
            crud_graph.update_graph_status_sync(
                graph_id=graph_id,
                new_status="success",
                db_session=db_session,
                field="relationships",
            )

        logger.info(
            f"[KUZU_GRAPH_EXTRACTION_TASK] Wrote {numb_relationships} relationships"
        )
        return {
            "success": True,
            "message": "Kuzu relationships extraction task executed. Relationships extracted and saved.",
            "dataset_id": dataset_id,
            "graph_id": graph_id,
        }
    except Exception as e:
        logger.error(
            f"[KUZU_GRAPH_EXTRACTION_TASK] Extraction failed: {e}", exc_info=True
        )
        # Update graph status with sync session
        with SyncSessionLocal() as db_session:
            crud_graph.update_graph_status_sync(
                graph_id=graph_id,
                new_status="failed",
                db_session=db_session,
                field="relationships",
            )
        return {
            "success": False,
            "message": f"Kuzu graph extraction task failed: {e}",
            "dataset_id": dataset_id,
            "graph_id": graph_id,
        }


@celery.task(name="tasks.extract_text_from_dataset", bind=True, acks_late=True)
def extract_text_from_dataset(
    self, dataset_id: str, graph_id: str, entity_types: Optional[list[str]] = None
):
    """
    Extract text from a dataset using sync database operations.
    """

    client = get_gpt41_client()

    with SyncSessionLocal() as db_session:
        document_ids = document_crud.get_document_ids_by_dataset_sync(
            db=db_session, dataset_id=dataset_id
        )
        entities: list[ExtractedEntity] = []

        for id in document_ids:
            chunks = document_crud.get_chunks_text_by_document_sync(
                db=db_session, document_id=id
            )
            for chunk in chunks:
                # Use sync entity extraction
                chunk_entities = get_entities_sync(client, chunk, entity_types)
                entities.extend(chunk_entities)

        # Deduplicate entities using sync method
        entities = deduplicate_entities_sync(entities, client)

        # Save entities with sync method
        extracted_entity_crud.create_batch_sync(
            db_session=db_session, graph_id=graph_id, items=entities
        )

        # Update graph status with sync method
        crud_graph.update_graph_status_sync(
            db_session=db_session,
            graph_id=graph_id,
            new_status="success",
            field="entities",
        )

        return {
            "success": True,
            "message": "Entities extracted and saved.",
            "dataset_id": dataset_id,
            "graph_id": graph_id,
        }
