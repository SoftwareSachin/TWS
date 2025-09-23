from fastapi import HTTPException

from app.api.v2.endpoints.chatsession import _get_db_details
from app.be_core.config import settings
from app.be_core.logger import logger
from app.be_core.vanna_connector_manager import VannaConnectorManager
from app.crud.dataset_crud_v2 import dataset_v2
from app.db.connection_handler import DatabaseConnectionHandler
from app.db.session import SessionLocal
from app.tools.system_tools.schemas.texttosql_schema import (
    AgentDeps,
    SQLProcessResponse,
)
from app.utils.csv_export_utils import (
    export_large_dataframe_to_csv,
    get_workspace_id_from_dataset_ids,
)


# Tool 1: SQL Processing (Updated with schema)
async def process_sql_chat_app(request_data: AgentDeps) -> SQLProcessResponse:
    """
    Handles SQL chat app operations: database connection, SQL generation,
    and execution. Returns table data for further processing.
    """
    logger.info("Invoke SQL tool")
    logger.info(f"SQL request tool {str(request_data)}")

    db_session = None
    try:
        db_session = SessionLocal()

        result = None
        for dataset_id in request_data.dataset_ids:
            source_id = await dataset_v2.get_source_id_by_dataset_id(
                dataset_id=dataset_id
            )
            logger.info(f"source id we get {source_id} for dataset_id {dataset_id}")
            if not source_id:
                logger.warning(f"source_id not found for dataset_id {dataset_id}")
                continue

            try:
                db_details, db_type = await _get_db_details(source_id)
                logger.info(f"Resolved db_type {db_type}")
                logger.info(f"getting the db_details {db_details}")

                logger.info("into the vannaconnector manager")
                vector_db_name = f"{db_details['database_name']}_{dataset_id}"
                vn = VannaConnectorManager(
                    source_db_name=vector_db_name,
                    llm_model=request_data.llm_model,
                ).vn
                logger.info("out of the vannaconnector manager")
                DatabaseConnectionHandler.connect_database(vn, db_details, db_type)
                logger.info(
                    f"✅ Connected to {db_type} database: {db_details['database_name']}"
                )

                # Step 4: Generate SQL using LLM and execute
                generated_sql = vn.generate_sql(
                    request_data.query, allow_llm_to_see_data=True
                )
                logger.info(f"Generated SQL: {generated_sql}")

                # Validate generated SQL to prevent DDL statements
                if generated_sql and generated_sql.strip().upper().startswith(
                    ("CREATE", "DROP", "ALTER", "TRUNCATE")
                ):
                    logger.warning(
                        f"Generated SQL contains DDL statement, skipping execution: {generated_sql}"
                    )
                    raise HTTPException(
                        status_code=400,
                        detail="Generated SQL contains DDL statements which are not allowed for security reasons.",
                    )

                try:
                    data_frame = vn.run_sql(generated_sql)
                    logger.debug(f"📊 Pulled DataFrame:\n{data_frame}")

                    csv_file_id = None
                    filename = None
                    rows, cols = data_frame.shape
                    if rows > settings.MAX_SQL_ROWS or cols > settings.MAX_SQL_COLUMNS:
                        logger.info(
                            f"DataFrame has {rows} rows and {cols} columns, exporting to CSV file"
                        )

                        # Get workspace ID from dataset IDs
                        workspace_id = await get_workspace_id_from_dataset_ids(
                            request_data.dataset_ids, db_session
                        )

                        if workspace_id:
                            # Export DataFrame to CSV using utility function
                            csv_result = await export_large_dataframe_to_csv(
                                data_frame=data_frame,
                                workspace_id=workspace_id,
                                db_session=db_session,
                                filename_prefix="sql_export",
                            )

                            if csv_result:
                                csv_file_id, filename = csv_result
                            else:
                                csv_file_id = None
                                filename = None
                        else:
                            logger.warning(
                                "Could not determine workspace_id, skipping CSV export"
                            )

                    # Always truncate dataframe for agent response (keep first 50 rows and first 50 columns)
                    data_frame = data_frame.head(100)
                    data_frame = data_frame.iloc[:, :100]
                    logger.info(f"Final DataFrame shape for agent: {data_frame.shape}")

                    # Process date columns for JSON serialization
                    for col in data_frame.columns:
                        if col == "Date":
                            data_frame[col] = data_frame[col].apply(
                                lambda x: (
                                    x.strftime("%Y-%m-%d")
                                    if hasattr(x, "strftime")
                                    else x
                                )
                            )

                    answer_as_json = data_frame.to_json(orient="records")
                    full_table_data = data_frame.to_dict(orient="records")

                    result = SQLProcessResponse(
                        generated_sql=generated_sql,
                        answer=answer_as_json,
                        table_data=full_table_data,
                        query=request_data.query,
                        csv_file_id=csv_file_id,
                        csv_file_name=filename,
                    )
                    break

                except Exception as e:
                    logger.warning(
                        f"SQL execution failed for dataset_id {dataset_id}. Exception: {e}",
                        exc_info=True,
                    )
                    continue

            except Exception as e:
                logger.warning(
                    f"Failed to initialize DB connection for dataset_id {dataset_id}. Exception: {e}",
                    exc_info=True,
                )
                continue

        if result:
            return result
        else:
            logger.error(
                f"Unable to process query for any provided dataset_id. request_data: {request_data}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=400,
                detail="Unable to process query for any provided dataset_id.",
            )

    except Exception as e:
        logger.error(f"Unexpected error in process_sql_chat_app: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}",
        )
    finally:
        if db_session:
            await db_session.close()
