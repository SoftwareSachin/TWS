import time
import traceback
from datetime import timedelta
from typing import Dict, List, Optional, Union
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi_pagination import Params

from app import crud
from app.api import deps
from app.be_core.logger import logger
from app.be_core.vanna_connector_manager import VannaConnectorManager
from app.crud.chat_app_crud import chatapp
from app.crud.chat_history_crud import chathistory
from app.crud.chat_session_crud import chatsession
from app.crud.dataset_crud import dataset
from app.crud.source_connector_crud import CRUDSource
from app.models.chat_app_generation_config_model import ChatAppGenerationConfigBase
from app.models.chat_history_model import ChatHistory
from app.models.source_model import Source
from app.schemas.chat_schema import (
    ChatAppRagGenerationRequest,
    ChatHistoryLine,
    IChatSessionCreate,
    IChatSessionRead,
)
from app.schemas.common_schema import IOrderEnum
from app.schemas.rag_generation_schema import (
    ChatModelEnum,
    GenerationSettings,
    IFileContextAggregation,
    IFileContextChunk,
    IWorkspaceGenerationResponse,
    RagContext,
    RagContextWrapper,
)
from app.schemas.response_schema import (
    IGetResponsePaginated,
    IPostResponseBase,
    IPutResponseBase,
    create_response,
)
from app.schemas.role_schema import IRoleEnum
from app.schemas.search_schema import (
    ImageSearchResult,
    R2RVectorSearchResult,
)
from app.schemas.user_schema import UserData
from app.utils.exceptions.common_exception import IdNotFoundException
from app.utils.llm_fns.azure_openai import (  # noqa: F401
    query_llm_with_history,
    query_openai_chat_history,
)

router = APIRouter()
chatsession_crud = chatsession
chathistory_crud = chathistory
chatapp_crud = chatapp


@router.post("/chat_app/{chatapp_id}/chat_session")
async def create_chatsession(
    chatapp_id: UUID,
    request_data: IChatSessionCreate,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.chatapp_check),
) -> IPostResponseBase[IChatSessionRead]:
    """
    Creates a new chat session in a workspace.

    Required roles:
    - admin
    - developer
    """
    new_chatsession = await chatsession_crud.create_chat_session(
        obj_in=request_data, chatapp_id=chatapp_id, user_id=current_user.id
    )
    return create_response(
        data=new_chatsession, message="Chat Session created successfully"
    )


@router.put("/chat_app/{chatapp_id}/chat_session/{chat_session_id}")
async def update_chatsession(
    chatapp_id: UUID,
    chat_session_id: UUID,
    request_data: IChatSessionCreate,
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.developer, IRoleEnum.member]
        )
    ),
    _=Depends(deps.chatsession_check),
) -> IPutResponseBase[IChatSessionRead]:
    """
    Update a chat session in a chatapp.

    Required roles:
    - admin
    - developer
    - member
    """
    updated_chatsession = await chatsession_crud.update_chat_session(
        obj_in=request_data, chat_session_id=chat_session_id, chatapp_id=chatapp_id
    )
    return create_response(
        data=updated_chatsession, message="Chat Session updated successfully"
    )


@router.delete("/chat_app/{chatapp_id}/chat_session/{chat_session_id}")
async def delete_chatsession(
    chatapp_id: UUID,
    chat_session_id: UUID,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.chatsession_check),
) -> IPutResponseBase[IChatSessionRead]:
    """
    Delete a chat session in a chatapp.

    Required roles:
    - admin
    - developer
    - member
    """
    deleted_chatsession = await chatsession_crud.soft_delete_chatapp_by_id(
        chat_session_id=chat_session_id
    )
    return create_response(
        data=deleted_chatsession, message="Chat Session deleted successfully"
    )


@router.get(
    "/chat_app/{chatapp_id}/chat_session",
    response_model=IGetResponsePaginated[IChatSessionRead],
)
async def get_chatsessions(
    chatapp_id: UUID,
    pagination_params: Params = Depends(),
    order: IOrderEnum = Query(IOrderEnum.ascendent),
    current_user: UserData = Depends(
        deps.get_current_user(
            required_roles=[IRoleEnum.admin, IRoleEnum.member, IRoleEnum.developer]
        )
    ),
    _=Depends(deps.chatapp_check),
) -> IGetResponsePaginated[IChatSessionRead]:
    """
    Retrieves all chatsessions for a chatapp.

    Required roles:
    - admin
    - member
    - developer
    """
    paginated_chatsessions = await chatsession_crud.get_chat_sessions(
        chatapp_id=chatapp_id,
        user_id=current_user.id,
        pagination_params=pagination_params,
        order=order,
    )

    pydantic_chatsessions = [
        IChatSessionRead.model_validate(chatsession)
        for chatsession in paginated_chatsessions.items
    ]

    paginated_response = IGetResponsePaginated.create(
        items=pydantic_chatsessions,
        total=paginated_chatsessions.total,
        params=pagination_params,
    )

    return create_response(data=paginated_response)


@router.get("/chat_app/{chatapp_id}/chat_session/{chat_session_id}/chat_history")
async def get_chat_history(
    chatapp_id: UUID,
    chat_session_id: UUID,
    pagination_params: Params = Depends(),
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.chatsession_check),
) -> IGetResponsePaginated[ChatHistoryLine]:
    paginated_chat_histories = await chathistory_crud.get_chat_history_by_session_id(
        chat_session_id=chat_session_id,
        pagination_params=pagination_params,
    )
    pydantic_chat_histories = [
        ChatHistoryLine.from_orm(dataset) for dataset in paginated_chat_histories.items
    ]
    paginated_response = IGetResponsePaginated.create(
        items=pydantic_chat_histories,
        total=paginated_chat_histories.total,
        params=pagination_params,
    )

    if paginated_response:
        return create_response(data=paginated_response)
    else:
        raise IdNotFoundException("No history found for this chat session.")


async def get_title(query: str) -> str:
    system_prompt = """
    You are an assistant that generates short titles for chat sessions.
    Based on the query, generate a short (around 5 words) title for the chat.
    Respond with only the title, nothing else.
    """
    user_prompt = f"Query to generate title for: {query}\n Respond with only the title (around 5 words long)"
    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = await query_openai_chat_history(history, ChatModelEnum.GPT4o)

    return response.choices[0].message.content


async def process_sql_chat_app(
    chatapp_id: UUID,
    chat_session_id: UUID,
    request_data: ChatAppRagGenerationRequest,
    llm_model: str,
) -> dict:
    """
    Handles SQL chat app operations: database connection, SQL generation,
    execution, and response preparation. (No chat history handling here.)
    """
    dataset_id = await chatapp.get_dataset_id_by_chatapp_id(chatapp_id=chatapp_id)
    source_id = await dataset.get_source_id_by_dataset_id(dataset_id=dataset_id)
    if not source_id:
        raise HTTPException(
            status_code=400,
            detail="source_id is required to establish a database connection.",
        )

    try:
        crud_source = CRUDSource(Source)
        db_details = await crud_source.get_pg_db_details(source_id)

        if not db_details:
            raise HTTPException(
                status_code=404, detail="Invalid source_id or no database found."
            )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch DB details: {str(e)}"
        )

    # Initialize Vanna
    vn_instance = VannaConnectorManager(
        source_db_name=db_details["database_name"],
        llm_model=llm_model,
    )
    logger.info(f"Connected to PostgreSQL vector {db_details['database_name']}")
    vn = vn_instance.vn
    start_time = time.time()

    # Connect to source PostgreSQL
    vn.connect_to_postgres(
        host=db_details["host"],
        dbname=db_details["database_name"],
        user=db_details["username"],
        password=db_details["password"],
        port=db_details["port"],
    )

    elapsed_time = time.time() - start_time
    logger.info(f"Connected to PostgreSQL in {elapsed_time:.3f} seconds.")

    # Generate SQL, execute it, and create visualizations
    generated_sql = vn.generate_sql(request_data.query, allow_llm_to_see_data=True)
    logger.debug(f"Generated SQL: {generated_sql}")
    try:
        data_frame = vn.run_sql(generated_sql)
        logger.debug(f"Pulled DF: \n{data_frame}")
        plotly_code = vn.generate_plotly_code(
            question=request_data.query, sql=generated_sql, df_metadata=data_frame
        )
        logger.debug(f"Generated Plotly Code: {plotly_code}")
        plotly_figure = vn.get_plotly_figure(plotly_code, df=data_frame)

        # Convert results to JSON
        figure_as_json = plotly_figure.to_json()
        for col in data_frame.columns:
            if col == "Date":
                data_frame[col] = data_frame[col].apply(
                    lambda x: x.strftime("%Y-%m-%d")
                )
        answer_as_json = data_frame.to_json(orient="records")

        return {
            "generated_sql": generated_sql,
            "answer": answer_as_json,
            "plotly_code": plotly_code,
            "plotly_figure": figure_as_json,
        }

    except Exception as e:
        logger.warning(f"SQL Run or plotly generation failed. Exception: {e}")
        return {
            "generated_sql": generated_sql,
            "answer": "Sorry, I was unable to process your query into a SQL query and Plotly graph.",
            "plotly_code": "",
            "plotly_figure": "",
        }


def extract_chunk_info(
    result: Union[R2RVectorSearchResult, ImageSearchResult],
) -> tuple:
    if isinstance(result, R2RVectorSearchResult):
        return result.chunk_id, result.unstructured_page_number, "text"
    elif isinstance(result, ImageSearchResult) and result.chunk_metadata:
        return (
            result.chunk_metadata.chunk_id,
            None,
            result.chunk_metadata.match_type,
        )
    return None, None, "text"


def initialize_aggregated_context(context: RagContext) -> Optional[tuple]:
    if context.file and context.file.get("file_id"):
        file_id_str = str(context.file["file_id"])
        agg_context = IFileContextAggregation(
            file_id=UUID(file_id_str),
            file_name=context.file.get("filename"),
            mimetype=context.file.get("mime_type"),
            dataset_id=context.file.get("dataset_id"),
            download_url=context.download_url,
            texts=[],
            max_search_score=0.0,
        )
        return file_id_str, agg_context
    return None


def update_aggregated_context(
    aggregated_context: IFileContextAggregation,
    result: Union[R2RVectorSearchResult, ImageSearchResult],
):
    chunk_id, page_number, match_type = extract_chunk_info(result)

    aggregated_context.texts.append(
        IFileContextChunk(
            text=result.text,
            chunk_id=chunk_id,
            page_number=page_number,
            search_score=result.search_score,
            match_type=match_type,
        )
    )
    if (
        aggregated_context.max_search_score is None
        or result.search_score > aggregated_context.max_search_score
    ):
        aggregated_context.max_search_score = result.search_score


async def aggregate_file_contexts(
    contexts_found: List[RagContext],
    raw_results: List[Union[R2RVectorSearchResult, ImageSearchResult]],
) -> List[IFileContextAggregation]:
    file_aggregated_contexts: Dict[str, IFileContextAggregation] = {}

    for context in contexts_found:
        result = initialize_aggregated_context(context)
        if result:
            file_id_str, agg_context = result
            file_aggregated_contexts[file_id_str] = agg_context

    for result in raw_results:
        file_id_str = str(result.file_id)
        if file_id_str in file_aggregated_contexts:
            update_aggregated_context(file_aggregated_contexts[file_id_str], result)

    return sorted(
        file_aggregated_contexts.values(),
        key=lambda x: x.max_search_score or 0,
        reverse=True,
    )


async def generate_llm_response(
    query: str,
    contexts_found: RagContextWrapper,
    system_prompt: Optional[str],
    chat_histories: list[ChatHistory],
    gen_settings: ChatAppGenerationConfigBase,
) -> IWorkspaceGenerationResponse:
    if not system_prompt or system_prompt == "string":
        system_prompt = (
            "You are an expert at analyzing text documents and images. "
            "Use only the provided context to answer the user’s query. "
            "Do not use your own external knowledge or training data. "
            "If the context does not contain relevant information, respond that you don't have info on the subject. "
            "Do not make up information."
            "When generating lists, always place the colon after the bolded key. For example, use '**Name**:' instead of '**Name:**'."
            "When encountering text like “Dispense: 90 (ninety) capsules” or “Days Supply: 90 (ninety)”, avoid repeating both the numeric and written forms. Keep only one form — preferably the numeric value. For example, change “90 (ninety)” to just “90”."
        )
    else:
        system_prompt = system_prompt
    chat_history = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]
    for line in chat_histories:
        chat_history.append({"role": "user", "content": line.user_query})
        chat_history.append({"role": "assistant", "content": line.llm_response})

    try:
        data = await query_llm_with_history(
            query=query,
            history=chat_history,
            contexts=contexts_found,  # Extract contexts from wrapper
            generation_settings=GenerationSettings(
                chat_model=gen_settings.llm_model,
                chat_model_kwargs={
                    "temperature": gen_settings.temperature,
                    "top_p": gen_settings.top_p,
                },
            ),
        )

        return IWorkspaceGenerationResponse(
            answer=data.answer,
            contexts_found=contexts_found,
            full_response=data.full_response,
            ssml=data.ssml,
            input_tokens=data.input_tokens,
            output_tokens=data.output_tokens,
        )

    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"Error during llm query: {str(e)}\nTraceback: {tb_str}")
        raise HTTPException(status_code=500, detail={"Error during llm query": str(e)})


async def check_title(
    query: str,
    chat_histories: List[ChatHistory],
    chat_session_id: UUID,
    chatapp_id: UUID,
) -> bool:
    """
    Check if the title of the chat session is empty.
    """
    if not chat_histories:
        chatsession_record = await chatsession_crud.get_chat_session_by_id(
            chat_session_id=chat_session_id,
            chatapp_id=chatapp_id,
        )
        if abs(
            chatsession_record.updated_at - chatsession_record.created_at
        ) < timedelta(seconds=1):
            ai_title = await get_title(query=query)
            await crud.chatsession.update_chat_session(
                chat_session_id=chat_session_id,
                chatapp_id=chatapp_id,
                obj_in=IChatSessionCreate(title=ai_title),
            )
