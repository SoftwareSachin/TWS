import json
import traceback
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse  # type: ignore

from app import crud
from app.api import deps
from app.api.v2.endpoints.search import search_files
from app.be_core.logger import logger
from app.be_core.vanna_connector_manager import VannaConnectorManager
from app.crud import chat_history_crud
from app.crud.chat_app_crud import chatapp
from app.crud.chat_history_crud import chathistory
from app.crud.chat_session_crud import chatsession
from app.crud.dataset_crud_v2 import dataset_v2
from app.crud.source_connector_crud import CRUDSource
from app.db.connection_handler import DatabaseConnectionHandler
from app.models.chat_app_generation_config_model import ChatAppGenerationConfigBase
from app.models.chat_history_model import ChatHistory
from app.models.source_model import Source
from app.schemas.chat_schema import (
    ChatAppRagGenerationRequest,
    IChatAppRead,
    IChatHistoryCreate,
    IChatHistoryLineCreate,
    IChatSessionCreate,
)
from app.schemas.rag_generation_schema import (
    ChatModelEnum,
    GenerationSettings,
    IChatAppResponse,
    IFileContextAggregation,
    IFileContextChunk,
    IWorkspaceGenerationResponse,
    RagContext,
    RagContextWrapper,
    SqlChatAppRAGGenerationRead,
)
from app.schemas.response_schema import (
    IPostResponseBase,
    create_response,
)
from app.schemas.role_schema import IRoleEnum
from app.schemas.search_schema import (
    ImageSearchResult,
)
from app.schemas.user_schema import UserData
from app.utils.image_processing_utils import encode_image
from app.utils.llm_fns.azure_openai import (  # noqa: F401
    query_llm_with_history,
    query_openai_chat_history,
)

router = APIRouter()
chatsession_crud = chatsession
chathistory_crud = chathistory
chatapp_crud = chatapp


async def _fetch_source_ids(chatapp_id: UUID) -> tuple[UUID, UUID]:
    dataset_id = await chatapp.get_dataset_id_by_chatapp_id(chatapp_id=chatapp_id)
    source_id = await dataset_v2.get_source_id_by_dataset_id(dataset_id=dataset_id)
    if not source_id:
        raise HTTPException(
            status_code=400, detail="source_id is required to connect to DB."
        )
    return dataset_id, source_id


async def _get_db_details(source_id: UUID) -> tuple[dict, str]:
    crud_source = CRUDSource(Source)
    db_details = await crud_source.get_sql_db_details(source_id, db_type="mysql_db")
    db_type = "mysql" if db_details else None
    if not db_details:
        db_details = await crud_source.get_sql_db_details(source_id, db_type="pg_db")
        db_type = "postgresql" if db_details else None

    if not db_details:
        raise HTTPException(
            status_code=404, detail="Invalid source_id or no database found."
        )

    logger.info(f"DB type detected: {db_type}")
    return db_details, db_type


async def _get_chat_history(chatapp_id, chat_session_id, user_query):
    chat_histories = await chat_history_crud.chathistory.get_chat_history_by_session_id_no_pagination(
        chat_session_id=chat_session_id
    )
    if not chat_histories:
        chatsession_record = await chatsession_crud.get_chat_session_by_id(
            chat_session_id=chat_session_id, chatapp_id=chatapp_id
        )
        if abs(
            chatsession_record.updated_at - chatsession_record.created_at
        ) < timedelta(seconds=1):
            ai_title = await get_title(query=user_query)
            await crud.chatsession.update_chat_session(
                chat_session_id=chat_session_id,
                chatapp_id=chatapp_id,
                obj_in=IChatSessionCreate(title=ai_title),
            )
    return chat_histories


def _build_conversation(chat_histories):
    messages = [{"role": "system", "content": "you are a helpful assistant"}]
    for line in chat_histories:
        messages.append({"role": "user", "content": line.user_query})
        messages.append({"role": "assistant", "content": line.llm_response})
    return messages


async def _generate_response_with_vanna(
    vn, query: str, chat_history_messages: list[dict]
) -> dict:
    try:
        sql = vn.generate_sql(query, allow_llm_to_see_data=True)
        df = vn.run_sql(sql)
        plotly_code = vn.generate_plotly_code(query, sql, df)
        plotly_figure = vn.get_plotly_figure(plotly_code, df)

        if "Date" in df.columns:
            df["Date"] = df["Date"].apply(lambda x: x.strftime("%Y-%m-%d"))

        return {
            "generated_sql": sql,
            "answer": df.to_json(orient="records"),
            "plotly_code": plotly_code,
            "plotly_figure": plotly_figure.to_json(),
        }
    except Exception as e:
        logger.warning(f"SQL/Plotly generation failed: {e}")
        return {
            "generated_sql": "",
            "answer": "Could not process your query.",
            "plotly_code": "",
            "plotly_figure": "",
        }


async def _store_chat_history(
    chatapp_id, chat_session_id, query, llm_response, llm_model
):
    record = IChatHistoryLineCreate(
        user_query=query,
        contexts=[],
        llm_response=json.dumps(llm_response),
        llm_model=llm_model,
    )
    await chat_history_crud.chathistory.add_chat_session_history(
        chatapp_id=chatapp_id,
        chat_session_id=chat_session_id,
        obj_in=IChatHistoryCreate(histories=[record]),
    )


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
    logger.info(
        f"🔍 Starting SQL chat app for chatapp_id={chatapp_id}, session_id={chat_session_id}"
    )

    dataset_id, source_id = await _fetch_source_ids(chatapp_id)
    db_details, db_type = await _get_db_details(source_id)

    vn = VannaConnectorManager(
        source_db_name=db_details["database_name"], llm_model=llm_model
    ).vn
    DatabaseConnectionHandler.connect_database(vn, db_details, db_type)
    logger.info(f"✅ Connected to {db_type} database.")

    chat_histories = await _get_chat_history(
        chatapp_id, chat_session_id, request_data.query
    )
    chat_history_messages = _build_conversation(chat_histories)

    llm_response = await _generate_response_with_vanna(
        vn, request_data.query, chat_history_messages
    )

    await _store_chat_history(
        chatapp_id, chat_session_id, request_data.query, llm_response, llm_model
    )

    return llm_response


async def perform_search(
    request: Request,
    query: str,
    image_dataset_ids: List[UUID],
    max_chunks: int,
    base64_cache: dict,
) -> Tuple[List[RagContext], List[Any]]:
    contexts_found = []
    raw_results = []

    search_results = await search_files(
        query=query,
        dataset_ids=image_dataset_ids,
        top_k=max_chunks,
    )
    for result in search_results:
        file_path = result.file_path
        is_image = result.mimetype and result.mimetype.startswith("image/")

        if is_image and file_path not in base64_cache:
            base64_cache[file_path] = await encode_image(file_path)
            result.chunk_metadata.base64 = base64_cache[file_path]
        else:
            result.chunk_metadata.base64 = None

        contexts_found.append(
            RagContext(
                text=result.text,
                file={
                    "filename": file_path.split("/")[-1],
                    "filepath": str(file_path),
                    "mime_type": result.mimetype,
                    "file_id": str(result.file_id),
                    "dataset_id": str(result.dataset_id),
                },
                download_url=str(
                    request.url_for("download_file", file_id=result.file_id)
                ),
                chunk_id=result.chunk_metadata.chunk_id,
            )
        )
    raw_results.extend(search_results)
    return contexts_found, raw_results


async def get_unstructured_context_v2(
    request: Request,
    request_data: ChatAppRagGenerationRequest,
    chatapp_obj: IChatAppRead,
) -> RagContextWrapper:
    """
    Retrieves image-only RAG contexts for all datasets in the chatapp.
    """
    rag_gen_settings: ChatAppGenerationConfigBase = chatapp_obj.generation_config

    all_datasets = chatapp_obj.datasets

    base64_cache = {}
    contexts, results = await perform_search(
        request=request,
        query=request_data.query,
        image_dataset_ids=all_datasets,
        max_chunks=rag_gen_settings.max_chunks_retrieved,
        base64_cache=base64_cache,
    )

    results.sort(key=lambda x: x.search_score, reverse=True)
    top_results = results[: rag_gen_settings.max_chunks_retrieved]
    filtered_contexts = [
        context
        for context in contexts
        if any(
            str(result.file_id) == str(context.file.get("file_id"))
            for result in top_results
        )
    ]

    return RagContextWrapper(
        rag_contexts=filtered_contexts,
        raw_results=top_results,
    )


async def initialize_aggregated_context_v2(context: RagContext) -> Optional[tuple]:
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


async def update_aggregated_context_v2(
    aggregated_context: IFileContextAggregation,
    result: ImageSearchResult,
):
    aggregated_context.texts.append(
        IFileContextChunk(
            text=result.text,
            chunk_id=result.chunk_metadata.chunk_id,
            page_number=None,
            search_score=result.search_score,
            match_type=result.chunk_metadata.match_type,
            table_html=(
                getattr(result.chunk_metadata, "table_html", None)
                if getattr(result.chunk_metadata, "table_html", None)
                else None
            ),
        )
    )

    if (
        aggregated_context.max_search_score is None
        or result.search_score > aggregated_context.max_search_score
    ):
        aggregated_context.max_search_score = result.search_score


async def aggregate_file_contexts_v2(
    contexts_found: List[RagContext],
    raw_results: List[ImageSearchResult],
) -> List[IFileContextAggregation]:
    file_aggregated_contexts: Dict[str, IFileContextAggregation] = {}

    for context in contexts_found:
        result = await initialize_aggregated_context_v2(context)
        if result:
            file_id_str, agg_context = result
            file_aggregated_contexts[file_id_str] = agg_context

    for result in raw_results:
        file_id_str = str(result.file_id)
        if file_id_str in file_aggregated_contexts:
            await update_aggregated_context_v2(
                file_aggregated_contexts[file_id_str], result
            )

    return sorted(
        file_aggregated_contexts.values(),
        key=lambda x: x.max_search_score or 0,
        reverse=True,
    )


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


@router.post("/chat_app/{chatapp_id}/chat_session/{chat_session_id}/rag_generation")
async def rag_generation_chat_session(
    request: Request,
    chatapp_id: UUID,
    chat_session_id: UUID,
    request_data: ChatAppRagGenerationRequest,
    current_user: UserData = Depends(
        deps.get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(deps.chatsession_check),
) -> IPostResponseBase[IChatAppResponse]:
    try:
        chatapp_obj = await chatapp_crud.get_chatapp_by_id(chatapp_id=chatapp_id)
    except Exception as e:
        logger.error(f"Error during get chatapp: {str(e)}")
        return JSONResponse(
            content={"Error during get chatapp": str(e)}, status_code=500
        )

    if chatapp_obj.chat_app_type == "sql_chat_app":
        try:
            llm_model = (
                chatapp_obj.generation_config.llm_model
            )  # <- Get model from config
            logger.info(f"this is the model we are using:{llm_model}")
            vn_results = await process_sql_chat_app(
                chatapp_id, chat_session_id, request_data, llm_model=llm_model
            )

            # Prepare response based on SqlChatAppRAGGenerationRead schema
            response = SqlChatAppRAGGenerationRead(
                chat_app_type="sql_chat_app",
                sql_prompt=request_data.query,
                generated_sql=vn_results["generated_sql"],
                answer=vn_results["answer"],
                plotly_code=vn_results["plotly_code"],
                plotly_figure=vn_results["plotly_figure"],
                # chats=[],
            )
            return create_response(
                data=IChatAppResponse(chats=response.dict()), message="Query Success"
            )
        except Exception as e:
            logger.error(f"Error during SQL query: {str(e)}")
            return JSONResponse(
                content={"Error during SQL query": str(e)}, status_code=500
            )

    elif chatapp_obj.chat_app_type == "unstructured_chat_app":
        try:
            contexts_found = await get_unstructured_context_v2(
                request=request,
                request_data=request_data,
                chatapp_obj=chatapp_obj,
            )
            aggregated_file_contexts = await aggregate_file_contexts_v2(
                contexts_found.rag_contexts, contexts_found.raw_results
            )

            # Retrieve chat histories
            chat_histories = (
                await chathistory_crud.get_chat_history_by_session_id_no_pagination(
                    chat_session_id=chat_session_id
                )
            )

            # Generate LLM response
            rag_gen_settings = chatapp_obj.generation_config
            data = await generate_llm_response(
                query=request_data.query,
                contexts_found=contexts_found,
                system_prompt=chatapp_obj.system_prompt,
                chat_histories=chat_histories,
                gen_settings=rag_gen_settings,
            )
            data.file_contexts_aggregated = aggregated_file_contexts
            await check_title(
                query=request_data.query,
                chat_histories=chat_histories,
                chat_session_id=chat_session_id,
                chatapp_id=chatapp_id,
            )

            # Calculate total tokens
            total_tokens = data.input_tokens + data.output_tokens

            # Store chat history
            chat_history_record = IChatHistoryLineCreate(
                user_query=request_data.query,
                contexts=[
                    context.chunk_id
                    for context in contexts_found.rag_contexts
                    if context.chunk_id
                ],
                llm_response=data.answer,
                llm_model=rag_gen_settings.llm_model,
                input_tokens=data.input_tokens,
                output_tokens=data.output_tokens,
                total_tokens=total_tokens,
            )

            await chathistory_crud.add_chat_session_history(
                chatapp_id=chatapp_id,
                chat_session_id=chat_session_id,
                obj_in=IChatHistoryCreate(histories=[chat_history_record]),
            )

            # Build response
            response_data = IChatAppResponse(
                chats=data,
            )
            response_data.chats.contexts_found = None

            return create_response(data=response_data, message="Query Success")

        except Exception as e:
            logger.error(f"Error during RAG generation: {str(e)}")
            raise HTTPException(status_code=500, detail="Error During RAG generation")
