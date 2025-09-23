from datetime import datetime
from pathlib import Path
from typing import Dict, List
from uuid import UUID, uuid4

import aiofiles
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.deps import get_current_user, user_workspace_access_check
from app.be_core.celery import celery
from app.be_core.config import settings

# from phoenix.evals import llm_generate, OpenAIModel, RelevanceEvaluator
from app.be_core.logger import logger
from app.models.user_model import User
from app.schemas.eval_schema import (
    DatasetEvalResults,
    EvalProcessedQuestionsResponse,
    IWorkspaceEvalRequest,
    IWorkspaceEvalResponse,
    RagProcessedRequest,
    RagRequest,
)
from app.schemas.response_schema import IPostResponseBase, create_response
from app.schemas.role_schema import IRoleEnum
from app.utils.azure_fns.get_scores import get_precision
from app.utils.csv_parser import find_csv_column

router = APIRouter()


def _dataset_batch_helper(request: RagProcessedRequest, qa_pairs: List[Dict[str, str]]):
    search_limit = request.vector_search_settings.search_limit
    search_index_type = request.vector_search_settings.search_index_type
    probes = request.vector_search_settings.probes
    ef_search = request.vector_search_settings.ef_search

    search_batch_task = celery.signature(
        "tasks.dataset_search_batch_task",
        kwargs={
            "qa_pairs": qa_pairs,
            "dataset_ids": request.dataset_ids,
            "search_limit": search_limit,
            "search_index_type": search_index_type,
            "probes": probes,
            "ef_search": ef_search,
        },
    ).apply_async()

    return search_batch_task.id


@router.post("/workspace/{workspace_id}/eval/upload_question_answer_pairs")
async def upload_question_answer_pairs(
    workspace_id: UUID,
    qa_file: UploadFile = File(
        description="Please upload a CSV file with 'questions' and 'answers' column"
    ),
    current_user: User = Depends(
        get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(user_workspace_access_check),
):

    try:
        logger.info(f"Uploading questions CSV for workspace_id={workspace_id}")

        if not qa_file.filename.endswith(".csv"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file type. Please upload a .csv file.",
            )

        file_uuid = str(uuid4())
        stored_qa_filename = f"{file_uuid}_{qa_file.filename}"
        file_path = Path(settings.EVAL_QUESTION_ANSWER_FILES_PATH) / stored_qa_filename

        async with aiofiles.open(file_path, "wb") as f:
            while True:
                content = await qa_file.read(1024 * 1024)
                if not content:
                    break
                await f.write(content)

        logger.info(
            f"Successfully Uploaded File to: {file_path} for workspace: {workspace_id}"
        )

        return create_response(
            message="uploaded question-answer file for evaluation successfully",
            data={"eval_file_id": file_uuid},
        )

    except Exception as e:
        logger.error(f"Upload Fialed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload the file",
        )


acceptable_questions_columns = {"question", "questions", "Question", "Questions"}
acceptable_answers_columns = {"answer", "answers", "Answer", "Answers"}


@router.post("/workspace/{workspace_id}/eval/process_questions")
async def process_questions(
    workspace_id: UUID,
    request: RagRequest,
    current_user: User = Depends(
        get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(user_workspace_access_check),
) -> IPostResponseBase[EvalProcessedQuestionsResponse]:

    logger.info(
        f"Processing questions for workspace_id={workspace_id} with eval_file_id={request.eval_file_id}"
    )

    eval_dir = Path(settings.EVAL_QUESTION_ANSWER_FILES_PATH)
    matched_files = list(eval_dir.glob(f"{request.eval_file_id}_*.csv"))

    if not matched_files:
        logger.error(f"No file exists with file id {request.eval_file_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to find the file with the given file id",
        )

    qa_file_path = matched_files[0]

    try:
        df = pd.read_csv(qa_file_path)
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to find the file with the given file id",
        )

    question_col = find_csv_column(df.columns, acceptable_questions_columns)
    answer_col = find_csv_column(df.columns, acceptable_answers_columns)

    if not question_col or not answer_col:
        logger.error("CSV does not contain question or answer column")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV does not contain question or answer column",
        )

    questions = df[question_col].dropna().tolist()
    answers = df[answer_col].dropna().tolist()

    if len(questions) != len(answers):
        logger.error("Number of questions and answers in CSV do not match")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Number of questions and answers in CSV do not match",
        )

    qa_pairs = [{q: a} for q, a in zip(questions, answers)]

    eval_id = _dataset_batch_helper(request, qa_pairs)

    response = EvalProcessedQuestionsResponse(
        dataset_ids=request.dataset_ids,
        workspace_id=workspace_id,
        eval_task_id=eval_id,
        started_at=(datetime.utcnow()).isoformat(),
    )

    return create_response(
        message="Started RAG Retrieval and evaluation. Use task ID to check status.",
        data=response,
    )


@router.get("/workspace/{workspace_id}/eval/retrieve_results")
async def retrieve_results(
    workspace_id: UUID,
    eval_task_id: str,
    current_user: User = Depends(
        get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(user_workspace_access_check),
) -> IPostResponseBase[DatasetEvalResults]:

    logger.info(
        f"Finalizing evaluation for workspace_id={workspace_id}, task_id={eval_task_id}"
    )

    task_result = celery.AsyncResult(eval_task_id, app=celery)

    if not task_result.ready():
        return create_response(
            message="Task still in progress",
            data=None,
        )

    if not task_result.result:
        return create_response(
            message="Task completed but returned no results.",
            meta={"workspace_id": workspace_id, "task_id": eval_task_id},
            data={},
        )

    response = task_result.result

    logger.debug(
        f"results retrieved from celery task for rag eval successfully with {response}"
    )

    return create_response(
        message="Finished relevance scoring.",
        data=response,
    )


@router.post("/workspace/{workspace_id}/eval")
async def eval_result(
    workspace_id: UUID,
    request: IWorkspaceEvalRequest,
    current_user: User = Depends(
        get_current_user(required_roles=[IRoleEnum.admin, IRoleEnum.developer])
    ),
    _=Depends(user_workspace_access_check),
) -> IPostResponseBase[IWorkspaceEvalResponse]:
    precision, ndcg_score, in_tokens = get_precision(
        query=request.query,
        contexts=[result.text for result in request.contexts],
        doc_scores=[result.search_score for result in request.contexts],
        k=len(request.contexts),
    )
    response = IWorkspaceEvalResponse(
        query=request.query,
        contexts=request.contexts,
        precision=precision,
        ndcg_score=ndcg_score,
        in_tokens=in_tokens,
    )
    return create_response(
        message="Eval completed successfully",
        data=response,
    )
