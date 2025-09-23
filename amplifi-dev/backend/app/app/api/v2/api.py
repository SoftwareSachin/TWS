from fastapi import APIRouter

from app.api.v1.endpoints import (
    agent,
    api_client,
    audit_log,
)
from app.api.v1.endpoints import chatsession as chatsession_v1
from app.api.v1.endpoints import (
    chunking_config,
)
from app.api.v1.endpoints import dataset as dataset_v1
from app.api.v1.endpoints import (
    destination,
    documents,
    download_file,
    entities_extractor,
    eval,
    group,
    login,
    logout,
    organization,
    passwordReset,
    platform,
    refresh_token,
    role,
    source_connector,
    token_counts,
    tool,
    user,
    workflow,
    workspace,
    workspace_tool,
)
from app.api.v2.endpoints import (
    chat,
    chatapp,
)
from app.api.v2.endpoints import chatsession as chatsession_v2
from app.api.v2.endpoints import dataset as dataset_v2
from app.api.v2.endpoints import (
    graph,
    ingest_file,
    search,
    train,
)

api_router = APIRouter()
api_router.include_router(login.router, prefix="/login", tags=["login"])
api_router.include_router(role.router, prefix="/role", tags=["role"])
api_router.include_router(user.router, prefix="/user", tags=["user"])
api_router.include_router(group.router, prefix="/group", tags=["group"])
api_router.include_router(chunking_config.router, tags=["chunking_config"])
api_router.include_router(workspace.router, tags=["workspace"])
api_router.include_router(organization.router, tags=["organization"])
# api_router.include_router(embeddingConfig.router, tags=["embeddingconfig"])
api_router.include_router(workflow.router, tags=["workflow"])
api_router.include_router(refresh_token.router, tags=["refresh_token"])
api_router.include_router(passwordReset.router, tags=["passwordReset"])
# api_router.include_router(team.router, prefix="/team", tags=["team"])
# api_router.include_router(cache.router, prefix="/cache", tags=["cache"])
# api_router.include_router(report.router, prefix="/report", tags=["report"])
# api_router.include_router(
#     natural_language.router, prefix="/natural_language", tags=["natural_language"]
# )
# api_router.include_router(
#     periodic_tasks.router, prefix="/periodic_tasks", tags=["periodic_tasks"]
# )
api_router.include_router(dataset_v1.router, tags=["dataset"])
api_router.include_router(dataset_v2.router, tags=["dataset"])
api_router.include_router(source_connector.router, tags=["source_connector"])
api_router.include_router(ingest_file.router, tags=["ingest_files"])
api_router.include_router(destination.router, tags=["destination"])
api_router.include_router(audit_log.router, tags=["audit_logs"])
api_router.include_router(eval.router, tags=["eval"])
api_router.include_router(search.router, tags=["search"])
api_router.include_router(platform.router, tags=["platform"])
api_router.include_router(logout.router, tags=["logout"])
api_router.include_router(chat.router, tags=["bot"])
api_router.include_router(chatapp.router, tags=["chat"])
api_router.include_router(chatsession_v1.router, tags=["chat"])
api_router.include_router(chatsession_v2.router, tags=["chat"])
api_router.include_router(token_counts.router, tags=["Observability"])
api_router.include_router(download_file.router, tags=["Download File"])
api_router.include_router(documents.router, tags=["documents"])
api_router.include_router(agent.router, tags=["agent"])
api_router.include_router(workspace_tool.router, tags=["Tool"])
api_router.include_router(tool.router, tags=["CreateTool"])
api_router.include_router(entities_extractor.router, tags=["EntitiesExtractor"])
api_router.include_router(api_client.router, tags=["API Clients"])
api_router.include_router(graph.router, tags=["graph"])
api_router.include_router(train.router, tags=["train"])
