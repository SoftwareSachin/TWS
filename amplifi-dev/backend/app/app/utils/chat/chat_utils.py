import json
from collections import defaultdict
from itertools import chain
from uuid import UUID

from app.be_core.logger import logger
from app.constants.chat_constants import TOOL_TYPE_MCP, TOOL_TYPE_SYSTEM
from app.crud import chathistory
from app.crud.agent_crud import agent
from app.crud.chat_app_crud import chatapp
from app.crud.chat_session_crud import chatsession
from app.crud.tool_crud import tool_crud
from app.crud.workspace_tool_crud import workspace_tool_crud
from app.models import WorkspaceTool
from app.schemas.chat_schema import (
    ChatRequest,
    IChatHistoryCreate,
    IChatHistoryLineCreate,
    IChatSessionCreate,
)
from app.schemas.rag_generation_schema import ChatModelEnum
from app.schemas.workspace_tool_schema import ISystemToolRead
from app.utils.chat.dynamic_agent import DynamicToolAgent
from app.utils.llm_fns.azure_openai import query_openai_chat_history
from app.utils.llm_fns.lang_detect import construct_ssml_multilingual_with_gpt

chatAppCrud = chatapp
agentCrud = agent
toolCrud = tool_crud
workspaceToolCrud = workspace_tool_crud
chatHistoryCrud = chathistory
chatSessionCrud = chatsession


def filter_failed_tool_calls(contexts):
    """
    Filter out failed tool calls from contexts array.

    Failed tool calls are identified by:
    - Having an "error" field in the content
    - Having "tool_return" type with error content
    - Having content.result.error (nested error structure)
    """
    if not contexts:
        return contexts

    filtered_contexts = []
    for context in contexts:
        # Check if this is a failed tool call
        is_failed_tool_call = False

        if isinstance(context, dict):
            # Check for direct error field
            if "error" in context:
                is_failed_tool_call = True

            # Check for tool_return type with error content
            elif context.get("part_kind") == "tool-return":
                content = context.get("content", {})
                if isinstance(content, dict):
                    # Check for direct error in content
                    if "error" in content:
                        is_failed_tool_call = True
                    # Check for nested error in content.result.error
                    elif (
                        "result" in content
                        and isinstance(content["result"], dict)
                        and "error" in content["result"]
                    ):
                        is_failed_tool_call = True

        # Only add to filtered contexts if it's not a failed tool call
        if not is_failed_tool_call:
            filtered_contexts.append(context)

    return filtered_contexts


async def get_title(query: str) -> str:
    """Generate a title for chat session based on user query."""
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


async def check_and_update_title(
    query: str,
    chat_histories: list,
    chat_session_id: UUID,
    chatapp_id: UUID,
) -> str | None:
    """Check if chat session title needs updating and update if needed."""
    try:

        if not chat_histories:

            chatsession_record = await chatSessionCrud.get_chat_session_by_id(
                chat_session_id=chat_session_id,
                chatapp_id=chatapp_id,
            )

            # Check if title is still a default/auto-generated title
            title_lower = (
                chatsession_record.title.lower() if chatsession_record.title else ""
            )
            needs_title_update = (
                not chatsession_record.title
                or chatsession_record.title.strip() == ""
                or title_lower.startswith("untitled chat")
                or title_lower.startswith("new chat")
                or title_lower.startswith("chat session")
                or title_lower in ["new chat", "untitled", "chat session"]
            )

            if needs_title_update:

                ai_title = await get_title(query=query)

                await chatSessionCrud.update_chat_session(
                    chat_session_id=chat_session_id,
                    chatapp_id=chatapp_id,
                    obj_in=IChatSessionCreate(title=ai_title),
                )

                return ai_title  # Return the new title
            else:

                return None
        else:

            return None

    except Exception:

        return None


async def getChatResponse(chatRequest: ChatRequest, organization_id: UUID):
    try:
        logger.debug(
            "===============================NEW CALL==========================================="
        )
        chatApp = await chatAppCrud.get_chatapp_v2_by_id(
            chatapp_id=chatRequest.chat_app_id
        )

        agent_data = await agentCrud.get_agent_by_id_in_workspace(
            workspace_id=chatApp.workspace_id, agent_id=chatApp.agent_id
        )

        # Use workspace_tool_ids directly from agent
        workspace_tools = await workspaceToolCrud.get_workspace_tools_by_ids(
            workspace_id=chatApp.workspace_id,
            workspace_tool_ids=agent_data.workspace_tool_ids,
        )

        # Get all tool IDs from workspace tools
        tool_ids = list(
            set(chain.from_iterable(item.tool_ids for item in workspace_tools))
        )

        # Get tools to check their tool_type
        tools = await toolCrud.get_tools_by_ids(
            tool_ids=tool_ids, current_user_org_id=organization_id
        )

        # Determine tool type based on tool_type field
        grouped_tools = group_tools_by_type(tools)

        chat_history = (
            await chatHistoryCrud.get_chat_history_by_session_id_no_pagination(
                chat_session_id=chatRequest.chat_session_id
            )
        )

        # Check and update chat session title if this is the first query
        updated_session_title = await check_and_update_title(
            query=chatRequest.query,
            chat_histories=chat_history,
            chat_session_id=chatRequest.chat_session_id,
            chatapp_id=chatRequest.chat_app_id,
        )
        agent = None
        context = {"query": chatRequest.query}

        if TOOL_TYPE_MCP in grouped_tools and grouped_tools[TOOL_TYPE_MCP]:
            agent = await _handle_mcp_agent(
                chatApp,
                agent_data,
                grouped_tools[TOOL_TYPE_MCP],
            )
            context["agent_id"] = agent_data.id
            # await agent.start_mcp_servers()
            logger.info("Started mcp server")
        if TOOL_TYPE_SYSTEM in grouped_tools and grouped_tools[TOOL_TYPE_SYSTEM]:
            agent = await _handle_system_tool_agent(
                agent,
                agent_data,
                grouped_tools[TOOL_TYPE_SYSTEM],
            )
            context["dataset_ids"] = flatten_dataset_ids(workspace_tools)
            context["tools_map"] = map_tools_to_datasets(tools, workspace_tools)

            # Debug logging to trace tools_map
            logger.info(
                f"CONTEXT_DEBUG: tools_map keys: {list(context['tools_map'].keys()) if context.get('tools_map') else 'None'}"
            )
            logger.info(f"CONTEXT_DEBUG: tools_map content: {context['tools_map']}")
            logger.info(f"CONTEXT_DEBUG: context keys: {list(context.keys())}")

        agent.convert_to_pydantic_history(chat_history)

        chat_response = await agent.chat(
            chatRequest.query,
            context=context,
            chat_session_id=str(chatRequest.chat_session_id),
        )
        logger.info(f"Chat response final {str(chat_response)}")

        # Filter out failed tool calls from contexts before saving to history and returning to frontend
        filtered_contexts = filter_failed_tool_calls(chat_response["contexts"])

        history_obj = IChatHistoryCreate(
            histories=[
                IChatHistoryLineCreate(
                    user_query=chatRequest.query,
                    contexts=filtered_contexts,
                    llm_response=json.dumps(
                        {"responses": chat_response["responses"]},
                        default=str,
                        ensure_ascii=False,
                    ),
                    llm_model=agent_data.llm_model,
                    input_tokens=12,
                    output_tokens=8,
                    total_tokens=20,
                    pydantic_message=chat_response["pydantic_messages"],
                )
            ]
        )
        await chatHistoryCrud.add_chat_session_history(
            chatapp_id=chatRequest.chat_app_id,
            chat_session_id=chatRequest.chat_session_id,
            obj_in=history_obj,
        )

        response = {
            "responses": chat_response["responses"],
            "pydantic_message": chat_response["pydantic_messages"],
            "contexts": filtered_contexts,
            "ssml": await construct_ssml_multilingual_with_gpt(
                "; ".join(item["response"] for item in chat_response["responses"])
            ),
        }

        # Include updated session title if it was changed
        if updated_session_title:
            response["updated_session_title"] = updated_session_title

        return response
    except Exception as e:
        logger.error(f"Error occurred while processing chat request: {str(e)}")
        return {"error": str(e)}


def group_tools_by_type(tools) -> dict:
    """Determine if tools are MCP or system tools based on tool_type field"""
    grouped_tools = defaultdict(list)

    if not tools:
        grouped_tools["system"] = []
        return dict(grouped_tools)

    for tool in tools:
        # Default to 'system' if tool_type is not present
        tool_type_str = "system"

        if hasattr(tool, "tool_type"):
            tool_type_value = tool.tool_type
            if hasattr(tool_type_value, "value"):
                tool_type_str = tool_type_value.value
            else:
                tool_type_str = str(tool_type_value)

        grouped_tools[tool_type_str].append(tool)

    return dict(grouped_tools)


async def _handle_system_tool_agent(agent, agent_data, tools):
    """Handle agents that use system tools (existing behavior)"""
    logger.info(f"Handling system tool agent: {agent_data.id} for chat app.")

    # Create system tool agent with agent configuration
    agent_config = {
        "system_prompt": getattr(agent_data, "system_prompt", None),
        "temperature": getattr(agent_data, "temperature", 0.7),
        "prompt_instructions": getattr(agent_data, "prompt_instructions", None),
    }

    logger.info(f"Agent configured with temperature: {agent_config['temperature']}")

    if agent is None:
        agent = DynamicToolAgent(
            llm_model=getattr(agent_data, "llm_model", "GPT4o"),
            agent_config=agent_config,
        )
    else:
        agent.agent_config["system_prompt"] = agent_config.pop("system_prompt", None)
    logger.info(f"System Tool Agent initialized {str(agent)}")

    agent.register_multiple_tools(tools)
    return agent


async def _handle_mcp_agent(chat_app, agent_data, tools):
    """Handle agents that use MCP servers"""
    logger.info(f"Handling MCP agent: {agent_data.id} for chat app {chat_app.id}")

    # Extract MCP server configurations from MCP tools
    mcp_server_configs = []
    for tool in tools:
        # Check for tool_type instead of tool_kind
        if hasattr(tool, "tool_type"):
            tool_type_value = tool.tool_type
            if hasattr(tool_type_value, "value"):
                tool_type_str = tool_type_value.value
            else:
                tool_type_str = str(tool_type_value)

            if tool_type_str == "mcp":
                # For schema objects (IMCPToolRead), access mcp_server_config directly
                if hasattr(tool, "mcp_server_config") and tool.mcp_server_config:
                    mcp_config = tool.mcp_server_config
                    mcp_server_configs.append(mcp_config)
                    logger.info(f"Added MCP server config from tool: {tool.name}")
                # For database models, access through mcp_tool relationship
                elif hasattr(tool, "mcp_tool") and tool.mcp_tool:
                    mcp_config = tool.mcp_tool.mcp_server_config
                    if mcp_config:
                        mcp_server_configs.append(mcp_config)
                        logger.info(f"Added MCP server config from tool: {tool.name}")
                    else:
                        logger.warning(f"MCP tool {tool.name} has no server config")
                else:
                    logger.warning(f"MCP tool {tool.name} missing server configuration")

    logger.info(f"Total MCP server configs found: {len(mcp_server_configs)}")

    if not mcp_server_configs:
        # Provide more detailed error information
        mcp_tools_without_config = [
            tool.name
            for tool in tools
            if hasattr(tool, "tool_type")
            and str(getattr(tool.tool_type, "value", tool.tool_type)) == "mcp"
        ]
        logger.warning(
            f"No MCP server configurations found for tools: {mcp_tools_without_config}"
        )
        error_msg = "MCP agent missing server configuration in tools. "
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Create agent configuration
    agent_config = {
        "system_prompt": getattr(agent_data, "system_prompt", None),
        "temperature": getattr(agent_data, "temperature", 0.7),
        "prompt_instructions": getattr(agent_data, "prompt_instructions", None),
    }

    # Create MCP agent with multiple server configs

    return DynamicToolAgent(
        llm_model=getattr(agent_data, "llm_model", "GPT4o"),
        agent_config=agent_config,
        mcp_server_configs=mcp_server_configs,
    )


def flatten_dataset_ids(workspace_tools: list[WorkspaceTool]) -> list[UUID]:
    return [
        dataset_id
        for tool in workspace_tools
        if tool.dataset_ids
        for dataset_id in tool.dataset_ids
    ]


def map_tools_to_datasets(
    system_tools: list[ISystemToolRead], workspace_tools: list[WorkspaceTool]
) -> dict[UUID, dict[str, object]]:
    """
    Maps each tool_id to its associated dataset_ids, dataset_names, and tool_name.

    Returns:
        Dict[UUID, {
            'dataset_ids': List[UUID],
            'dataset_names': Dict[str, str],  # Maps UUID -> Name
            'tool_name': str
        }]
    """
    from sqlalchemy import select

    from app.db.session import SyncSessionLocal
    from app.models.dataset_model import Dataset

    # Step 1: Map tool_id to tool name from system_tools
    tool_name_map = {tool.tool_id: tool.name for tool in system_tools}

    # Step 2: Collect dataset_ids for each tool_id from workspace_tools
    tool_map = defaultdict(
        lambda: {"dataset_ids": set(), "dataset_names": {}, "tool_name": ""}
    )

    for ws_tool in workspace_tools:
        for tool_id in ws_tool.tool_ids:
            tool_map[tool_id]["dataset_ids"].update(ws_tool.dataset_ids)
            tool_map[tool_id]["tool_name"] = tool_name_map.get(tool_id, "Unknown")

    #############################################
    # Step 3: Fetch dataset names for all collected dataset IDs
    all_dataset_ids = set()
    for data in tool_map.values():
        all_dataset_ids.update(data["dataset_ids"])

    dataset_name_map = {}
    if all_dataset_ids:
        try:
            with SyncSessionLocal() as session:
                query = select(Dataset).where(
                    Dataset.id.in_(all_dataset_ids), Dataset.deleted_at.is_(None)
                )
                result = session.execute(query)
                datasets = result.scalars().all()
                dataset_name_map = {
                    str(dataset.id): dataset.name for dataset in datasets
                }
        except Exception as e:
            # Log error but continue with empty dataset names
            from app.be_core.logger import logger

            logger.error(f"Error fetching dataset names: {str(e)}")

    # Step 4: Add dataset names to each tool's mapping
    for data in tool_map.values():
        for dataset_id in data["dataset_ids"]:
            dataset_id_str = str(dataset_id)
            data["dataset_names"][dataset_id_str] = dataset_name_map.get(
                dataset_id_str, f"Unknown-{dataset_id_str[:8]}"
            )
    ####################################################

    # Step 5: Convert sets to lists for serialization
    return {
        tool_id: {
            "dataset_ids": list(data["dataset_ids"]),
            "dataset_names": data["dataset_names"],
            "tool_name": data["tool_name"],
        }
        for tool_id, data in tool_map.items()
    }
