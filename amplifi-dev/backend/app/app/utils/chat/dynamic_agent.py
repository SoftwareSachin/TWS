import ast
import asyncio
import inspect
from importlib import import_module
from pydoc import locate
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, parse_obj_as
from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_core import to_jsonable_python

from app.api.deps import get_async_gpt4o_client, get_llm_model_name, get_model_client
from app.be_core.logger import logger
from app.models import ChatHistory
from app.schemas.rag_generation_schema import ChatModelEnum
from app.schemas.workspace_tool_schema import ToolReadUnion
from app.utils.agent_prompts import AgentPrompts
from app.utils.agent_utils import DatasetNameConverter
from app.utils.logfire_config import logfire_manager
from app.utils.mcp_server_manager import MCPServerManager

client_4o = get_async_gpt4o_client()


class DynamicToolAgent:
    def __init__(
        self,
        llm_model: ChatModelEnum,
        agent_config: Optional[dict] = None,
        mcp_server_configs: Optional[List[dict]] = None,
    ):
        model = OpenAIModel(
            get_llm_model_name(llm_model),
            provider=OpenAIProvider(openai_client=get_model_client(llm_model)),
        )

        self.agent_config = agent_config or {}
        self.mcp_servers = []
        self._mcp_server_tasks = []
        # self._mcp_context_active = False
        self.message_history = []
        self.chat_session_id = None  # Add this line
        self.dataset_converter = None  # Will be initialized when context is available
        self.base_prompt = AgentPrompts.get_agent_base_system_prompt()
        self.base_prompt = f"{self.base_prompt}\n\n{AgentPrompts.get_response_formatting_requirements()}\n\n{AgentPrompts.get_response_handling_requirements()}"

        # Append custom system prompt if provided
        if "system_prompt" in agent_config and agent_config["system_prompt"]:
            self.base_prompt = f"{self.base_prompt}\n\nCUSTOM SYSTEM INSTRUCTIONS:\n{agent_config['system_prompt']}"
            logger.info("Added custom system prompt to agent")

        # Append user instructions if provided
        if (
            "prompt_instructions" in agent_config
            and agent_config["prompt_instructions"]
        ):
            self.base_prompt = f"{self.base_prompt}\n\nUSER INSTRUCTIONS:\n{agent_config['prompt_instructions']}"
            logger.info("Added user prompt instructions to agent")

        if mcp_server_configs is not None and len(mcp_server_configs) > 0:
            logger.info("Initializing MCP servers")
            self.mcp_manager = MCPServerManager()

            self.agent = Agent(
                model,
                system_prompt=self.base_prompt,
                mcp_servers=self._create_mcp_servers(mcp_server_configs),
            )
        else:
            # Create agent with system tools (existing behavior)
            self.agent = Agent(
                model,
                system_prompt=self.base_prompt,
            )

        # Instrument the agent with logfire using the manager
        logfire_manager.instrument_pydantic_ai(self.agent)

        # Verify that the dynamic system prompt was registered
        logger.info(
            "DEBUG: Dynamic system prompt decorator has been registered with PydanticAI agent"
        )

        self.registered_tools = {}

    def _create_mcp_servers(
        self, mcp_server_configs: List[dict]
    ) -> List[MCPServerStdio]:
        """Create MCP servers from configurations"""
        servers = []
        for config in mcp_server_configs:
            try:
                # Extract the actual MCP server config from the wrapper
                actual_config = config.get("mcp_server_config", config)
                server = self.mcp_manager.create_server(actual_config)
                servers.append(server)
                self.mcp_servers.append(server)
                logger.info(
                    f"Created MCP server from config: {list(actual_config.keys())}"
                )
            except Exception as e:
                logger.error(f"Failed to create MCP server: {e}")
        return servers

    def convert_to_pydantic_history(self, chat_history: List[ChatHistory]):
        """Convert chat history from database to pydantic format for agent context."""
        chat_history_messages: List[Dict[str, Any]] = []

        # Handle empty or None chat history
        if not chat_history:
            logger.info("No chat history provided, starting with empty message history")
            self.message_history = []
            return

        for record in chat_history:
            # Skip records without pydantic_message
            if not record.pydantic_message:
                logger.debug(
                    f"Skipping chat history record without pydantic_message: {record.id}"
                )
                continue

            for msg in record.pydantic_message:
                try:
                    # Validate the message structure before accessing keys
                    if not isinstance(msg, dict):
                        logger.warning(f"Skipping non-dict message: {type(msg)}")
                        continue

                    # Check if required keys exist and are not None
                    user_prompt = msg.get("user_prompt")
                    model_response = msg.get("model_response")

                    if not user_prompt:
                        logger.warning(
                            f"Skipping message with missing/null user_prompt: {msg}"
                        )
                        continue

                    if not model_response or not isinstance(model_response, dict):
                        logger.warning(
                            f"Skipping message with missing/invalid model_response: {msg}"
                        )
                        continue

                    content = model_response.get("content")
                    if not content:
                        logger.warning(
                            f"Skipping message with missing/null content in model_response: {msg}"
                        )
                        continue

                    # Flatten: extend the list instead of appending nested list
                    validated = ModelMessagesTypeAdapter.validate_python(
                        [
                            ModelRequest(parts=[UserPromptPart(content=user_prompt)]),
                            ModelResponse(parts=[TextPart(content=content)]),
                        ]
                    )
                    chat_history_messages.extend(validated)
                    logger.debug("Successfully converted message to pydantic format")

                except Exception as e:
                    logger.error(
                        f"Error converting pydantic message to history: {e}. Message: {msg}"
                    )
                    continue

        logger.info(f"Converted {len(chat_history_messages)} chat history messages")
        self.message_history = self.keep_recent_messages(messages=chat_history_messages)

    def keep_recent_messages(self, messages: list[any]) -> list[ModelMessage]:
        """Keep only the last 5 messages to manage token usage."""
        return messages[-5:] if len(messages) > 5 else messages

    def _filter_history_for_tool_selection(
        self, message_history: List[ModelMessage], current_query: str
    ) -> List[ModelMessage]:
        """Filter history to remove tool usage bias while keeping relevant context"""

        filtered_history = []

        for msg in message_history:
            # Keep user queries and basic responses, but remove tool calls
            if isinstance(msg, ModelRequest):
                # Keep user prompts for context
                filtered_history.append(msg)
            elif isinstance(msg, ModelResponse):
                # Filter out tool-related parts, keep only text responses
                text_parts = []
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        text_parts.append(part)
                    # Skip ToolCallPart and ToolReturnPart

                if text_parts:
                    # Create new response with only text parts
                    filtered_msg = ModelResponse(
                        parts=text_parts,
                        model_name=msg.model_name,
                        timestamp=msg.timestamp,
                        kind=msg.kind,
                    )
                    filtered_history.append(filtered_msg)

        # Limit to last few exchanges to prevent overwhelming context
        return filtered_history[-6:] if len(filtered_history) > 6 else filtered_history

    def _create_unbiased_system_prompt(
        self, query: str, tools_map: dict, query_intent: str
    ) -> str:
        """Create system prompt that emphasizes current query over history"""

        base_prompt = AgentPrompts.generate_system_prompt_with_datasets(
            tools_map, query_intent
        )

        unbiased_instructions = f"""
🎯 CURRENT QUERY FOCUS:
The user is asking: "{query}"

IMPORTANT INSTRUCTIONS:
1. Base your tool selection on the CURRENT query requirements, not previous patterns
2. Even if you used different tools before, choose the RIGHT tool for THIS query
3. Query intent classification: {query_intent}

CRITICAL: Ignore any previous tool usage patterns in conversation history.
Focus ONLY on what the current query is asking for.

{base_prompt}
"""

        return unbiased_instructions

    async def get_available_mcp_tools(self) -> List[dict]:
        """Get available tools from MCP servers"""
        all_tools = []

        # Start MCP servers temporarily to get tools
        async with self.agent.run_mcp_servers():
            for server in self.mcp_servers:
                try:
                    await asyncio.sleep(1)  # Give server time to initialize
                    raw_tools = await server.list_tools()
                    tools = [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters_schema": tool.inputSchema,
                        }
                        for tool in raw_tools
                    ]
                    all_tools.extend(tools)
                    logger.info(f"Found {len(tools)} tools in MCP server")
                except Exception as e:
                    logger.error(f"Error listing MCP server tools: {e}")

        return all_tools

    async def execute_tool_async(
        self,
        tool_config: ToolReadUnion,
        input_dict: Dict[str, Any],
        context: Optional[RunContext] = None,
    ) -> Any:
        """Execute a system tool asynchronously."""
        try:
            # Dynamically load schema and function
            logger.info("Into execute tool")
            mod = import_module(tool_config.python_module)
            logger.info(f"FUNCTION: {getattr(mod, tool_config.function_name)}")
            func = getattr(mod, tool_config.function_name)
            logger.info(f"FUNC SIG: {inspect.signature(func)}")
            input_schema_cls = locate(tool_config.input_schema)
            output_schema_cls = locate(tool_config.output_schema)
            logger.info(f"FUNC SIG: {input_schema_cls}")

            # Validate input
            logger.debug(
                f"[DEBUG] Executing {tool_config.name} with raw input_dict: {input_dict}"
            )
            validated_input = parse_obj_as(input_schema_cls, input_dict)
            # Run the tool
            logger.info(f"Input after validation {str(validated_input)}")
            if asyncio.iscoroutinefunction(func):
                result = await func(validated_input)
            else:
                result = func(validated_input)

            # Ensure output matches the schema
            return parse_obj_as(output_schema_cls, result)

        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}

    def _ensure_dataset_converter_initialized(self):
        """
        Initialize the dataset converter if it hasn't been initialized yet.
        This should be called whenever we need to use dataset name conversion.
        """
        if (
            self.dataset_converter is None
            and hasattr(self.agent, "context")
            and self.agent.context
        ):
            tools_map = self.agent.context.get("tools_map", {})
            if tools_map:
                self.dataset_converter = DatasetNameConverter(tools_map)
                logger.info("Dataset name converter initialized with tools map")

                # Log available conversions for debugging
                conversion_info = self.dataset_converter.get_conversion_info()
                logger.debug(
                    f"Available dataset name conversions: {conversion_info['name_to_uuid_mappings']}"
                )
            else:
                logger.warning(
                    "No tools_map available in context - dataset converter not initialized"
                )

    def _is_valid_dataset_id(self, dataset_id_str: str) -> bool:
        """
        Validate if a given ID is a valid dataset ID by checking against the agent's context.

        Args:
            dataset_id_str: String representation of the dataset ID

        Returns:
            True if valid dataset ID, False otherwise
        """
        # Get dataset IDs from agent context
        if hasattr(self.agent, "context") and self.agent.context:
            # Check in dataset_ids list
            context_dataset_ids = self.agent.context.get("dataset_ids", [])
            if dataset_id_str in [str(ds_id) for ds_id in context_dataset_ids]:
                return True

            # Check in tools_map for dataset mappings
            tools_map = self.agent.context.get("tools_map", {})
            for tool_info in tools_map.values():
                if isinstance(tool_info, dict):
                    tool_dataset_ids = tool_info.get("dataset_ids", [])
                    if dataset_id_str in [str(ds_id) for ds_id in tool_dataset_ids]:
                        return True

        return False

    def _get_valid_dataset_ids_from_context(self) -> List[str]:
        """
        Extract all valid dataset IDs from the agent's context.

        Returns:
            List of valid dataset ID strings
        """
        valid_dataset_ids = []

        if hasattr(self.agent, "context") and self.agent.context:
            # Get from direct dataset_ids context
            context_dataset_ids = self.agent.context.get("dataset_ids", [])
            valid_dataset_ids.extend([str(ds_id) for ds_id in context_dataset_ids])

            # Get from tools_map
            tools_map = self.agent.context.get("tools_map", {})
            for tool_info in tools_map.values():
                if isinstance(tool_info, dict):
                    tool_dataset_ids = tool_info.get("dataset_ids", [])
                    valid_dataset_ids.extend([str(ds_id) for ds_id in tool_dataset_ids])

        # Remove duplicates while preserving order
        seen = set()
        unique_ids = []
        for ds_id in valid_dataset_ids:
            if ds_id not in seen:
                seen.add(ds_id)
                unique_ids.append(ds_id)

        return unique_ids

    def _tool_uses_dataset_ids(self, tool_config: ToolReadUnion) -> bool:
        """
        Check if a tool uses dataset_ids parameter by examining its input schema.

        Args:
            tool_config: Tool configuration containing schema information

        Returns:
            True if tool uses dataset_ids, False otherwise
        """
        try:
            input_schema_cls = locate(tool_config.input_schema)
            if input_schema_cls and hasattr(input_schema_cls, "__fields__"):
                # Check if dataset_ids field exists in the schema
                return "dataset_ids" in input_schema_cls.__fields__
            return False
        except Exception as e:
            logger.debug(f"Could not determine if tool uses dataset_ids: {e}")
            return False

    def _normalize_dataset_ids(self, dataset_ids: List[Any]) -> List[UUID]:
        """
        Simple normalization for dataset_ids - convert to UUIDs and remove duplicates.

        Args:
            dataset_ids: List of dataset IDs (can be UUID objects or strings)

        Returns:
            List of unique UUID objects
        """
        if not dataset_ids:
            return []

        # Remove duplicates while preserving order and convert to UUIDs
        seen = set()
        unique_ids = []
        for item in dataset_ids:
            # Convert to UUID if it's a string, keep if already UUID
            try:
                uuid_obj = item if isinstance(item, UUID) else UUID(str(item))
                uuid_str = str(uuid_obj)

                if uuid_str not in seen and self._is_valid_dataset_id(uuid_str):
                    seen.add(uuid_str)
                    unique_ids.append(uuid_obj)

            except ValueError as e:
                logger.warning(f"Invalid UUID format for dataset ID {item}: {e}")
                continue

        return unique_ids

    def register_tool(self, tool_config: ToolReadUnion):
        """Register a system tool with the PydanticAI agent."""
        # if self.is_mcp_agent:
        #     logger.warning("Cannot register system tools on MCP agent")
        #     return None

        self.registered_tools[tool_config.name] = tool_config

        # Create a wrapper function for PydanticAI
        async def tool_wrapper(_context: RunContext[Any], **kwargs) -> Dict[str, Any]:
            """Dynamically execute the registered tool."""
            logger.info(f"In tool wrapper {str(_context)}")
            tool_input = {**kwargs}

            #########################################
            logger.debug(f"CONTEXT: {str(_context)}")
            logger.debug(f"TOOL INPUT: {str(tool_input)}")

            # Debug logging for dataset_ids handling
            tool_uses_dataset_ids = self._tool_uses_dataset_ids(tool_config)
            dataset_ids_in_input = "dataset_ids" in tool_input
            has_context = hasattr(self.agent, "context") and self.agent.context
            context_has_dataset_ids = (
                has_context and "dataset_ids" in self.agent.context
            )

            logger.info(
                f"Tool {tool_config.name} uses dataset_ids: {tool_uses_dataset_ids}"
            )
            logger.info(f"Dataset_ids in tool_input: {dataset_ids_in_input}")
            logger.info(f"Agent has context: {has_context}")
            logger.info(f"Context has dataset_ids: {context_has_dataset_ids}")
            if dataset_ids_in_input:
                logger.info(f"Provided dataset_ids: {tool_input.get('dataset_ids')}")

            # Handle dataset_ids injection and validation for ALL tools that use it
            if tool_uses_dataset_ids:
                # STEP 1: Apply automatic dataset name conversion FIRST if dataset_ids are provided
                if "dataset_ids" in tool_input:
                    logger.info(
                        f"Applying automatic dataset name conversion for {tool_config.name}"
                    )
                    try:
                        # Initialize converter if not already done
                        self._ensure_dataset_converter_initialized()

                        # Convert any dataset names to UUIDs
                        if self.dataset_converter:
                            original_ids = tool_input["dataset_ids"]
                            converted_ids = (
                                self.dataset_converter.convert_dataset_identifiers(
                                    original_ids
                                )
                            )
                            tool_input["dataset_ids"] = converted_ids

                            if converted_ids != [str(id) for id in original_ids]:
                                logger.info(
                                    f"Dataset name conversion applied: {original_ids} -> {converted_ids}"
                                )

                    except Exception as e:
                        logger.error(f"Dataset name conversion failed: {e}")
                        # Continue with original IDs if conversion fails

                # STEP 2: Handle injection from context if no dataset_ids provided
                if (
                    "dataset_ids" not in tool_input
                    and hasattr(self.agent, "context")
                    and self.agent.context
                    and "dataset_ids" in self.agent.context
                ):
                    try:
                        # Get dataset_ids from context and normalize them to UUIDs
                        context_dataset_ids = self.agent.context["dataset_ids"]
                        normalized_ids = self._normalize_dataset_ids(
                            context_dataset_ids
                        )

                        # If no valid dataset IDs found, get all available ones from context
                        if not normalized_ids:
                            logger.warning(
                                f"No valid dataset IDs found for {tool_config.name}, retrieving all available from context"
                            )
                            all_context_dataset_ids = (
                                self._get_valid_dataset_ids_from_context()
                            )

                            # Convert to UUIDs
                            normalized_ids = []
                            for ds_id in all_context_dataset_ids:
                                if self._is_valid_dataset_id(ds_id):
                                    try:
                                        normalized_ids.append(UUID(ds_id))
                                    except ValueError as e:
                                        logger.warning(
                                            f"Invalid UUID format for dataset ID {ds_id}: {e}"
                                        )
                                        continue

                            logger.info(
                                f"Found {len(normalized_ids)} valid dataset IDs from context for {tool_config.name}"
                            )

                        if not normalized_ids:
                            logger.error(
                                f"No valid dataset IDs available for {tool_config.name} operation"
                            )
                            return {
                                "error": f"No valid dataset IDs available for {tool_config.name} operation"
                            }

                        tool_input["dataset_ids"] = normalized_ids

                        logger.info(
                            f"Injected {len(tool_input['dataset_ids'])} unique dataset_ids into {tool_config.name}"
                        )

                    except Exception as e:
                        logger.error(
                            f"Could not inject dataset_ids for {tool_config.name}: {e}"
                        )
                        return {
                            "error": f"Dataset ID injection failed for {tool_config.name}: {str(e)}"
                        }

                # STEP 3: Validate and normalize provided dataset_ids (after conversion)
                elif "dataset_ids" in tool_input:
                    logger.info(
                        f"Validating provided dataset_ids for {tool_config.name}"
                    )
                    try:
                        original_ids = tool_input["dataset_ids"]
                        validated_ids = self._normalize_dataset_ids(original_ids)

                        if len(validated_ids) != len(original_ids):
                            logger.warning(
                                f"Filtered out {len(original_ids) - len(validated_ids)} invalid dataset IDs for {tool_config.name}"
                            )

                        if not validated_ids:
                            logger.error(
                                f"All provided dataset IDs are invalid for {tool_config.name}"
                            )
                            # Fallback: try to get valid dataset IDs from context
                            logger.info(
                                f"Falling back to context dataset_ids for {tool_config.name}"
                            )
                            all_context_dataset_ids = (
                                self._get_valid_dataset_ids_from_context()
                            )

                            validated_ids = []
                            for ds_id in all_context_dataset_ids:
                                try:
                                    validated_ids.append(UUID(ds_id))
                                except ValueError:
                                    continue

                            if not validated_ids:
                                return {
                                    "error": f"All provided dataset IDs are invalid for {tool_config.name}"
                                }
                            logger.info(
                                f"Fallback successful, using {len(validated_ids)} dataset IDs from context"
                            )

                        tool_input["dataset_ids"] = validated_ids
                        logger.info(
                            f"Validated {len(validated_ids)} dataset IDs for {tool_config.name}"
                        )

                    except Exception as e:
                        logger.error(
                            f"Dataset ID validation failed for {tool_config.name}: {e}"
                        )
                        return {
                            "error": f"Dataset ID validation failed for {tool_config.name}: {str(e)}"
                        }

            logger.info(f"Final tool input for {tool_config.name}: {str(tool_input)}")

            try:
                result = await self.execute_tool_async(
                    tool_config, tool_input, _context
                )

                # If result is a Pydantic model, convert to dict
                if isinstance(result, ToolReturn):
                    return result
                elif hasattr(result, "dict"):
                    return result.dict()
                elif hasattr(result, "__dict__"):
                    return result.__dict__
                else:
                    return {"result": result}

            except Exception as e:
                # Log and return error - filtering happens in chat_utils.py
                logger.error(f"Tool {tool_config.name} execution failed: {str(e)}")
                return {"error": f"Tool execution failed: {str(e)}"}

        # Get the input schema to extract parameter info
        try:
            input_schema_cls = locate(tool_config.input_schema)
            if input_schema_cls:
                # Extract field information from the Pydantic model
                schema_fields = (
                    input_schema_cls.__fields__
                    if hasattr(input_schema_cls, "__fields__")
                    else {}
                )

                # Build function signature dynamically
                sig_params = []
                annotations = {}

                context_param = inspect.Parameter(
                    name="_context",
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=RunContext[Any],
                )
                sig_params.append(context_param)
                annotations["_context"] = RunContext[Any]
                for field_name, field_info in schema_fields.items():
                    if field_name.startswith("_"):
                        continue
                    field_type = (
                        field_info.type_ if hasattr(field_info, "type_") else Any
                    )
                    default_value = (
                        field_info.default
                        if hasattr(field_info, "default")
                        and field_info.default is not Ellipsis
                        else inspect.Parameter.empty
                    )

                    param = inspect.Parameter(
                        field_name,
                        inspect.Parameter.KEYWORD_ONLY,
                        default=default_value,
                        annotation=field_type,
                    )
                    sig_params.append(param)
                    annotations[field_name] = field_type

                # Set return type annotation
                annotations["return"] = Dict[str, Any]

                # Create new signature
                tool_wrapper.__signature__ = inspect.Signature(sig_params)
                tool_wrapper.__annotations__ = annotations
        except Exception as e:
            logger.info(
                f"Warning: Could not extract schema info for {tool_config.name}: {e}"
            )

        logger.info(f"Tool config: {str(tool_config)}")
        logger.info(f"Tool name: {tool_config.function_name}")
        # Set function metadata
        tool_wrapper.__name__ = tool_config.function_name
        tool_wrapper.__doc__ = tool_config.description

        # Register the tool with PydanticAI
        logger.info(f"Tool wrapper sig {inspect.signature(tool_wrapper)}")
        self.agent.tool(tool_wrapper)
        logger.info(
            f"[DEBUG] Registered tool '{tool_wrapper.__name__}' with signature: {inspect.signature(tool_wrapper)}"
        )
        return tool_wrapper

    def register_multiple_tools(self, tool_configs: List[ToolReadUnion]):
        """Register multiple system tools at once."""
        for tool_config in tool_configs:
            self.register_tool(tool_config)

    # def provide_previous_history(self, history: List[ChatHistory]):

    async def chat(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        mcp_tools: Optional[List[str]] = None,
        chat_session_id: Optional[str] = None,  # Add this parameter
    ):
        """Chat with the agent - supports both system tools and MCP tools"""
        logger.info(f"Chat query input {message}")
        self.agent.context = context or {}

        # Initialize dataset converter when context is available
        if context and context.get("tools_map"):
            try:
                self.dataset_converter = DatasetNameConverter(context["tools_map"])
                logger.info("Dataset name converter initialized in chat method")

                # Log available conversions for debugging
                conversion_info = self.dataset_converter.get_conversion_info()
                if conversion_info["name_to_uuid_mappings"]:
                    logger.debug(
                        f"Available dataset name conversions: {conversion_info['name_to_uuid_mappings']}"
                    )
                else:
                    logger.debug("No dataset name mappings found in tools_map")
            except Exception as e:
                logger.warning(f"Failed to initialize dataset converter: {e}")

        # Store chat_session_id if provided
        if chat_session_id:
            self.chat_session_id = chat_session_id

        # Classify query intent to help with tool selection (only if needed)
        tools_map = context.get("tools_map", {}) if context else {}
        query_intent = await classify_query_intent(message, tools_map)
        logger.info(f"Query classified as: {query_intent}")

        sub_questions = await split_questions(message)

        responses = []
        pydantic_messages = []
        tool_response = []

        async with self.agent.run_mcp_servers():
            for question in sub_questions:
                try:
                    # Use filtered history that removes tool usage bias
                    filtered_history = self._filter_history_for_tool_selection(
                        self.message_history, question
                    )

                    # Generate enhanced system prompt if tools_map is available and store it in context
                    logger.debug(
                        f"DEBUG: Checking enhanced prompt condition - context exists: {bool(context)}, tools_map exists: {bool(context and context.get('tools_map'))}"
                    )
                    if context and context.get("tools_map"):
                        logger.debug(
                            "DEBUG: Enhanced prompt condition met, generating enhanced system prompt"
                        )
                        clean_tools_map = {}
                        for tool_id, tool_info in context["tools_map"].items():
                            tool_name = tool_info["tool_name"]
                            if tool_name == "File System Navigator":
                                clean_tools_map[tool_id] = {
                                    "tool_name": tool_info["tool_name"],
                                    "dataset_names": tool_info["dataset_names"],
                                    "dataset_ids": list(tool_info["dataset_ids"]),
                                    # Deliberately exclude dataset_ids to prevent LLM from seeing UUIDs
                                }
                            else:
                                clean_tools_map[tool_id] = {
                                    "tool_name": tool_info["tool_name"],
                                    "dataset_ids": list(tool_info["dataset_ids"]),
                                }

                        # Create unbiased system prompt that emphasizes current query
                        enhanced_system_prompt = self._create_unbiased_system_prompt(
                            question, clean_tools_map, query_intent
                        )

                    # Wrap agent run with logfire span using the manager
                    with logfire_manager.span(
                        "agent_run",
                        chat_session_id=self.chat_session_id,  # This will now work
                        question=question,
                    ):

                        # Prepare deps for RunContext
                        if context and context.get("tools_map"):
                            # Include enhanced system prompt in deps for the dynamic system prompt function
                            filtered_history.insert(
                                0,
                                ModelRequest(
                                    parts=[
                                        SystemPromptPart(content=enhanced_system_prompt)
                                    ]
                                ),
                            )

                        # Use filtered history to prevent tool selection bias
                        filtered_history.insert(
                            0,
                            ModelRequest(
                                parts=[SystemPromptPart(content=self.base_prompt)]
                            ),
                        )

                        result = await self.agent.run(
                            question, message_history=filtered_history
                        )

                    pydantic_msg = {"user_prompt": None, "model_response": None}

                    # Extract user prompt and model response from the result
                    user_prompt_found = False
                    model_response_found = False

                    for msg in result.new_messages():
                        for part in msg.parts:
                            logger.info(f"Py message: {part}")
                            if isinstance(msg, ModelRequest) and isinstance(
                                part, ToolReturnPart
                            ):
                                tool_return_data = to_jsonable_python(part)
                                # Add to context - failed calls will be filtered in chat_utils.py
                                tool_response.append(tool_return_data)
                            elif isinstance(msg, ModelRequest) and isinstance(
                                part, UserPromptPart
                            ):
                                if not user_prompt_found:
                                    user_prompt_content = to_jsonable_python(part)
                                    if user_prompt_content and user_prompt_content.get(
                                        "content"
                                    ):
                                        pydantic_msg["user_prompt"] = (
                                            user_prompt_content["content"]
                                        )
                                        user_prompt_found = True
                            elif isinstance(msg, ModelResponse) and isinstance(
                                part, TextPart
                            ):
                                if not model_response_found:
                                    model_response_content = to_jsonable_python(part)
                                    if model_response_content:
                                        pydantic_msg["model_response"] = (
                                            model_response_content
                                        )
                                        model_response_found = True

                    # Only add the message if we have both user_prompt and model_response
                    if pydantic_msg["user_prompt"] and pydantic_msg["model_response"]:
                        pydantic_messages.append(pydantic_msg)
                        logger.debug(
                            f"Successfully created pydantic message: user_prompt={bool(pydantic_msg['user_prompt'])}, model_response={bool(pydantic_msg['model_response'])}"
                        )
                    else:
                        logger.warning(
                            f"Skipping incomplete pydantic message: user_prompt={bool(pydantic_msg['user_prompt'])}, model_response={bool(pydantic_msg['model_response'])}"
                        )

                    responses.append(self.format_response(result))

                except Exception as e:
                    logger.error(f"Error processing question '{question}': {e}")
                    responses.append({"error": f"Failed to process question: {str(e)}"})

        await self._cleanup()
        # Combine all structured responses
        return {
            "responses": responses,
            "pydantic_messages": pydantic_messages,
            "contexts": tool_response,
        }

    def chat_sync(self, message: str):
        """Synchronous chat wrapper."""
        return asyncio.run(self.chat(message))

    async def _cleanup(self):
        """Cleanup MCP servers if any"""
        if self.mcp_servers:
            logger.info("Cleaning up MCP servers")
            # The servers will be cleaned up automatically when the context manager exits
            self.mcp_servers.clear()

    @staticmethod
    def format_response(result):
        output = result.output
        if isinstance(output, BaseModel):
            structured = output.dict()
        elif hasattr(output, "__dict__"):
            structured = output.__dict__
        elif isinstance(output, dict):
            structured = output
        else:
            structured = {"response": output}
        return structured


# ... rest of your existing functions remain the same ...
def _has_ambiguous_tools(tools_map: dict[UUID, dict[str, object]]) -> bool:
    """
    Check if the agent has both File Navigation and Vector Search tools,
    which requires the mandatory two-step semantic search workflow.
    """
    tool_names = set()
    for tool_info in tools_map.values():
        if isinstance(tool_info, dict) and "tool_name" in tool_info:
            tool_names.add(tool_info["tool_name"])

    # Check for tools that require the two-step workflow
    has_file_navigation = "File System Navigator" in tool_names
    has_vector_search = "Vector Search Tool" in tool_names or any(
        "vector" in name.lower() and "search" in name.lower()
        for name in tool_names
        if name != "File System Navigator"
    )

    logger.info(
        f"Tool availability check: File Navigation={has_file_navigation}, Vector Search={has_vector_search}"
    )
    return has_file_navigation and has_vector_search


async def classify_query_intent(
    prompt: str, tools_map: dict[UUID, dict[str, object]]
) -> str:
    """
    Classify user query to determine tool selection - Vector Search vs SQL vs others.
    This helps route content queries to Vector Search instead of defaulting to SQL.
    """
    logger.debug(f"Classifying query intent for: {prompt}")

    # Check if we have any tools that require classification
    tool_names = set()
    for tool_info in tools_map.values():
        if isinstance(tool_info, dict) and "tool_name" in tool_info:
            tool_names.add(tool_info["tool_name"])

    has_vector_search = "Vector Search Tool" in tool_names or any(
        "vector" in name.lower() and "search" in name.lower() for name in tool_names
    )
    has_text_to_sql = "Text to SQL Tool" in tool_names or any(
        "sql" in name.lower() for name in tool_names
    )

    # Only classify if we have tools that need disambiguation
    if not (has_vector_search or has_text_to_sql):
        logger.debug(
            "No Vector Search or Text-to-SQL tools available, skipping classification"
        )
        return "GENERAL"  # No classification needed

    logger.info(
        f"Tools requiring classification available: Vector Search={has_vector_search}, Text-to-SQL={has_text_to_sql}"
    )

    system_prompt = AgentPrompts.query_intent_system_prompt()

    try:
        chat_response = await client_4o.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        intent = chat_response.choices[0].message.content.strip().upper()

        if intent in ["CONTENT", "FILES", "SQL", "GRAPH"]:
            logger.info(f"Query classified as: {intent}")
            return intent
        else:
            logger.warning(
                f"Invalid classification result: {intent}, defaulting to CONTENT"
            )
            return "CONTENT"  # Default to content search when unsure

    except Exception as e:
        logger.error(f"Failed to classify query intent: {e}")
        return "CONTENT"  # Default to content search on error


async def split_questions(prompt: str) -> List[str]:
    """Use LLM to split compound or multi-part queries into individual questions."""
    logger.info(f"Splitting questions for input: {prompt}")

    system_prompt = AgentPrompts.split_question_system_prompt()

    try:
        chat_response = await client_4o.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        content = chat_response.choices[0].message.content.strip()

        # Parse the response
        questions = ast.literal_eval(content)

        # Robust validation
        if not isinstance(questions, list):
            logger.warning(
                f"Response is not a list: {type(questions)}, using fallback."
            )
            return [prompt]

        if not questions:
            logger.warning("Empty question list returned, using fallback.")
            return [prompt]

        if not all(isinstance(q, str) for q in questions):
            logger.warning("Non-string items in question list, using fallback.")
            return [prompt]

        # Additional safety checks
        if len(questions) == 1:
            # If only one question returned, use it (this is expected for most cases)
            logger.info("Single question returned (no splitting needed)")
            return questions

        # If multiple questions returned, validate they make sense
        if len(questions) > 3:
            # Be very conservative - if more than 3 questions, likely an error
            logger.warning(
                f"Too many questions returned ({len(questions)}), likely an error. Using fallback."
            )
            return [prompt]

        # Check if split questions are too short (likely parsing error)
        min_length = 10  # Minimum reasonable question length
        if any(len(q.strip()) < min_length for q in questions):
            logger.warning(
                "Some split questions are too short, likely parsing error. Using fallback."
            )
            return [prompt]

        # Check if any question is just a fragment (no verb or question words)
        question_words = [
            "what",
            "how",
            "when",
            "where",
            "who",
            "why",
            "which",
            "show",
            "list",
            "find",
            "get",
            "tell",
            "give",
            "create",
            "make",
            "analyze",
            "compare",
            "calculate",
        ]
        for q in questions:
            q_lower = q.lower()
            if not any(word in q_lower for word in question_words):
                logger.warning(
                    f"Split question '{q}' doesn't contain typical question words. Using fallback."
                )
                return [prompt]

        # If we get here, the split seems valid
        logger.info(f"Successfully split into {len(questions)} independent questions")
        return questions

    except (SyntaxError, ValueError) as e:
        logger.warning(
            f"Failed to parse LLM response as Python list: {e}. Using fallback."
        )
        return [prompt]
    except Exception as e:
        logger.error(f"Unexpected error in question splitting: {e}. Using fallback.")
        return [prompt]
