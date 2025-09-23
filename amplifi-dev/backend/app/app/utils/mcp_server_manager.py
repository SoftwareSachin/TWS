import asyncio
import os
import re
from typing import Any, Dict, List
from uuid import UUID

from pydantic_ai.mcp import MCPServerStdio

from app.be_core.logger import logger


class MCPServerManager:
    """Simple manager for MCP servers without agent/LLM dependencies"""

    DELAY_BETWEEN_TOOL_TESTS = 5  # seconds
    RETRY_DELAY_SECONDS = 15  # seconds
    EXPONENTIAL_BACKOFF_BASE = 2  # seconds
    EXPONENTIAL_BACKOFF_MULTIPLIER = 2  # multiplier for exponential backoff
    TOOL_TEST_TIMEOUT = 10  # seconds for individual tool tests

    def __init__(self):
        self._servers: Dict[UUID, MCPServerStdio] = {}

    def _resolve_env_vars(self, value: str) -> str:
        """Resolve environment variables in configuration values"""
        if not isinstance(value, str):
            return value

        pattern = r"\$\{([^}]+)\}"

        def replace_var(match):
            var_name = match.group(1)
            return os.getenv(var_name, f"${{{var_name}}}")

        return re.sub(pattern, replace_var, value)

    def _process_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process configuration dictionary to resolve variables"""
        processed = {}
        for key, value in config.items():
            if isinstance(value, str):
                processed[key] = self._resolve_env_vars(value)
            elif isinstance(value, dict):
                processed[key] = self._process_config(value)
            elif isinstance(value, list):
                processed[key] = [
                    self._resolve_env_vars(item) if isinstance(item, str) else item
                    for item in value
                ]
            elif hasattr(value, "model_dump"):  # Pydantic model
                # Convert Pydantic model to dict and process recursively for consistency
                # This ensures nested dictionaries get proper environment variable resolution
                model_dict = value.model_dump()
                processed[key] = self._process_config(model_dict)
            elif hasattr(value, "__dict__"):  # Any object with attributes
                # Convert object to dict and process recursively
                processed[key] = self._process_config(vars(value))
            else:
                processed[key] = value
        return processed

    async def list_tools(
        self, server_config: dict, timeout_seconds: int = 30
    ) -> List[dict]:
        """List available tools from an MCP server with timeout"""
        try:
            logger.info(
                f"Attempting to list tools for server: {server_config.get('name', 'unknown')}"
            )
            server = self.create_server(server_config)

            # Use async context manager with timeout
            async with asyncio.timeout(timeout_seconds):
                async with server:
                    await asyncio.sleep(1)  # Give server time to initialize

                    # List tools and convert to dictionaries
                    raw_tools = await server.list_tools()
                    tools = [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters_schema": tool.inputSchema,
                        }
                        for tool in raw_tools
                    ]
                    logger.info(f"Found {len(tools)} tools in MCP server")
                    return tools

        except asyncio.TimeoutError:
            error_msg = f"MCP server connection timeout after {timeout_seconds} seconds"
            logger.error(error_msg)
            raise ConnectionError(error_msg)
        except FileNotFoundError as e:
            error_msg = f"MCP server executable not found: {str(e)}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        except Exception as e:
            logger.error(f"Error listing MCP server tools: {e}", exc_info=True)
            raise

    def create_server(self, config: dict) -> MCPServerStdio:
        """Create an MCP server from configuration"""
        try:
            processed_config = self._process_config(config)

            if not processed_config:
                raise ValueError("Empty MCP server configuration")

            for server_name, server_config in processed_config.items():
                logger.info(f"Creating MCP server {server_name}...")

                # Ensure server_config is a dictionary
                if not isinstance(server_config, dict):
                    raise ValueError(
                        f"Server config for {server_name} must be a dictionary, got {type(server_config)}"
                    )

                # Validate required fields
                if "command" not in server_config:
                    raise ValueError(
                        f"Missing 'command' in server config for {server_name}"
                    )

                if "args" not in server_config:
                    raise ValueError(
                        f"Missing 'args' in server config for {server_name}"
                    )

                return MCPServerStdio(
                    command=server_config["command"],
                    args=server_config["args"],
                    env=server_config.get("env", {}),
                    cwd=server_config.get("cwd"),
                    timeout=server_config.get("timeout", 30),
                )

            raise ValueError("No valid server configuration found")

        except Exception as e:
            logger.error(f"Error creating MCP server: {e}")
            raise

    async def test_tool_execution(
        self,
        server_config: dict,
        timeout_seconds: int = 30,
        max_retries: int = 3,
        enable_retries: bool = True,
    ) -> List[dict]:
        """
        Test actual execution of tools to validate authentication and permissions

        Args:
            server_config: MCP server configuration
            timeout_seconds: Overall timeout for all tool tests
            max_retries: Maximum number of retry attempts for rate-limited tools
            enable_retries: Whether to enable retry logic for rate-limited tools
        """
        try:
            logger.info(
                f"Testing tool execution for server: {server_config.get('name', 'unknown')}"
            )
            server = self.create_server(server_config)

            tool_test_results = []

            # First, quickly get the tool count to calculate proper timeout
            temp_timeout = min(timeout_seconds, 10)  # Quick check for tool count
            try:
                async with asyncio.timeout(temp_timeout):
                    async with server:
                        raw_tools = await server.list_tools()
                        tool_count = len(raw_tools)
            except Exception:
                # If we can't get tool count, use original timeout
                tool_count = 1

            # Calculate dynamic timeout: base timeout + delays for rate limiting and retries
            # Account for: 5 seconds delay per tool + potential retry delays (2+4+8=14 seconds max per tool)
            dynamic_timeout = (
                self.TOOL_TEST_TIMEOUT
                + max(0, (tool_count - 1) * self.DELAY_BETWEEN_TOOL_TESTS)
                + (tool_count * self.RETRY_DELAY_SECONDS)
            )
            logger.info(
                f"Using timeout of {dynamic_timeout} seconds for testing {tool_count} tools with retries"
            )

            # Use async context manager with calculated timeout
            async with asyncio.timeout(dynamic_timeout):
                async with server:
                    await asyncio.sleep(1)  # Give server time to initialize

                    # Get list of available tools
                    raw_tools = await server.list_tools()

                    for i, tool in enumerate(raw_tools):
                        test_result = {
                            "tool_name": tool.name,
                            "test_status": "unknown",
                            "error_message": None,
                            "execution_time_ms": None,
                        }

                        try:
                            # Add delay between tool tests to respect rate limits
                            # Skip delay for first tool to avoid unnecessary wait
                            if i > 0:
                                logger.info(
                                    f"Waiting {self.DELAY_BETWEEN_TOOL_TESTS} seconds before testing tool '{tool.name}' to respect rate limits..."
                                )
                                await asyncio.sleep(self.DELAY_BETWEEN_TOOL_TESTS)

                            # Implement retry logic with exponential backoff for rate limiting
                            actual_max_retries = max_retries if enable_retries else 0
                            base_delay = (
                                self.EXPONENTIAL_BACKOFF_BASE
                            )  # Start with 2 seconds

                            for retry_attempt in range(actual_max_retries + 1):
                                try:
                                    start_time = asyncio.get_event_loop().time()

                                    # Try to call the tool with minimal/empty parameters
                                    # Most tools should handle empty or minimal input gracefully
                                    test_params = self._generate_test_parameters(tool)

                                    # Execute the tool with a shorter timeout for individual tests
                                    async with asyncio.timeout(
                                        10
                                    ):  # 10 second timeout per tool
                                        _ = await server.call_tool(
                                            tool.name, test_params
                                        )

                                    end_time = asyncio.get_event_loop().time()
                                    execution_time = int((end_time - start_time) * 1000)

                                    test_result.update(
                                        {
                                            "test_status": "success",
                                            "execution_time_ms": execution_time,
                                        }
                                    )

                                    if retry_attempt > 0:
                                        logger.info(
                                            f"Tool '{tool.name}' test successful on retry {retry_attempt}"
                                        )
                                    else:
                                        logger.info(
                                            f"Tool '{tool.name}' test successful"
                                        )

                                    # Success - break out of retry loop
                                    break

                                except Exception as retry_error:
                                    error_msg = str(retry_error)

                                    # Check if this is a rate limiting error
                                    is_rate_limited = (
                                        "rate limit" in error_msg.lower()
                                        or "429" in error_msg
                                        or "too many requests" in error_msg.lower()
                                    )

                                    if (
                                        is_rate_limited
                                        and retry_attempt < actual_max_retries
                                    ):
                                        # Calculate exponential backoff delay
                                        retry_delay = base_delay * (
                                            self.EXPONENTIAL_BACKOFF_MULTIPLIER
                                            ** retry_attempt
                                        )
                                        logger.info(
                                            f"Tool '{tool.name}' hit rate limit (attempt {retry_attempt + 1}/{actual_max_retries + 1}). "
                                            f"Retrying in {retry_delay} seconds..."
                                        )
                                        await asyncio.sleep(retry_delay)
                                        continue
                                    else:
                                        # Either not rate limited, or we've exhausted retries
                                        raise retry_error

                        except asyncio.TimeoutError:
                            test_result.update(
                                {
                                    "test_status": "timeout",
                                    "error_message": "Tool execution timed out",
                                }
                            )
                            logger.warning(f"Tool '{tool.name}' test timed out")

                        except Exception as e:
                            error_msg = str(e)

                            # Categorize common error types
                            if (
                                "rate limit" in error_msg.lower()
                                or "429" in error_msg
                                or "too many requests" in error_msg.lower()
                            ):
                                # If retries were enabled and max retries > 0, this means we failed after retries
                                if enable_retries and max_retries > 0:
                                    test_result["test_status"] = (
                                        "rate_limited_after_retries"
                                    )
                                else:
                                    test_result["test_status"] = "rate_limited"
                            elif (
                                "authentication" in error_msg.lower()
                                or "api key" in error_msg.lower()
                            ):
                                test_result["test_status"] = "auth_error"
                            elif (
                                "permission" in error_msg.lower()
                                or "unauthorized" in error_msg.lower()
                            ):
                                test_result["test_status"] = "permission_error"
                            elif "not found" in error_msg.lower():
                                test_result["test_status"] = "not_found"
                            else:
                                test_result["test_status"] = "error"

                            test_result["error_message"] = error_msg
                            logger.warning(
                                f"Tool '{tool.name}' test failed: {error_msg}"
                            )

                        tool_test_results.append(test_result)

                    logger.info(f"Completed testing {len(tool_test_results)} tools")
                    return tool_test_results

        except asyncio.TimeoutError:
            error_msg = f"Tool testing timeout after {timeout_seconds} seconds"
            logger.error(error_msg)
            raise ConnectionError(error_msg)
        except Exception as e:
            logger.error(f"Error testing tool execution: {e}", exc_info=True)
            raise

    def _generate_test_parameters(self, tool) -> dict:
        """Generate minimal test parameters for a tool based on its schema"""
        try:
            # Get the tool's parameter schema
            schema = tool.inputSchema

            if not schema or "properties" not in schema:
                return {}

            test_params = {}
            properties = schema.get("properties", {})
            required = schema.get("required", [])

            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "string")

                # Only provide required parameters with minimal safe values
                if param_name in required:
                    if param_type == "string":
                        # Use safe test values or examples if available
                        examples = param_info.get("examples", [])
                        if examples:
                            test_params[param_name] = examples[0]
                        else:
                            test_params[param_name] = "test"
                    elif param_type == "integer":
                        test_params[param_name] = 1
                    elif param_type == "number":
                        test_params[param_name] = 1.0
                    elif param_type == "boolean":
                        test_params[param_name] = False
                    elif param_type == "array":
                        test_params[param_name] = []
                    elif param_type == "object":
                        test_params[param_name] = {}

            return test_params

        except Exception as e:
            logger.warning(
                f"Could not generate test parameters for tool {tool.name}: {e}"
            )
            return {}

    async def validate_config(
        self, server_config: dict, timeout_seconds: int = 30
    ) -> dict:
        """
        Validate MCP server configuration and return detailed results

        Returns:
            dict: Validation results with server info and tools
        """
        try:
            tools = await self.list_tools(server_config, timeout_seconds)
            server_name = list(server_config.keys())[0] if server_config else "unknown"

            return {
                "valid": True,
                "server_name": server_name,
                "tools_count": len(tools),
                "available_tools": tools,
                "message": f"Successfully connected to MCP server '{server_name}'. Found {len(tools)} tools.",
            }

        except Exception as e:
            server_name = list(server_config.keys())[0] if server_config else "unknown"
            return {
                "valid": False,
                "server_name": server_name,
                "tools_count": 0,
                "available_tools": [],
                "message": f"Failed to validate MCP server '{server_name}'",
                "error_details": str(e),
            }
