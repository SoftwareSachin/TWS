from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator


class MCPServerConfig(BaseModel):
    """Model for individual MCP server configuration"""

    command: str
    args: List[str]
    env: Optional[Dict[str, str]] = None

    @field_validator("command")
    @classmethod
    def validate_command(cls, v):
        if not v or not v.strip():
            raise ValueError("Command cannot be empty or whitespace")
        return v.strip()

    @field_validator("args")
    @classmethod
    def validate_args(cls, v):
        if not isinstance(v, list):
            raise ValueError("Args must be a list")
        # Allow empty args list but ensure all args are strings and not empty
        for i, arg in enumerate(v):
            if not isinstance(arg, str):
                raise ValueError(f"Argument at index {i} must be a string")
            if not arg.strip():
                raise ValueError(f"Argument at index {i} cannot be empty or whitespace")
        return [arg.strip() for arg in v]


class MCPConfigValidationRequest(BaseModel):
    """Request model for MCP configuration validation"""

    mcp_server_config: Dict[str, MCPServerConfig]
    timeout_seconds: Optional[int] = 120
    test_tool_execution: Optional[bool] = False
    enable_retries: Optional[bool] = True
    max_retries: Optional[int] = 3

    @field_validator("mcp_server_config")
    @classmethod
    def validate_mcp_server_config(cls, v):
        if not v:
            raise ValueError("MCP server configuration cannot be empty")

        if not isinstance(v, dict):
            raise ValueError("MCP server configuration must be a dictionary")

        for server_name, config in v.items():
            if not server_name or not server_name.strip():
                raise ValueError("Server name cannot be empty or whitespace")

            # Ensure config is properly structured
            if not isinstance(config, (dict, MCPServerConfig)):
                raise ValueError(
                    f"Configuration for server '{server_name}' must be a valid MCPServerConfig object or dictionary"
                )

        return v

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Timeout must be a positive number")
        return v

    @field_validator("max_retries")
    @classmethod
    def validate_max_retries(cls, v):
        if v is not None and (v < 0 or v > 10):
            raise ValueError("Max retries must be between 0 and 10")
        return v


class MCPConfigValidationResponse(BaseModel):
    """Response model for MCP configuration validation"""

    valid: bool
    server_name: str
    tools_count: int
    available_tools: List[Dict[str, Any]]
    message: str
    error_details: Optional[str] = None
    tool_test_results: Optional[List[Dict[str, Any]]] = None
