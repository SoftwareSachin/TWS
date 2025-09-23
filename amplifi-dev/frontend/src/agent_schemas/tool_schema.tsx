export type UUID = string;

export type ToolType = "mcp" | "system";

// Replace with specific MCP types if known
export type MCPType = "internal" | "external";

// ---- Base Interfaces ----

export interface IToolBase {
  name: string;
  description?: string;
  deprecated?: boolean;
  metadata?: Record<string, any>;
}

export interface ISystemToolCreate {
  python_module: string;
  function_name: string;
  is_async: boolean;
  input_schema?: string;
  output_schema?: string;
  function_signature?: string;
}

export interface IMCPToolCreate {
  mcp_subtype: string;
  mcp_server_config: Record<string, any>;
  timeout_secs?: number;
}

// ---- Tool Creation ----

export interface IToolCreate extends IToolBase {
  tool_kind: ToolType;
  system_tool?: ISystemToolCreate;
  mcp_tool?: IMCPToolCreate;
}

export interface IToolCreatedResponse {
  id: UUID;
  name: string;
  tool_kind: ToolType;
}

// ---- Tool Update ----

export interface IToolUpdate {
  name: string;
  description?: string;
  deprecated?: boolean;
  metadata?: Record<string, any>;
  system_tool?: ISystemToolCreate;
  mcp_tool?: IMCPToolCreate;
}

// ---- Tool IDs List ----

export interface IToolIdsList {
  tool_ids: UUID[];
}
