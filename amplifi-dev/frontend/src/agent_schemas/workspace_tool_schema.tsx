// types/workspaceTools.tsx

export type UUID = string;

// ToolType and MCPType can be enums if they're defined as Enums in Python
export type ToolType = "system" | "mcp"; // Update based on your actual Enum definition
export type MCPType = "internal" | "external"; // Replace with specific values if known

// IWorkspaceToolAdd
export interface IWorkspaceToolAdd {
  name?: string;
  description?: string;
  tool_ids?: UUID[];
  dataset_ids?: UUID[];
  mcp_tools?: string[];
}
export interface IWorkspaceToolUpdate {
  name?: string;
  description?: string;
  tool_ids?: UUID[];
  dataset_ids?: UUID[];
  mcp_tools?: string[];
}

// IToolAssignmentResponse
export interface IToolAssignmentResponse {
  id: UUID;
  tool_ids: UUID[];
}

// ISystemToolRead
export interface ISystemToolRead {
  tool_id: UUID;
  tool_type: ToolType;
  name: string;
  description?: string;
  deprecated?: boolean;
  metadata?: Record<string, any>;
  created_at?: string;
  updated_at?: string;
  python_module?: string;
  function_name?: string;
  is_async?: boolean;
  input_schema?: string;
  output_schema?: string;
  function_signature?: string;
  dataset_required?: boolean;
}

// IMCPToolRead
export interface IMCPToolRead {
  tool_id: UUID;
  tool_type: ToolType;
  name: string;
  description?: string;
  deprecated?: boolean;
  metadata?: Record<string, any>;
  created_at?: string;
  updated_at?: string;
  mcp_subtype?: MCPType;
  mcp_server_config?: Record<string, any>;
  timeout_secs?: number;
  dataset_required?: boolean;
}

// Union type for tools
export type ToolReadUnion = ISystemToolRead | IMCPToolRead;

// IWorkspaceToolListRead
export interface IWorkspaceToolListRead {
  id: UUID;
  name?: string;
  description?: string;
  tools: ToolReadUnion[];
  dataset_ids: UUID[];
  mcp_tools?: string[];
}
