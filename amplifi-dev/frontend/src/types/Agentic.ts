import { ApiResponse } from "./ApiResponse";
import { PaginatedResponse } from "./Paginated";

export interface ExternalMcpFormData {
  mcpName: string;
  description: string;
  configCode: string;
  tool_kind: "mcp";
  mcp_tool?: {
    mcp_subtype: "external";
    mcp_server_config?: string;
  };
}

export interface AddExternalMcpFormProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: ExternalMcpFormData) => void;
  isEditMode?: boolean;
  initialData?: ExternalMcpFormData;
  isReadOnly?: boolean;
}

export interface AddWorkspaceToolFormData {
  agentName: string;
  description: string;
  workspaceTool: string;
  selectedMCPs?: string[];
  selectedSystemTools?: string[];
  selectedDatasets?: string[];
}

export interface OptionType {
  label: string;
  value: string;
  [key: string]: any;
}

export interface AddWorkspaceToolPageProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: AddWorkspaceToolFormData) => void;
  fieldLabels?: Partial<Record<keyof AddWorkspaceToolFormData, string>>;
  mcpOptions?: OptionType[];
  systemToolOptions?: OptionType[];
  datasetsOptions?: OptionType[];
  loadingOptions?: boolean;
  isEditMode: boolean;
  initialData: any;
  showDataset?: boolean;
  showLLMProvider: boolean;
  showPromptInstructions?: boolean;
  agentToEdit?: Agent | null;
  fetchSystemTools?: (toolKind: string) => void;
  fetchMCPTools?: (toolKind: string) => void;
  onSystemToolSelectChange?: (selectedIds: string[]) => void;
  isReadOnly?: boolean;
}

export interface ToolItem {
  id: string;
  name: string;
  description: string;
  tools: {
    tool_id: string;
    name: string;
    description: string;
    tool_type: string;
  }[];
  tool_ids?: string[];
  dataset_ids?: string[];
  total?: number;
}

export interface Tool {
  id: string;
  description: string;
  title: string;
  tag: string;
  tool_type: string;
  selectedMCPs?: string[];
  selectedSystemTools?: string[];
  selectedDatasets?: string[];
  dataset_required?: boolean;
}
export interface AgentFormData {
  agentName: string;
  description: string;
  workspaceTool: string[];
  llmProvider: string;
  prompt_instructions: string;
  llmModel?: string;
}
export interface AddWorkspaceToolFormData {
  agentName: string;
  description: string;
  workspaceTool: string;
  selectedMCPs?: string[];
  selectedSystemTools?: string[];
  selectedDatasets?: string[];
  llmProvider?: string;
  prompt_instructions?: string;
}

export interface MCP {
  id: string;
  title: string;
  description: string;
  tag: string;
  mcp_subtype?: string; // Added for subtype (e.g., "external" or "internal")
  mcp_server_config?: Record<string, any>; // Updated from Json to Record<string, any>
}

export interface AgentItem {
  prompt_instructions: any;
  llm_provider: any;
  id: string;
  name: string;
  description: string;
  agent_type: string;
  workspace_tool_ids: string[];
  workspace_tools?: string[];
  tool_type: string;
  llm_model: string;
}

export interface Agent {
  id: string;
  title: string;
  description?: string;
  tag?: string;
  allToolNames?: string[];
  allAgentNames?: string[];
  workspace_tool_ids?: string[];
  llm_provider?: string;
  prompt_instructions?: string;
  llm_model?: string;
  tool_type: string;
}

export interface CreateAgentFormProps {
  isOpen: boolean;
  isEditMode: boolean;
  isReadOnly?: boolean;
  onClose: () => void;
  onSubmit: (data: AgentFormData) => void;
  fieldLabels: {
    agentName: string;
    description: string;
    workspaceTool: string;
    llmProvider: string;
    prompt_instructions: string;
  };
  workspaceToolOptions: OptionType[];
  onToolsDropdownOpen?: () => void;
  loadingTools: boolean;
  agentToEdit?: Agent | null;
  initialData?: AgentFormData;
}

export type AgenticPaginatedResponse<T> = ApiResponse<PaginatedResponse<T>>;
