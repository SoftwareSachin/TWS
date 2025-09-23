export type UUID = string;

export enum ChatModelEnum {
  GPT35 = "GPT35",
  GPT41 = "GPT41",
  GPT4o = "GPT4o",
  GPTo3 = "o3-mini",
  GPT5 = "GPT5",
  // Add more if your Enum includes others
}

export const LLMOptions = [
  { value: "GPT35", label: "OpenAI GPT-3.5", provider: "openai" },
  { value: "GPT41", label: "OpenAI GPT-4.1", provider: "openai" },
  { value: "GPT4o", label: "OpenAI GPT-4o ", provider: "openai" },
  { value: "GPTo3", label: "OpenAI o3-mini Reasoning", provider: "openai" },
  { value: "GPT5", label: "OpenAI GPT-5", provider: "openai" },
];

// IAgentCreate
export interface IAgentCreate {
  name: string;
  description?: string;
  prompt_instructions?: string;
  llm_provider?: string;
  llm_model: string | undefined;
  temperature?: number;
  system_prompt?: string;
  memory_enabled?: boolean;
  agent_metadata?: Record<string, any>;
  workspace_tool_ids: UUID[];
}

// IWorkspaceAgentRead extends IAgentCreate
export interface IWorkspaceAgentRead extends IAgentCreate {
  id: UUID;
}

// IAgentUpdate
export interface IAgentUpdate {
  name?: string;
  description?: string;
  prompt_instructions?: string;
  llm_provider?: string;
  llm_model?: string | undefined;
  temperature?: number;
  system_prompt?: string;
  memory_enabled?: boolean;
  agent_metadata?: Record<string, any>;
  workspace_tool_ids?: UUID[];
}
