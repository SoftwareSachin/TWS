import { ApiResponse } from "@/types/ApiResponse";
import { PaginatedResponse } from "@/types/Paginated";
import { OptionType } from "@/types/Agentic";
export interface ChatAppResponse {
  name: string;
  description: string;
  chat_retention_days: number;
  voice_enabled: boolean;
  id: string;
  agents: string;
}

export interface ChatAppForm {
  name: string;
  description: string;
  selectedAgent: string;
  enableVoice: boolean;
}

export type ChatAppPaginatedResponse = ApiResponse<
  PaginatedResponse<ChatAppResponse>
>;

export interface ChatApp {
  id: string;
  name: string;
  description: string;
  agent_id?: string;
  agent_name?: string;
  voice_enabled: boolean;
  created_at?: string;
  updated_at?: string;
}

export interface ChatAppFormData {
  name: string;
  description: string;
  selectedAgent: string;
  enableVoice: boolean;
}
export interface CreateChatAppFormProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (data: ChatAppFormData) => void;
  fieldLabels: {
    name: string;
    description: string;
    selectedAgent: string;
    enableVoice: string;
  };
  agentOptions: OptionType[];
  onAgentsDropdownOpen: () => void;
  loadingAgents: boolean;
  chatAppToEdit?: ChatApp | null;
}
