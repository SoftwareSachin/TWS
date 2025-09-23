import { Page, PaginatedResponse } from "@/types/Paginated";
import { http } from "..";
import {
  IWorkspaceToolAdd,
  IWorkspaceToolListRead,
  IWorkspaceToolUpdate,
} from "@/agent_schemas/workspace_tool_schema";
import { IAgentCreate, IAgentUpdate } from "@/agent_schemas/agent_schema";
import { IToolCreate, IToolUpdate } from "@/agent_schemas/tool_schema";
import { AgenticPaginatedResponse } from "@/types/Agentic";
import { ApiResponse } from "@/types/ApiResponse";

export const getToolsForWorkspace = (
  workspaceId: string,
  pagination?: Page,
  toolKind?: string,
) => {
  const params = new URLSearchParams();

  if (toolKind) {
    params.append("tool_kind", toolKind);
  }

  return http.get<ApiResponse<PaginatedResponse<IWorkspaceToolListRead>>>(
    `/workspace/${workspaceId}/tool?${params.toString()}&page=${pagination?.page}&size=${pagination?.size}`,
  );
};

export const getToolForWorkspaceByID = async (
  workspaceId: string,
  workspaceToolId: string,
) => {
  return await http.get<ApiResponse<IWorkspaceToolListRead>>(
    `/workspace/${workspaceId}/workspace_tool/${workspaceToolId}`,
  );
};

export const createToolForWorkspace = (
  workspaceId: string,
  toolPayload: IWorkspaceToolAdd,
) => {
  return http.post<AgenticPaginatedResponse<IWorkspaceToolAdd>>(
    `workspace/${workspaceId}/tool`,
    toolPayload,
  );
};

export const updateToolForWorkspaceById = (
  workspaceId: string,
  workspaceToolId: string,
  toolPayload: IWorkspaceToolUpdate,
) => {
  return http.put<AgenticPaginatedResponse<IWorkspaceToolUpdate>>(
    `workspace/${workspaceId}/workspace_tool/${workspaceToolId}`,
    toolPayload,
  );
};
export const deleteToolsForWorkspaceByID = (
  workspaceId: string,
  toolId: string,
) => {
  return http.delete(`/workspace/${workspaceId}/workspace_tool/${toolId}`);
};

export const getAgentsForWorkspaceByID = (
  workspaceId: string,
  pagination?: Page,
) => {
  return http.get(
    `/workspace/${workspaceId}/agent?page=${pagination?.page}&size=${pagination?.size}`,
  );
};

export const createAgentForWorkspace = (
  workspaceId: string,
  agentPayload: IAgentCreate,
) => {
  return http.post<AgenticPaginatedResponse<IAgentCreate>>(
    `workspace/${workspaceId}/agent`,
    agentPayload,
  );
};

export const deleteAgentsForWorkspaceByID = (
  workspaceId: string,
  agentId: string,
) => {
  return http.delete(`/workspace/${workspaceId}/agent/${agentId}`);
};

export const createExternalMCP = (payload: IToolCreate) => {
  return http.post<AgenticPaginatedResponse<IToolCreate>>(`/tool`, payload);
};

export const getToolsToAssociate = (toolKind?: string, mcpType?: string) => {
  const params = new URLSearchParams();

  if (toolKind) params.append("tool_kind", toolKind);
  if (mcpType) params.append("mcp_type", mcpType);

  const url = `/tool?${params.toString()}`;
  return http.get(url);
};

export const validateConfigCode = (configCode: string) => {
  const parsedConfig = JSON.parse(configCode);
  return http.post(`/tool/validate-mcp-config`, parsedConfig);
};

export const deleteMcpForWorkspaceById = (tool_id: string) => {
  return http.delete(`/tool/${tool_id}`);
};

export const updateMcpById = (toolId: string, payload: IToolCreate) => {
  return http.put<AgenticPaginatedResponse<IToolUpdate>>(
    `/tool/${toolId}`,
    payload,
  );
};

export const updateAgentForWorkspace = (
  workspaceId: string,
  agentId: string,
  agentPayload: IAgentUpdate,
) => {
  return http.put<AgenticPaginatedResponse<IAgentUpdate>>(
    `/workspace/${workspaceId}/agent/${agentId}`,
    agentPayload,
  );
};
