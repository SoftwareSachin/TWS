"use client";

import React, { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import {
  createAgentForWorkspace,
  deleteAgentsForWorkspaceByID,
  getAgentsForWorkspaceByID,
  getToolsForWorkspace,
  getToolForWorkspaceByID,
  updateAgentForWorkspace,
} from "@/api/agents";
import { showError, showSuccess } from "@/utils/toastUtils";
import DeleteModal from "@/components/forms/deleteModal";

import { Agent, AgentFormData, AgentItem, OptionType } from "@/types/Agentic";
import CreateAgentForm from "@/components/forms/createAgentForm";
import WorkspaceUtilCard from "@/components/ui/WorkspaceUtilCard";
import WorkspacePageWrapper from "@/components/Agentic/layout";
import { agentic_constants } from "@/lib/AgenticConstants";
import {
  identifyUserFromObject,
  captureEvent,
  hashString,
} from "@/utils/posthogUtils";
import { useUser } from "@/context_api/userContext";

const AgentsPage = () => {
  const { user } = useUser();
  const [agents, setAgents] = useState<Agent[]>([]);
  const [workspaceToolsName, setWorkspaceToolsName] = useState<
    Record<string, string>
  >({});
  const [workspaceTools, setWorkspaceTools] = useState<OptionType[]>([]);
  const [workspaceToolIds, setWorkspaceToolIds] = useState<number[]>([]);
  const [loading, setLoading] = useState(true);
  const [pageError, setPageError] = useState<string | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [agentToEdit, setAgentToEdit] = useState<Agent | null>(null);
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [loadingOptions, setLoadingOptions] = useState(true);
  const [isReadOnlyMode, setIsReadOnlyMode] = useState(false);

  const [pagination, setPagination] = useState({
    page: 1,
    size: 50,
  });

  const [totalPages, setTotalPages] = useState(1);

  const params = useParams();
  const workspaceId = Array.isArray(params.workspaceId)
    ? params.workspaceId[0]
    : params.workspaceId;

  const fetchAgents = async () => {
    if (!workspaceId) {
      setPageError("Workspace ID is required");
      setLoading(false);
      return;
    }

    setLoading(true);
    try {
      const res = await getAgentsForWorkspaceByID(workspaceId, pagination);

      const items: AgentItem[] = res.data.data.items;
      const toolNameMap: Record<string, string> = {};
      items.forEach((item) => {
        if (item.workspace_tools) {
          item.workspace_tools.forEach((tool: any) => {
            toolNameMap[tool.id] = tool.name;
          });
        }
      });
      const allToolIds = items.flatMap((item) => item.workspace_tool_ids || []);
      setWorkspaceToolIds(allToolIds.map((id) => Number(id)));
      setWorkspaceToolsName(toolNameMap);

      const mappedAgents: Agent[] = items.map((agent) => {
        const toolIds: string[] = agent.workspace_tool_ids || [];

        return {
          id: agent.id,
          title: agent.name,
          description: agent.description,
          tag: toolNameMap[toolIds?.[0]] ?? "Other",
          allToolNames: toolIds.map((id) => toolNameMap[id]),
          workspace_tool_ids: toolIds,
          llm_provider: agent.llm_provider,
          prompt_instructions: agent.prompt_instructions,
          tool_type:
            agent.tool_type === agentic_constants.TOOL_KIND.MCP
              ? agentic_constants.TOOL_TYPE.MCP_TOOL
              : agentic_constants.TOOL_TYPE.SYSTEM_TOOL,
          llm_model: agent.llm_model,
        };
      });

      setAgents(mappedAgents);
      const totalAgents = res.data?.data?.total ?? 0;
      const calculatedTotalPages = Math.ceil(totalAgents / pagination.size);
      setTotalPages(calculatedTotalPages);
    } catch (err: any) {
      setPageError(err.response?.data?.detail ?? err.detail ?? "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const tools = async () => {
    if (!workspaceId) {
      setError("Workspace ID is required");
      setLoadingOptions(false);
      return;
    }

    setLoadingOptions(true);
    try {
      const systemToolsRes = await getToolsForWorkspace(
        workspaceId,
        pagination,
      );
      const items = systemToolsRes?.data?.data?.items ?? [];

      const systemTools = items.map((item: any) => ({
        value: item.id,
        label: item.name,
        tool_type: item.tools[0].tool_type,
      }));

      setWorkspaceTools(systemTools);
    } catch (err: any) {
      setError(err.response?.data?.detail ?? err.detail ?? "Unknown error");
    } finally {
      setLoadingOptions(false);
    }
  };

  useEffect(() => {
    if (workspaceId) {
      fetchAgents();
    }
  }, [workspaceId, pagination]);

  const handleCreateOrUpdateAgent = async (data: AgentFormData) => {
    if (!workspaceId) {
      showError("Workspace ID is required");
      return;
    }

    try {
      const workspaceToolIds = [...(data.workspaceTool || [])];
      const agentPayload = {
        name: data.agentName,
        description: data.description,
        workspace_tool_ids: workspaceToolIds,
        llm_provider: data.llmProvider,
        prompt_instructions: data.prompt_instructions,
        llm_model: data.llmModel,
      };
      if (agentToEdit) {
        // Update existing agent
        const res = await updateAgentForWorkspace(
          workspaceId,
          agentToEdit.id,
          agentPayload,
        );
        if (res.status === 200) {
          showSuccess("Agent updated successfully");
        }
      } else {
        // Create new agent
        const res = await createAgentForWorkspace(workspaceId, agentPayload);
        if (res.status === 200) {
          showSuccess("Agent created successfully");
        }
      }
      setIsModalOpen(false);
      setAgentToEdit(agentToEdit);
      await fetchAgents();
    } catch (err: any) {
      showError(err.response?.data?.detail ?? err.detial);
    }
  };

  const handleOpenEditModal = async (agent: Agent) => {
    identifyUserFromObject(user);
    captureEvent("agent_edited", {
      agent_id_hash: hashString(agent.id || ""),
      field_changed: "config",
      has_config_change: true,
      workspace_id_hash: hashString(workspaceId || ""),
      description: "User updates config of existing agent",
    });

    let cleanedPromptInstructions = "";

    try {
      const parsed = JSON.parse(agent.prompt_instructions || "{}");

      // If mcp_server_config exists, use its inner object as the prompt_instructions
      if (
        parsed?.mcp_server_config &&
        typeof parsed.mcp_server_config === "object"
      ) {
        cleanedPromptInstructions = JSON.stringify(
          parsed.mcp_server_config,
          null,
          2,
        );
      } else {
        cleanedPromptInstructions = JSON.stringify(parsed, null, 2);
      }
    } catch (err) {
      console.warn("Invalid JSON in prompt_instructions:", err);
      cleanedPromptInstructions = agent.prompt_instructions ?? "";
    }

    setAgentToEdit({
      ...agent,
      prompt_instructions: cleanedPromptInstructions,
    });

    tools(); // your tool-loading logic
    setIsModalOpen(true);
  };

  const handleShowDetails = async (agent: Agent) => {
    identifyUserFromObject(user);
    captureEvent("agent_chat_started", {
      agent_id_hash: hashString(agent.id || ""),
      user_id_hash: hashString(user?.clientId || ""),
      input_length_bucket: "0-50", // You can make this dynamic based on actual input
      tool_used: agent.allToolNames?.join(",") || "",
      workspace_id_hash: hashString(workspaceId || ""),
      description: "Chat session initiated with agent",
    });

    let cleanedPromptInstructions = "";

    try {
      const parsed = JSON.parse(agent.prompt_instructions || "{}");
      if (
        parsed?.mcp_server_config &&
        typeof parsed.mcp_server_config === "object"
      ) {
        cleanedPromptInstructions = JSON.stringify(
          parsed.mcp_server_config,
          null,
          2,
        );
      } else {
        cleanedPromptInstructions = JSON.stringify(parsed, null, 2);
      }
    } catch (err) {
      cleanedPromptInstructions = agent.prompt_instructions ?? "";
    }

    setAgentToEdit({
      ...agent,
      prompt_instructions: cleanedPromptInstructions,
    });

    tools();
    setIsReadOnlyMode(true);
    setIsModalOpen(true);
  };

  const handleOpenDeleteModal = (agentId: string) => {
    setSelectedAgentId(agentId);
    setIsDeleteModalOpen(true);
  };

  const handleDeleteAgent = async () => {
    setLoading(true);
    if (!selectedAgentId || !workspaceId) {
      showError("Workspace ID and Agent ID are required");
      setLoading(false);
      return;
    }

    identifyUserFromObject(user);
    captureEvent("agent_deleted", {
      agent_id_hash: hashString(selectedAgentId),
      workspace_id_hash: hashString(workspaceId),
      deleted_by_id_hash: hashString(user?.clientId || ""),
      description: "User deletes an agent",
    });

    try {
      await deleteAgentsForWorkspaceByID(workspaceId, selectedAgentId);
      showSuccess("Agent deleted successfully");
      await fetchAgents();
    } catch (err: any) {
      showError(
        "Failed to delete agent: " + (err.response?.data?.detail ?? err.detail),
      );
    } finally {
      setIsDeleteModalOpen(false);
      setSelectedAgentId(null);
      setLoading(false);
    }
  };

  return (
    <WorkspacePageWrapper
      title="Agents"
      itemCount={agents.length}
      searchTerm={searchTerm}
      onSearchChange={(term) => {
        setSearchTerm(term);
        if (term) {
          identifyUserFromObject(user);
          const filteredCount = agents.filter((agent) =>
            agent.title.toLowerCase().includes(term.toLowerCase()),
          ).length;
          captureEvent("agent_searched", {
            result_count: filteredCount,
            user_id_hash: hashString(user?.clientId || ""),
            workspace_id_hash: hashString(workspaceId || ""),
            description: "Using search bar within Agents tab",
          });
        }
      }}
      onCreateClick={() => {
        setAgentToEdit(null);
        setIsModalOpen(true);
      }}
      renderItems={() =>
        agents
          .filter((agent) =>
            agent.title.toLowerCase().includes(searchTerm.toLowerCase()),
          )
          .map((agent, idx) => (
            <WorkspaceUtilCard
              key={idx}
              title={agent.title}
              description={agent.description ?? ""}
              tag={agent.tag ?? ""}
              allToolNames={agent.allToolNames}
              onDelete={() => handleOpenDeleteModal(agent.id)}
              onEdit={() => handleOpenEditModal(agent)}
              onShowDetails={() => handleShowDetails(agent)}
            />
          ))
      }
      loading={loading}
      error={pageError}
      CreateModal={
        <CreateAgentForm
          isOpen={isModalOpen}
          onClose={() => {
            setIsModalOpen(false);
            setAgentToEdit(null);
            setIsReadOnlyMode(false);
          }}
          onSubmit={handleCreateOrUpdateAgent}
          fieldLabels={{
            agentName: "Agent Name",
            description: "Agent Description",
            workspaceTool: "Select Tools to Associate",
            llmProvider: "Select LLM Provider",
            prompt_instructions: "Agent Instructions",
          }}
          workspaceToolOptions={workspaceTools}
          loadingTools={false}
          onToolsDropdownOpen={() => tools()}
          initialData={
            agentToEdit
              ? {
                  agentName: agentToEdit.title,
                  description: agentToEdit.description || "",
                  workspaceTool: agentToEdit.workspace_tool_ids || [],
                  llmProvider: agentToEdit.llm_model || "",
                  prompt_instructions: agentToEdit.prompt_instructions || "",
                }
              : undefined
          }
          isEditMode={!!agentToEdit}
          isReadOnly={isReadOnlyMode}
        />
      }
      DeleteModal={
        <DeleteModal
          isOpen={isDeleteModalOpen}
          onClose={() => setIsDeleteModalOpen(false)}
          onDelete={handleDeleteAgent}
          title="Are you sure you want to delete this agent?"
        />
      }
      pagination={pagination}
      totalPages={totalPages}
      onPaginationChange={setPagination}
    />
  );
};

export default AgentsPage;
