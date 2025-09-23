"use client";

import React, { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import {
  createToolForWorkspace,
  deleteToolsForWorkspaceByID,
  getToolsToAssociate,
  getToolsForWorkspace,
  updateToolForWorkspaceById,
} from "@/api/agents";
import AddToolToWorkspacePage from "@/components/forms/addWorkspaceToolPage";
import { showError, showSuccess } from "@/utils/toastUtils";
import DeleteModal from "@/components/forms/deleteModal";
import { getDataSet } from "@/api/dataset";
import { AddWorkspaceToolFormData, OptionType, Tool } from "@/types/Agentic";
import WorkspaceUtilCard from "@/components/ui/WorkspaceUtilCard";
import { debugLog } from "@/utils/logger";
import WorkspacePageWrapper from "@/components/Agentic/layout";
import { agentic_constants } from "@/lib/AgenticConstants";
import { IWorkspaceToolAdd } from "@/agent_schemas/workspace_tool_schema";

const ToolsPage = () => {
  const [tools, setTools] = useState<Tool[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [mcpOptions, setMcpOptions] = useState<OptionType[]>([]);
  const [systemToolOptions, setSystemToolOptions] = useState<OptionType[]>([]);
  const [datasetOptions, setDatasetOptions] = useState<OptionType[]>([]);
  const [loadingOptions, setLoadingOptions] = useState(true);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [selectedToolId, setSelectedToolId] = useState<string | null>(null);
  const [editTool, setEditTool] = useState<Tool | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [pagination, setPagination] = useState({
    page: 1,
    size: 50,
  });
  const [isDatasetRequired, setIsDatasetRequired] = useState(false);
  const [totalPages, setTotalPages] = useState(1);
  const [isReadOnlyMode, setIsReadOnlyMode] = useState(false);

  const params = useParams();
  const workspaceId = params.workspaceId as string;

  const fetchTools = async () => {
    setLoading(true);
    try {
      const toolsRes = await getToolsForWorkspace(workspaceId, pagination);
      const { items, total } = toolsRes.data.data;
      const tools: Tool[] = items.map((tool) => {
        const isMCP = tool.tools.some(
          (t) => t.tool_type === agentic_constants.TOOL_KIND.MCP,
        );
        const isSystem = tool.tools.some(
          (t) => t.tool_type === agentic_constants.TOOL_KIND.SYSTEM,
        );
        let tool_type: string = agentic_constants.TOOL_KIND.SYSTEM;
        if (isMCP) tool_type = agentic_constants.TOOL_KIND.MCP;
        const datasetRequired =
          tool.tools?.some((t) => t.dataset_required === true) || false;

        return {
          id: tool.id,
          title: tool.name || "",
          description: tool.description || "",
          tag: isMCP
            ? agentic_constants.TOOL_TYPE.MCP_TOOL
            : agentic_constants.TOOL_TYPE.SYSTEM_TOOL,

          tool_type,

          selectedMCPs: isMCP
            ? tool.tools
                .filter((t) => t.tool_type === agentic_constants.TOOL_KIND.MCP)
                .map((t) => t.tool_id)
            : [],
          selectedSystemTools: isSystem
            ? tool.tools
                .filter(
                  (t) => t.tool_type === agentic_constants.TOOL_KIND.SYSTEM,
                )
                .map((t) => t.tool_id)
            : [],
          selectedDatasets: tool.dataset_ids || [],
          dataset_required: datasetRequired,
        };
      });
      setTools(tools);
      const totalTools = total || 0;
      const calculatedTotalPages = Math.ceil(totalTools / pagination.size);
      setTotalPages(calculatedTotalPages);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.detail || "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const fetchSystemTools = async (tool_kind?: string) => {
    setLoadingOptions(true);
    try {
      const systemToolsRes = await getToolsToAssociate(tool_kind);

      const items = systemToolsRes?.data?.data?.items || [];

      const systemTools = items.map((tool: any) => ({
        label: tool.name,
        value: tool.tool_id || tool.name,
        dataset_required: tool.dataset_required,
      }));

      setSystemToolOptions(systemTools);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.detail || "Unknown error");
    } finally {
      setLoadingOptions(false);
    }
  };

  const getMcpTools = async (tool_kind?: string, mcp_type?: string) => {
    setLoadingOptions(true);
    try {
      const mcptoolsRes = await getToolsToAssociate(
        tool_kind ?? "",
        mcp_type ?? "",
      );
      const items = mcptoolsRes?.data?.data?.items || [];

      const mcpTools = items.map((tool: any) => ({
        label: tool.name,
        value: tool.tool_id || tool.name,
      }));
      setMcpOptions(mcpTools);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.detail || "Unknown error");
    } finally {
      setLoadingOptions(false);
    }
  };

  const fetchDatasets = async (ingested: boolean) => {
    setLoadingOptions(true);
    try {
      const res = await getDataSet(
        workspaceId,
        { page: 1, size: 50 },
        ingested,
      );
      if (res.status === 200) {
        const items = res?.data?.data?.items || [];
        const datasets = items.map((dataset: any) => ({
          label: dataset.name,
          value: dataset.id,
          type:
            dataset.source_type &&
            (dataset.source_type === "pg_db" ||
              dataset.source_type === "mysql_db")
              ? agentic_constants.DATASET_TYPE.SQL
              : agentic_constants.DATASET_TYPE.UNSTRUCTURED,
          graph_status: dataset.graph_status,
        }));
        setDatasetOptions(datasets);
      }
    } catch (err: any) {
      showError(err.response?.data?.detail || err.detail || "Unknown error");
    } finally {
      setLoadingOptions(false);
    }
  };

  useEffect(() => {
    if (
      editTool &&
      editTool.tag === agentic_constants.TOOL_TYPE.SYSTEM_TOOL &&
      (editTool.selectedSystemTools?.length ?? 0) > 0
    ) {
      handleSystemToolSelectChange(editTool.selectedSystemTools || []);
    }
  }, [editTool, systemToolOptions]);

  useEffect(() => {
    if (workspaceId) {
      fetchTools();
    }
  }, [workspaceId, pagination]);
  const handleAddToolSubmit = async (data: AddWorkspaceToolFormData) => {
    try {
      const isMCP =
        data?.workspaceTool?.toLowerCase() === agentic_constants.TOOL_KIND.MCP;

      const toolPayload: IWorkspaceToolAdd = {
        name: data?.agentName,
        description: data?.description,
      };

      if (isMCP) {
        toolPayload.tool_ids = data?.selectedMCPs || [];
      } else if (
        Array?.isArray(data?.selectedSystemTools) &&
        data?.selectedSystemTools?.length > 0
      ) {
        ((toolPayload.dataset_ids = data?.selectedDatasets || []),
          (toolPayload.tool_ids = data?.selectedSystemTools || []));
      }

      debugLog("Sending tool configuration payload");

      let res;
      if (editTool) {
        res = await updateToolForWorkspaceById(
          workspaceId,
          editTool.id,
          toolPayload,
        );
        if (res?.status === 200) {
          showSuccess("Tool updated successfully");
          setIsModalOpen(false);
          setEditTool(null);
          await fetchTools();
        }
      } else {
        res = await createToolForWorkspace(workspaceId, toolPayload);
        if (res?.status === 200) {
          showSuccess("Tool created successfully");
          setIsModalOpen(false);
          await fetchTools();
        }
      }
    } catch (err: any) {
      showError(
        err?.response?.data?.detail ||
          err?.detail ||
          `Failed to ${editTool ? "update" : "create"} tool`,
      );
    }
  };

  const handleOpenEditModal = (tool: Tool) => {
    if (
      tool.tag === agentic_constants.TOOL_TYPE.MCP_TOOL &&
      mcpOptions.length === 0
    ) {
      getMcpTools(agentic_constants.TOOL_KIND.MCP);
    } else if (
      tool.tag === agentic_constants.TOOL_TYPE.SYSTEM_TOOL &&
      systemToolOptions.length === 0
    ) {
      fetchSystemTools(agentic_constants.TOOL_KIND.SYSTEM);
    }

    setEditTool(tool);
    setIsModalOpen(true);
  };

  const handleShowDetails = (tool: Tool) => {
    if (
      tool.tag === agentic_constants.TOOL_TYPE.MCP_TOOL &&
      mcpOptions.length === 0
    ) {
      getMcpTools(agentic_constants.TOOL_KIND.MCP);
    } else if (
      tool.tag === agentic_constants.TOOL_TYPE.SYSTEM_TOOL &&
      systemToolOptions.length === 0
    ) {
      fetchSystemTools(agentic_constants.TOOL_KIND.SYSTEM);
    }

    setEditTool(tool);
    setIsReadOnlyMode(true);
    setIsModalOpen(true);
  };

  const handleOpenDeleteModal = (toolId: string) => {
    setSelectedToolId(toolId);
    setIsDeleteModalOpen(true);
  };

  const handleDeleteTool = async () => {
    setLoading(true);
    debugLog("Deleting tool with ID:", selectedToolId);

    if (!selectedToolId) {
      return;
    }
    try {
      const res = await deleteToolsForWorkspaceByID(
        workspaceId,
        selectedToolId,
      );
      if (res.status === 204) {
        showSuccess("Tool deleted successfully");
      }
      await fetchTools();
    } catch (err: any) {
      showError(
        "Failed to delete tool: " + (err.response?.data?.detail || err.detail),
      );
    } finally {
      setIsDeleteModalOpen(false);
      setSelectedToolId(null);
      setLoading(false);
    }
  };

  const handleSystemToolSelectChange = (selectedIds: string[]) => {
    const selected = systemToolOptions.filter((tool) =>
      selectedIds.includes(tool.value),
    );
    const needsDataset = selected.some((tool) => tool.dataset_required);
    setIsDatasetRequired(needsDataset);

    if (needsDataset) {
      fetchDatasets(true);
    }
  };

  return (
    <WorkspacePageWrapper
      title="Tools"
      itemCount={tools.length}
      loading={loading}
      error={error}
      searchTerm={searchTerm}
      onSearchChange={setSearchTerm}
      onCreateClick={() => {
        setEditTool(null);
        setIsModalOpen(true);
      }}
      renderItems={() =>
        tools.map((tool, idx) => (
          <WorkspaceUtilCard
            key={idx}
            title={tool.title}
            description={tool.description}
            tag={tool.tag}
            onDelete={() => handleOpenDeleteModal(tool.id)}
            onEdit={() => handleOpenEditModal(tool)}
            onShowDetails={() => handleShowDetails(tool)}
          />
        ))
      }
      CreateModal={
        <AddToolToWorkspacePage
          fetchSystemTools={() =>
            fetchSystemTools(agentic_constants.TOOL_KIND.SYSTEM)
          }
          fetchMCPTools={() => getMcpTools(agentic_constants.TOOL_KIND.MCP)}
          isOpen={isModalOpen}
          onClose={() => {
            setIsModalOpen(false);
            setEditTool(null);
            setIsReadOnlyMode(false);
          }}
          onSubmit={handleAddToolSubmit}
          mcpOptions={mcpOptions}
          systemToolOptions={systemToolOptions}
          datasetsOptions={datasetOptions}
          loadingOptions={loadingOptions}
          isEditMode={!!editTool}
          isReadOnly={isReadOnlyMode}
          initialData={
            editTool
              ? {
                  agentName: editTool.title,
                  description: editTool.description || "",
                  workspaceTool:
                    editTool.tag === agentic_constants.TOOL_TYPE.MCP_TOOL
                      ? agentic_constants.TOOL_TYPE.MCP_TOOL
                      : agentic_constants.TOOL_TYPE.SYSTEM_TOOL,
                  selectedDatasets:
                    editTool.tag === agentic_constants.TOOL_TYPE.SYSTEM_TOOL &&
                    editTool.selectedDatasets
                      ? editTool.selectedDatasets
                      : [],
                  selectedMCPs:
                    editTool.tag === agentic_constants.TOOL_TYPE.MCP_TOOL &&
                    editTool.selectedMCPs
                      ? editTool.selectedMCPs
                      : [],
                  selectedSystemTools:
                    editTool.tag === agentic_constants.TOOL_TYPE.SYSTEM_TOOL &&
                    editTool.selectedSystemTools
                      ? editTool.selectedSystemTools
                      : [],
                  dataset_required: editTool.dataset_required,
                }
              : undefined
          }
          onSystemToolSelectChange={handleSystemToolSelectChange}
          showDataset={isDatasetRequired}
          showLLMProvider={false}
          showPromptInstructions={false}
        />
      }
      DeleteModal={
        <DeleteModal
          isOpen={isDeleteModalOpen}
          onClose={() => setIsDeleteModalOpen(false)}
          onDelete={handleDeleteTool}
          title="Are you sure you want to delete this tool?"
        />
      }
      pagination={pagination}
      totalPages={totalPages}
      onPaginationChange={setPagination}
    />
  );
};

export default ToolsPage;
