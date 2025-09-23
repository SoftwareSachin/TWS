"use client";
import WorkspaceUtilCard from "@/components/ui/WorkspaceUtilCard";
import React, { useEffect, useState } from "react";
import AddExternalMcpForm from "@/components/forms/addExternalMcpForm";
import {
  createExternalMCP,
  deleteMcpForWorkspaceById,
  getToolsToAssociate,
  updateMcpById,
} from "@/api/agents";
import { showError, showSuccess } from "@/utils/toastUtils";
import { ExternalMcpFormData, MCP } from "@/types/Agentic";
import { debugLog } from "@/utils/logger";
import DeleteModal from "@/components/forms/deleteModal";
import WorkspacePageWrapper from "@/components/Agentic/layout";
import { agentic_constants } from "@/lib/AgenticConstants";
import { Page } from "@/types/Paginated";
import { IToolCreate } from "@/agent_schemas/tool_schema";

const McpPage = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [mcps, setMcps] = useState<MCP[]>([]);
  const [countText, setCountText] = useState(0);
  const [pagination, setPagination] = useState({
    page: 1,
    size: 50,
  });
  const [totalPages, setTotalPages] = useState(1);
  const [selectedToolId, setSelectedToolId] = useState<string | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [editMcp, setEditMcp] = useState<MCP | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isReadOnlyMode, setIsReadOnlyMode] = useState(false);

  const fetchMcps = async (pagination?: Page) => {
    try {
      const res = await getToolsToAssociate(agentic_constants.TOOL_KIND.MCP);
      const responseData = res?.data?.data;
      const items = responseData?.items ?? [];

      const mapped: MCP[] = items.map((item: any) => {
        return {
          id: item.tool_id,
          title: item.name,
          description: item.description,
          tag:
            item.mcp_subtype === agentic_constants.MCP_SUBTYPE.EXTERNAL
              ? agentic_constants.MCP_SUBTYPE.EXTERNAL
              : agentic_constants.MCP_SUBTYPE.INTERNAL,
          mcp_subtype: item.mcp_subtype,
          mcp_server_config: item?.mcp_server_config,
        };
      });
      setMcps(mapped);
      const totalMcps = res.data?.data?.total;
      const calculatedTotalPages = Math.ceil(
        totalMcps / (pagination?.size ?? 50),
      );
      setTotalPages(calculatedTotalPages);
      setCountText(countText);
    } catch (error) {
      console.error("Failed to fetch MCPs:", error);
      showError("Failed to load MCPs.");
    } finally {
      setLoading(false);
    }
  };

  const handleOpenDeleteModal = (toolId: string) => {
    setSelectedToolId(toolId);
    setIsDeleteModalOpen(true);
  };

  const handleOpenEditModal = (mcp: MCP) => {
    setEditMcp(mcp);
    setIsModalOpen(true);
  };

  const handleShowDetails = (mcp: MCP) => {
    setEditMcp({
      ...mcp,
    });
    setIsReadOnlyMode(true);
    setIsModalOpen(true);
  };

  const handleDeleteTool = async () => {
    if (!selectedToolId) {
      showError("No MCP selected for deletion.");
      setIsDeleteModalOpen(false);
      return;
    }

    setLoading(true);
    debugLog("Deleting MCP with ID:", selectedToolId);

    try {
      const res = await deleteMcpForWorkspaceById(selectedToolId);
      if (res.status === 204) {
        showSuccess("MCP deleted successfully");
        await fetchMcps(pagination);
      }
    } catch (err: any) {
      showError(
        "Failed to delete MCP: " + (err.response?.data?.detail ?? err.detail),
      );
    } finally {
      setIsDeleteModalOpen(false);
      setSelectedToolId(null);
      setLoading(false);
    }
  };

  const handleAddOrUpdateMcp = async (data: ExternalMcpFormData) => {
    let parsedConfig: Record<string, any> = {};

    try {
      parsedConfig = JSON.parse(data.configCode);
    } catch (err) {
      console.error("Error parsing JSON config:", err);
      showError("Invalid JSON in config code.");
      return;
    }

    const payload: IToolCreate = {
      name: data.mcpName,
      description: data.description,
      tool_kind: data.tool_kind,
      mcp_tool: {
        mcp_subtype: agentic_constants.MCP_SUBTYPE.EXTERNAL,
        mcp_server_config: parsedConfig,
      },
    };

    debugLog(`${editMcp ? "Updating" : "Creating"} external MCP configuration`);

    try {
      let res;
      if (editMcp) {
        res = await updateMcpById(editMcp.id, {
          ...payload,
        });
        if (res.status === 200) {
          showSuccess("MCP updated successfully");
          await fetchMcps(pagination);
        }
      } else {
        res = await createExternalMCP(payload);
        if (res.status === 200) {
          showSuccess("External MCP created successfully");
          await fetchMcps(pagination);
        }
      }
    } catch (err: any) {
      showError(
        `Failed to ${editMcp ? "update" : "create"} MCP: ` +
          (err.response?.data?.detail ?? err.detail),
      );
    } finally {
      setIsModalOpen(false);
      setEditMcp(null);
    }
  };
  useEffect(() => {
    fetchMcps(pagination);
  }, [pagination]);

  const getConfigCodeFromPrompt = (input: string | Record<string, any>) => {
    try {
      const parsed = typeof input === "string" ? JSON.parse(input) : input;

      if (parsed?.mcp_server_config) {
        return JSON.stringify(parsed.mcp_server_config, null, 2);
      }
      return JSON.stringify(parsed, null, 2);
    } catch {
      return typeof input === "string" ? input : JSON.stringify(input, null, 2);
    }
  };

  return (
    <WorkspacePageWrapper
      title={`${agentic_constants.TOOL_TYPE.MCP_TOOL}s`}
      itemCount={mcps.length}
      loading={loading}
      error={error}
      searchTerm={searchTerm}
      onSearchChange={setSearchTerm}
      onCreateClick={() => {
        setEditMcp(null);
        setIsModalOpen(true);
      }}
      renderItems={() =>
        mcps.map((mcp) => (
          <WorkspaceUtilCard
            key={mcp.id}
            title={mcp.title}
            description={mcp.description}
            tag={mcp.tag}
            onDelete={() => handleOpenDeleteModal(mcp.id)}
            onEdit={() => handleOpenEditModal(mcp)}
            onShowDetails={() => handleShowDetails(mcp)}
          />
        ))
      }
      CreateModal={
        <AddExternalMcpForm
          isOpen={isModalOpen}
          onClose={() => {
            setIsModalOpen(false);
            setEditMcp(null);
            setIsReadOnlyMode(false);
          }}
          onSubmit={handleAddOrUpdateMcp}
          initialData={
            editMcp
              ? {
                  mcpName: editMcp.title,
                  description: editMcp.description,
                  configCode: getConfigCodeFromPrompt(
                    editMcp.mcp_server_config || {},
                  ),
                  tool_kind: "mcp",
                }
              : undefined
          }
          isEditMode={!!editMcp}
          isReadOnly={isReadOnlyMode}
        />
      }
      DeleteModal={
        <DeleteModal
          isOpen={isDeleteModalOpen}
          onClose={() => setIsDeleteModalOpen(false)}
          onDelete={handleDeleteTool}
          title="Are you sure you want to delete this MCP?"
        />
      }
      pagination={pagination}
      totalPages={totalPages}
      onPaginationChange={setPagination}
    />
  );
};

export default McpPage;
