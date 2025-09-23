"use client";

import React, { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import {
  createChatApp,
  getChatApp,
  updateChatApp,
  deleteChatApp,
} from "@/api/chatApp";
import { getAgentsForWorkspaceByID } from "@/api/agents";
import { showError, showSuccess } from "@/utils/toastUtils";
import DeleteModal from "@/components/forms/deleteModal";
import CreateChatAppForm from "@/components/forms/createChatAppForm";
import WorkspaceUtilCard from "@/components/ui/WorkspaceUtilCard";
import { debugLog } from "@/utils/logger";
import WorkspacePageWrapper from "@/components/Agentic/layout";
import {
  identifyUserFromObject,
  hashString,
  captureEvent,
} from "@/utils/posthogUtils";
import { useUser } from "@/context_api/userContext";

const ChatAppsPage = () => {
  const { user } = useUser();
  const [chatApps, setChatApps] = useState([]);
  const [loading, setLoading] = useState(true);
  const [pageError, setPageError] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [agentOptions, setAgentOptions] = useState([]);
  const [loadingAgents, setLoadingAgents] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [selectedChatAppId, setSelectedChatAppId] = useState(null);
  const [chatAppToEdit, setChatAppToEdit] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [pagination, setPagination] = useState({
    page: 1,
    size: 50,
  });
  const [totalPages, setTotalPages] = useState(1);

  const params = useParams();
  const workspaceId = Array.isArray(params.workspaceId)
    ? params.workspaceId[0]
    : params.workspaceId;

  const fetchChatApps = async () => {
    setLoading(true);
    try {
      const res = await getChatApp(workspaceId, pagination);
      if (res.status === 200) {
        const { items, total } = res.data.data;
        setChatApps(items);
        setTotalPages(Math.ceil(total / pagination.size));
      } else {
        throw new Error(`Unexpected status ${res.status}`);
      }
    } catch (err) {
      setPageError(
        err.response?.data?.detail || err.message || "Failed to load chat apps",
      );
    } finally {
      setLoading(false);
    }
  };

  const fetchAgents = async () => {
    try {
      setLoadingAgents(true);
      debugLog("Fetching agents for workspace", workspaceId);

      const res = await getAgentsForWorkspaceByID(workspaceId, pagination);
      const items = res.data.data.items;
      const options = items.map((item) => ({
        value: item.id,
        label: item.name,
      }));

      setAgentOptions(options);
    } catch (error) {
      console.error("Failed to fetch agents", error);
    } finally {
      setLoadingAgents(false);
    }
  };

  useEffect(() => {
    if (workspaceId) {
      fetchChatApps();
    }
  }, [workspaceId, pagination]);

  const handleCreateOrUpdateChatApp = async (data) => {
    try {
      const chatAppPayload = {
        name: data.name,
        description: data.description,
        agent_id: data.selectedAgent,
        voice_enabled: data.enableVoice,
        // chat_app_type: "agentic",
      };

      identifyUserFromObject(user);

      if (chatAppToEdit) {
        // Update existing chat app
        const res = await updateChatApp({
          id: workspaceId,
          chatAppId: chatAppToEdit.id,
          body: chatAppPayload,
        });
        if (res.status === 200) {
          // Track chat app edited event
          captureEvent("chat_app_edited", {
            chat_id_hash: hashString(chatAppToEdit.id || ""),
            field_changed: "config",
            has_config_change: true,
            user_id_hash: hashString(user?.clientId || ""),
            description: "User updates Chat App settings via edit",
          });

          showSuccess("Chat app updated successfully");
        }
      } else {
        // Create new chat app
        const res = await createChatApp({
          id: workspaceId,
          body: chatAppPayload,
        });
        if (res.status === 200) {
          // Track chat app created event
          captureEvent("chat_app_created", {
            description_present: !!data.description,
            agent_id_hash: hashString(data.selectedAgent || ""),
            voice_enabled: data.enableVoice,
            user_id_hash: hashString(user?.clientId || ""),
            description: "User submits form to create a new Chat App",
          });

          showSuccess("Chat app created successfully");
        }
      }
      setIsModalOpen(false);
      setChatAppToEdit(null);
      await fetchChatApps();
    } catch (err) {
      showError(err.response?.data?.detail || err.message);
    }
  };

  const handleOpenEditModal = async (chatApp) => {
    setChatAppToEdit(chatApp);
    setIsModalOpen(true);
  };

  const handleOpenDeleteModal = (chatAppId) => {
    setSelectedChatAppId(chatAppId);
    setIsDeleteModalOpen(true);
  };

  const handleDeleteChatApp = async () => {
    setLoading(true);
    if (!selectedChatAppId) return;
    try {
      const res = await deleteChatApp({
        id: workspaceId,
        chatAppId: selectedChatAppId,
      });
      if (res.status === 200) {
        // Track chat app deleted event
        identifyUserFromObject(user);
        captureEvent("chat_app_deleted", {
          chat_id_hash: hashString(selectedChatAppId || ""),
          deleted_by_id_hash: hashString(user?.clientId || ""),
          description: "User deletes a Chat App",
        });

        showSuccess("Chat app deleted successfully");
        await fetchChatApps();
      }
    } catch (err) {
      showError(
        "Failed to delete chat app: " +
          (err.response?.data?.detail || err.message),
      );
    } finally {
      setIsDeleteModalOpen(false);
      setSelectedChatAppId(null);
      setLoading(false);
    }
  };

  return (
    <WorkspacePageWrapper
      title="Chats"
      itemCount={chatApps.length}
      searchTerm={searchTerm}
      onSearchChange={(term) => {
        setSearchTerm(term);
        if (term) {
          identifyUserFromObject(user);
          const filteredCount = chatApps.filter((chatApp) =>
            chatApp.name.toLowerCase().includes(term.toLowerCase()),
          ).length;
          captureEvent("chat_search_used", {
            result_count: filteredCount,
            user_id_hash: hashString(user?.clientId || ""),
            description: "User uses search bar on Chat list",
          });
        }
      }}
      onCreateClick={() => {
        setChatAppToEdit(null);
        setIsModalOpen(true);
      }}
      renderItems={() =>
        chatApps
          .filter((chatApp) =>
            chatApp.name.toLowerCase().includes(searchTerm.toLowerCase()),
          )
          .map((chatApp, idx) => (
            <WorkspaceUtilCard
              key={idx}
              title={chatApp.name}
              description={chatApp.description ?? ""}
              tag={chatApp.chat_app_type || "No Agent"}
              onDelete={() => handleOpenDeleteModal(chatApp.id)}
              onEdit={() => handleOpenEditModal(chatApp)}
              actionUrl={`/chatapp/${chatApp.id}?chatAppType=${chatApp.chat_app_type}`}
              actionText="Launch Chat"
              openInNewTab={true}
              onActionClick={() => {
                // Track chat app launch event
                identifyUserFromObject(user);
                captureEvent("chat_app_launched", {
                  chat_id_hash: hashString(chatApp.id || ""),
                  launch_method: "button",
                  user_id_hash: hashString(user?.clientId || ""),
                  description: "User clicks Launch Chat on Chat App card",
                });
              }}
            />
          ))
      }
      loading={loading}
      error={pageError}
      CreateModal={
        <CreateChatAppForm
          isOpen={isModalOpen}
          onClose={() => {
            setIsModalOpen(false);
            setChatAppToEdit(null);
          }}
          onSubmit={handleCreateOrUpdateChatApp}
          fieldLabels={{
            name: "Chat Name",
            description: "Description",
            selectedAgent: "Select Agent",
            enableVoice: "Enable Voice",
          }}
          agentOptions={agentOptions}
          onAgentsDropdownOpen={fetchAgents}
          loadingAgents={loadingAgents}
          chatAppToEdit={chatAppToEdit}
        />
      }
      DeleteModal={
        <DeleteModal
          isOpen={isDeleteModalOpen}
          onClose={() => setIsDeleteModalOpen(false)}
          onDelete={handleDeleteChatApp}
          title="Are you sure you want to delete this chat app?"
        />
      }
      pagination={pagination}
      totalPages={totalPages}
      onPaginationChange={setPagination}
    />
  );
};

export default ChatAppsPage;
