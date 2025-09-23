import { useChat } from "@/context_api/chatContext";
import React, { useEffect, useRef, useState } from "react";
import {
  createChatSession,
  deleteChatSession,
  getChatSessionsByChatAppId,
  updateChatSession,
} from "@/api/chatApp";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import Image from "next/image";
import dots from "@/assets/icons/dots-vertical.svg";
import frame from "@/assets/icons/Frame.svg";
import search from "@/assets/icons/search.svg";
import { constants } from "@/lib/constants";
import { showError, showSuccess } from "@/utils/toastUtils";
import DeleteModal from "../forms/deleteModal";
import {
  identifyUserFromObject,
  hashString,
  captureEvent,
} from "@/utils/posthogUtils";
import { useUser } from "@/context_api/userContext";

const ChatHistoryComponent = () => {
  const { user } = useUser();
  const {
    chatAppId,
    chatSessionID,
    setChatSessionID,
    setChatSessionName,
    chatSessionName,
    firstQuery,
    chatAppName,
  } = useChat();
  const inputRef = useRef(null);
  const [page, setPage] = useState(1);
  const [chatSessions, setChatSessions] = useState([]);
  const [editSessionId, setEditSessionId] = useState(null);
  const [editSessionTitle, setEditSessionTitle] = useState(null);
  const [disableButton, setDisableButton] = useState(false);
  const [totalPages, setTotalPages] = useState(0);

  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [selectedChatSession, setSelectedChatSession] = useState(null);

  useEffect(() => {
    fetchChatSessions();
  }, [chatAppId, page]);
  useEffect(() => {
    if (firstQuery) fetchChatSessions();
  }, [firstQuery]);

  // Update local chatSessions when context chatSessionName changes
  useEffect(() => {
    if (chatSessionName && chatSessionID) {
      setChatSessions((prevSessions) =>
        prevSessions.map((session) =>
          session.id === chatSessionID
            ? { ...session, title: chatSessionName }
            : session,
        ),
      );
    }
  }, [chatSessionName, chatSessionID]);
  const fetchChatSessions = async () => {
    let chatSessions = await getChatSessionsByChatAppId(chatAppId, page);
    if (chatSessions.items.length === 0) {
      await createChatSession(chatAppId);
      chatSessions = await getChatSessionsByChatAppId(chatAppId, page);
    }
    setChatSessions(chatSessions.items);
    handleChatSessionChange(chatSessions.items[0]);
    setDisableButton(false);
    setTotalPages(chatSessions.pages);
  };

  const handleChatSessionChange = (chatSession) => {
    setChatSessionName(chatSession?.title);
    setChatSessionID(chatSession?.id);
  };

  const handleCreateChatSession = async () => {
    if (!disableButton) {
      setDisableButton(true);
      const res = await createChatSession(chatAppId);

      // Track chat session started event
      if (res) {
        identifyUserFromObject(user);
        captureEvent("chatapp_session_started", {
          agent_id_hash: hashString(chatAppId || ""),
          tool_used: "chat",
          session_id_hash: hashString(res.id || ""),
          user_id_hash: hashString(user?.clientId || ""),
          description: "Chat initiated on new interface",
        });
      }

      await fetchChatSessions();
    }
  };

  // const handlePageChange = (isIncrease) => {
  //   if (isIncrease) {
  //     setPage(page === totalPages ? page : page + 1);
  //   } else {
  //     setPage(page > 1 ? page - 1 : 1);
  //   }
  // };

  const handleEditChatSession = (chatSession) => {
    setEditSessionId(chatSession.id);
    setEditSessionTitle(chatSession.title);
    console.log(inputRef);
  };

  useEffect(() => {
    inputRef.current?.focus(); // Auto-focus input when component mounts
  }, [editSessionId]);

  const handleEditInputChange = (e) => {
    setEditSessionTitle(e.target.value);
  };

  const handleEditKeyEvents = (e) => {
    console.log(e.key);
    if (e.key === constants.ESC) {
      resetEdit();
    } else if (e.key === constants.ENTER) {
      resetEdit();
      handleUpdateChatSession(editSessionId, editSessionTitle);
    }
  };

  const handleUpdateChatSession = (chatSessionId, chatSessionTitle) => {
    updateChatSession(chatAppId, chatSessionId, {
      title: chatSessionTitle,
    }).then((res) => {
      showSuccess("Chat Session updated successfully.");
      setChatSessions((prevItems) =>
        prevItems.map((item) =>
          item.id === chatSessionId
            ? { ...item, title: chatSessionTitle }
            : item,
        ),
      );
    });
  };

  const resetEdit = () => {
    setEditSessionId(null);
    setEditSessionTitle(null);
  };

  const handleDeleteChatSession = async () => {
    try {
      if (!selectedChatSession) return;
      await deleteChatSession(chatAppId, selectedChatSession.id);
      showSuccess("Chat session deleted successfully");

      // Remove the deleted session from the list
      const updatedSessions = chatSessions.filter(
        (item) => item.id !== selectedChatSession.id,
      );
      setChatSessions(updatedSessions);

      // If the deleted session was the active one, select the first available session
      if (chatSessionID === selectedChatSession.id) {
        if (updatedSessions.length > 0) {
          // Select the first available session
          handleChatSessionChange(updatedSessions[0]);
        } else {
          // No sessions left, clear the current session
          setChatSessionID(null);
          setChatSessionName(null);
        }
      }

      setSelectedChatSession(null);

      // Refresh the chat sessions list to ensure UI is properly updated
      await fetchChatSessions();
    } catch (error) {
      showError("Failed to delete chat session");
    }
  };

  const handleSearch = () => {};

  return (
    <div className="flex flex-col h-full min-w-[200px] w-[220px] lg:w-[220px] xl:w-[250px] 2xl:w-[300px] bg-white">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-4 border-b border-gray-200">
        <h1
          className="text-[18px] font-bold text-[#656565] px-1 truncate whitespace-nowrap overflow-hidden max-w-[140px]"
          title={chatAppName || "Chat App"}
        >
          {chatAppName || "Chat App"}
        </h1>
        <div className="flex items-center gap-4 px-1">
          <button onClick={handleCreateChatSession}>
            <Image
              src={frame}
              alt="Create new chat session"
              className="cursor-pointer text-blue-500 font-light hover:underline"
            />
          </button>
        </div>
      </div>

      {/*Title */}
      <div className="flex items-center justify-between px-4 py-4 border-b border-gray-200 text-[12px] text-gray-600 font-semibold">
        <span className="items-start px-1">Chat History</span>
      </div>

      {/* Scrollable session list */}
      <div className="flex-1 overflow-y-auto">
        {chatSessions.map((session, index) => {
          const isActive = chatSessionID === session.id;

          return (
            <div
              key={session.id}
              className={`flex items-center justify-between px-2 py-4 border-b border-gray-200 
                ${isActive ? "bg-blue-50 border-l-4 border-blue-600" : ""}
              `}
            >
              {editSessionId === session.id ? (
                <input
                  ref={inputRef}
                  type="text"
                  value={editSessionTitle}
                  onChange={handleEditInputChange}
                  onKeyDown={handleEditKeyEvents}
                  className="flex-1 border-none focus:outline-none"
                />
              ) : (
                <>
                  <div
                    onClick={() => handleChatSessionChange(session)}
                    className={"flex  cursor-pointer flex-col"}
                    title={session.title}
                  >
                    <span className={"text-sm flex-1"}>
                      {/* IOCL Chat {chatSessions.length - index} */}
                      {session.title}
                    </span>
                    {/* <span className={"text-xs text-gray-400 bold flex-1"}>
                      {session.title}
                    </span> */}
                  </div>

                  <DropdownMenu>
                    <DropdownMenuTrigger>
                      <Image
                        src={dots}
                        alt="options"
                        className="cursor-pointer"
                      />
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="start">
                      <DropdownMenuItem
                        onClick={(e) => {
                          e.stopPropagation();
                          handleEditChatSession(session);
                        }}
                        className="hover:!bg-blue-100"
                      >
                        Edit
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedChatSession(session);
                          setIsDeleteModalOpen(true);
                        }}
                        className="hover:!bg-blue-100"
                      >
                        Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </>
              )}
            </div>
          );
        })}
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 py-2 text-gray-600 font-semibold">
          <button onClick={() => setPage((p) => Math.max(1, p - 1))}>
            &lt;
          </button>
          <span>
            {page} / {totalPages}
          </span>
          <button onClick={() => setPage((p) => Math.min(totalPages, p + 1))}>
            &gt;
          </button>
        </div>
      )}
      <DeleteModal
        isOpen={isDeleteModalOpen}
        onClose={() => setIsDeleteModalOpen(false)}
        onDelete={handleDeleteChatSession}
        title={`Delete chat session "${selectedChatSession?.title}"?`}
      />
    </div>
  );
};

export default ChatHistoryComponent;
