"use client";
import React from "react";
import ChatBoxComponentV2 from "@/components/chat-app/chatboxV2";
import { ChatProvider } from "@/context_api/chatContext";
import { useParams } from "next/navigation";
import ChatHistoryComponent from "@/components/chat-app/chatHistory";
import { Plus_Jakarta_Sans } from "next/font/google";
import { useEffect } from "react";
import { useChat } from "@/context_api/chatContext";
import ChatBoxComponent from "@/components/chat-app/chatbox";
import { constants } from "@/lib/constants";
import { useSearchParams } from "next/navigation";
import { UserProvider } from "@/context_api/userContext";

const plusJakartaSans = Plus_Jakarta_Sans({
  subsets: ["latin"],
  weight: ["200", "300", "400", "500", "600"], // Choose specific font weights
});

const ChatApp = () => {
  // const { user } = useUser();
  const { chatAppId } = useParams();
  const searchParams = useSearchParams();
  const chatAppType = searchParams.get("chatAppType");

  return (
    <div
      className={`flex flex-col h-[calc(100vh-56px)] w-full ${plusJakartaSans.className}`}
    >
      <UserProvider>
        <ChatProvider chatId={chatAppId}>
          <ChatAppContent chatAppType={chatAppType} />
        </ChatProvider>
      </UserProvider>
    </div>
  );
};

const ChatAppContent = ({ chatAppType }) => {
  const { chatAppName } = useChat();

  useEffect(() => {
    if (chatAppName) {
      document.title = chatAppName;
    }
  }, [chatAppName]);

  return (
    <div className={"flex h-full w-full gap-3"}>
      <ChatHistoryComponent />
      <div className={"flex flex-col flex-1 h-full min-w-0"}>
        {chatAppType === constants.UNSTRUCTURED_CHAT_APP ||
        chatAppType === constants.SQL_CHAT_APP ? (
          <ChatBoxComponent />
        ) : (
          <ChatBoxComponentV2 />
        )}
      </div>
    </div>
  );
};

export default ChatApp;
