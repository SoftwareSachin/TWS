"use client";
import React, { createContext, useContext, useRef, useState } from "react";

// Create the context
const ChatContext = createContext(undefined);

// Define the provider component
export const ChatProvider = ({ children, chatId }) => {
  const [chatAppId] = useState(chatId); // Shared state
  const [chatSessionID, setChatSessionID] = useState(""); // Shared state
  const [chatSessionName, setChatSessionName] = useState(""); // Shared state
  const [chatAppType, setChatAppType] = useState(""); // Shared state
  const [chatResponse, setChatResponse] = useState(null); // Shared state
  const [chatSource, setChatSource] = useState(null); // Shared state
  const [audioText, setAudioText] = useState("");
  const [firstQuery, setFirstQuery] = useState(false);
  const [chatAppName, setChatAppName] = useState("");
  return (
    <ChatContext.Provider
      value={{
        chatAppId,
        setChatAppName,
        chatAppName,
        chatSessionID,
        setChatSessionID,
        audioText,
        setAudioText,
        chatSessionName,
        setChatSessionName,
        chatResponse,
        setChatResponse,
        chatSource,
        setChatSource,
        firstQuery,
        setFirstQuery,
        chatAppType,
        setChatAppType,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
};

// Custom hook to use the context
export const useChat = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error("useChat must be used within a ChatProvider");
  }
  return context;
};
