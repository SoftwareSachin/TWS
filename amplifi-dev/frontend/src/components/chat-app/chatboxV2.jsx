"use client";
import dynamic from "next/dynamic";
import React, { useEffect, useRef, useState } from "react";
import Image from "next/image";
import * as SpeechSDK from "microsoft-cognitiveservices-speech-sdk";
import { showError, showSuccess } from "@/utils/toastUtils";
import { recognizerConfig, speechConfig } from "@/utils/speechConfig";
import { useChat } from "@/context_api/chatContext";
import {
  getChatAppV2ById,
  getChatHistory,
  getChatResponseV2,
} from "@/api/chatApp";
import { ChatSource } from "@/components/chat-app/chatSource";
import { MarkdownViewer } from "../ui/markdown-viewer";
import { MAX_QUERY_LENGTH } from "@/lib/file-constants";
import ChatNotFoundPage from "@/components/empty-screens/chatNotFound";
import { useUser } from "@/context_api/userContext";
import NewAvatar from "../ui/newAvatar";
import { isChunkNavigationInProgress } from "@/components/chat-app/chatSourceViewer";
import {
  identifyUserFromObject,
  hashString,
  captureEvent,
} from "@/utils/posthogUtils";
const GraphResponseComponent = dynamic(
  () => import("@/components/chat-app/graphResponse"),
  { ssr: false },
);
import SqlDatatableComponent from "./sqlDatatable";

const ThinkingIndicator = ({ isProcessing }) => {
  const [currentMessageIndex, setCurrentMessageIndex] = useState(0);

  const thinkingMessages = [
    "Analyzing your question",
    "Searching through knowledge base",
    "Processing context information",
    "Generating comprehensive response",
    "Almost there...",
    "Finalizing answer",
  ];

  useEffect(() => {
    if (isProcessing) {
      const interval = setInterval(() => {
        setCurrentMessageIndex((prev) => (prev + 1) % thinkingMessages.length);
      }, 2000);

      return () => clearInterval(interval);
    }
  }, [isProcessing]);

  return (
    <div className="flex items-center">
      <span className="text-gray-500 text-sm font-medium animate-pulse">
        {thinkingMessages[currentMessageIndex]}
      </span>
    </div>
  );
};

const ChatBoxComponentV2 = () => {
  const [queryInput, setQueryInput] = useState("");
  const [lastQueryInput, setLastQueryInput] = useState("");
  const [isListeningActive, setIsListeningActive] = useState(false);
  const [isVoiceEnabled, setIsVoiceEnabled] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [chatConversation, setChatConversation] = useState([]);
  const [isSpeakingActive, setIsSpeakingActive] = useState(false);
  const [currentSpeakingIdx, setCurrentSpeakingIdx] = useState(null);
  const [isTTSStarting, setIsTTSStarting] = useState(false);
  const [isChatMissing, setIsChatMissing] = useState(false);
  const conversationRef = useRef(null);
  const speechRecognizerRef = useRef(null);
  const speechSynthesizerRef = useRef(null);
  const plotInteractionRef = useRef(false);
  const mutationObserverRef = useRef(null);
  const inputTextAreaRef = useRef(null);
  const contextToggleRef = useRef(null);
  const tableViewToggleRef = useRef(null);
  const contextElementRefs = useRef({});
  const tableElementRefs = useRef({});
  const { user } = useUser();

  const {
    chatSessionID,
    setChatResponse,
    chatAppId,
    setFirstQuery,
    setChatAppName,
    setChatAppType,
    chatAppType,
    setChatSessionName,
  } = useChat();

  const extractTextBeforeGraph = (responseText) => {
    if (!responseText || typeof responseText !== "string") return "";

    const scriptMatch = responseText.match(
      /<!DOCTYPE html|<html|<script|```html|var data = \[|var layout = {|data = \[|layout = {|Plotly\.newPlot/i,
    );

    if (scriptMatch) {
      const textBeforeGraph = responseText
        .substring(0, scriptMatch.index)
        .trim();
      return textBeforeGraph;
    }

    return responseText;
  };

  const containsGraphData = (responseText) => {
    if (!responseText || typeof responseText !== "string") return false;

    return (
      responseText.includes("<script") ||
      responseText.includes("Plotly.newPlot") ||
      responseText.includes("plotly") ||
      responseText.includes("var data = [") ||
      responseText.includes("var layout = {") ||
      responseText.includes("data = [") ||
      responseText.includes("layout = {")
    );
  };

  // Function to extract text content without HTML/script tags for TTS
  const extractTextContent = (responseText) => {
    if (!responseText) return "";

    // Remove HTML tags and script content
    const withoutScripts = responseText.replace(
      /<script[^>]*>[\s\S]*?<\/script>/gi,
      "",
    );
    const withoutHTML = withoutScripts.replace(/<[^>]*>/g, "");

    // Clean up extra whitespace
    return withoutHTML.replace(/\s+/g, " ").trim();
  };

  useEffect(() => {
    if (chatAppId) {
      fetchChatApplication(chatAppId);
    } else {
      console.log("Skipping fetchChatApplication - missing chatAppId");
    }
  }, [chatAppId]);

  useEffect(() => {
    if (chatSessionID && chatAppType) {
      getChatSessionHistory();
    }
  }, [chatSessionID, chatAppType]);

  useEffect(() => {
    console.log(user);
  }, [user]);

  const getChatSessionHistory = async () => {
    const res = await getChatHistory(chatAppId, chatSessionID);
    console.log("chatHistoryStore", res);

    const histories = res.items.map((r) => {
      const res = JSON.parse(r.llm_response);

      let context = [];
      let tableData = [];
      let csvFileId = null;
      let csvFileName = null;
      let hasStructuredContext = false;

      r.contexts?.forEach((c) => {
        const toolName = c?.tool_name;

        if (toolName === "perform_vector_search") {
          const unstructured = c?.content?.aggregated || [];
          context = [...context, ...unstructured];
        }

        if (toolName === "process_sql_chat_app") {
          hasStructuredContext = true;
          const generatedSql = c?.content?.generated_sql || "";
          if (generatedSql) {
            context.push({
              generated_sql: generatedSql,
            });
          }
          const sqlTableData = c?.content?.table_data || [];
          if (sqlTableData.length > 0) {
            tableData.push(sqlTableData);
          }
          csvFileId = c?.content?.csv_file_id || null;
          csvFileName = c?.content?.csv_file_name || null;
        }
      });

      return {
        query: r.user_query,
        response: res.responses,
        isResponseCopied: false,
        showContext: false,
        contexts: context,
        showTable: false,
        tableData:
          hasStructuredContext && tableData.length > 0 ? tableData : [],
        csvFileId,
        csvFileName,
      };
    });

    console.log("fetched histories", histories);
    setChatConversation(histories);
  };

  useEffect(() => {
    const scrollToEnd = () => {
      if (
        conversationRef.current &&
        !plotInteractionRef.current &&
        contextToggleRef.current === null &&
        tableViewToggleRef.current === null
      ) {
        conversationRef.current.scrollTo({
          top: conversationRef.current.scrollHeight,
          behavior: "smooth",
        });
      } else {
        console.log("🚫 Skipping scroll due to plot interaction");
      }
    };

    const scrollToContext = (index) => {
      const contextElement = contextElementRefs.current[index];
      if (contextElement && conversationRef.current) {
        contextElement.scrollIntoView({
          behavior: "smooth",
          block: "nearest",
        });
      }
    };

    // Skip scrolling if chunk navigation is in progress
    if (isChunkNavigationInProgress()) {
      return;
    }

    // Check if a specific context was just toggled
    if (contextToggleRef.current !== null) {
      const toggledIndex = contextToggleRef.current;
      // Only scroll if the context is being opened (not closed)
      if (chatConversation[toggledIndex]?.showContext) {
        setTimeout(() => scrollToContext(toggledIndex), 100);
      }
    } else {
      scrollToEnd();
    }

    // Scroll to the table if it was just toggled
    const scrollToTable = (index) => {
      const tableElement = tableElementRefs.current[index];
      if (tableElement && conversationRef.current) {
        tableElement.scrollIntoView({
          behavior: "smooth",
          block: "nearest",
        });
      }
    };

    // Check if a specific table was just toggled
    if (tableViewToggleRef.current !== null) {
      const toggledIndex = tableViewToggleRef.current;
      // Only scroll if the context is being opened (not closed)
      if (chatConversation[toggledIndex]?.showTable) {
        setTimeout(() => scrollToTable(toggledIndex), 100);
      }
    } else {
      scrollToEnd();
    }

    const observer = new MutationObserver((mutations) => {
      // Skip all scrolling if chunk navigation is in progress
      if (isChunkNavigationInProgress()) {
        return;
      }

      // Check if mutation is related to minor content changes
      const isMinorContentChange = mutations.every((mutation) => {
        // Skip text-only changes
        if (mutation.type === "characterData") {
          return true;
        }

        // Skip changes within ChatSourceViewer content area
        if (mutation.type === "childList") {
          const targetElement = mutation.target;
          if (
            (targetElement &&
              (targetElement.classList?.contains("text-gray-700") ||
                targetElement.closest(".text-gray-700") ||
                targetElement.classList?.contains("whitespace-pre-wrap") ||
                targetElement.closest(".whitespace-pre-wrap"))) ||
            targetElement.closest(".chat-source-container")
          ) {
            return true;
          }
        }

        return false;
      });

      if (isMinorContentChange) {
        // Don't scroll for minor content changes
        return;
      }

      // Check if context was recently toggled
      if (contextToggleRef.current !== null) {
        const toggledIndex = contextToggleRef.current;
        // Only scroll if the context is being opened (not closed)
        if (chatConversation[toggledIndex]?.showContext) {
          setTimeout(() => scrollToContext(toggledIndex), 100);
        }
      } else if (!plotInteractionRef.current) {
        scrollToEnd();
      }

      // Check if table was recently toggled
      if (tableViewToggleRef.current !== null) {
        const toggledIndex = tableViewToggleRef.current;
        // Only scroll if the table is being opened (not closed)
        if (chatConversation[toggledIndex]?.showTable) {
          setTimeout(() => scrollToTable(toggledIndex), 100);
        }
      } else if (!plotInteractionRef.current) {
        scrollToEnd();
      }
    });

    if (conversationRef.current) {
      observer.observe(conversationRef.current, {
        childList: true,
        subtree: true,
        characterData: true,
      });
      mutationObserverRef.current = observer;
    }
    return () => {
      observer.disconnect();
    };
  }, [chatConversation, lastQueryInput]);

  useEffect(() => {
    if (isListeningActive && inputTextAreaRef.current) {
      inputTextAreaRef.current.scrollTop =
        inputTextAreaRef.current.scrollHeight;
    }
  }, [queryInput, isListeningActive]);

  const toggleMicrophone = () => {
    if (isListeningActive) {
      stopSpeechRecognition();
    } else {
      stopTextToSpeech();
      startSpeechRecognition();
    }
  };

  const handleCopyAction = async (selectedIdx) => {
    if (chatConversation[selectedIdx].isResponseCopied) return;
    try {
      const scrollPos = conversationRef.current.scrollTop;
      const updatedConversation = chatConversation.map((item, idx) =>
        selectedIdx === idx
          ? { ...item, isResponseCopied: !item.isResponseCopied }
          : item,
      );
      setChatConversation(updatedConversation);
      const response = chatConversation[selectedIdx].response;
      const textToCopy = Array.isArray(response)
        ? response.map((item) => item.response || "").join("\n")
        : response || "";

      if (!textToCopy.trim()) {
        showError("No valid text to copy");
        return;
      }
      await navigator.clipboard.writeText(textToCopy);

      // Track copy response event
      identifyUserFromObject(user);
      captureEvent("copy_response", {
        description: "Copy text response generated by agent",
      });

      showSuccess("Copied to clipboard!");
      conversationRef.current.scrollTop = scrollPos;
      setTimeout(() => {
        setChatConversation((prev) =>
          prev.map((item, idx) =>
            selectedIdx === idx ? { ...item, isResponseCopied: false } : item,
          ),
        );
      }, 6000);
    } catch (err) {
      console.log("Failed to copy!", err);
    }
  };

  const validateInputLength = (inputText) => {
    if (inputText.length > MAX_QUERY_LENGTH) {
      showError(`The input you submitted was too long.`);
      return false;
    }
    return true;
  };

  const toggleContextView = (selectedIdx) => {
    contextToggleRef.current = selectedIdx;
    const isShowingContext = !chatConversation[selectedIdx].showContext;

    setChatConversation((prev) =>
      prev.map((item, idx) =>
        selectedIdx === idx
          ? { ...item, showContext: !item.showContext }
          : item,
      ),
    );

    // Track context chunk viewed event when opening context
    if (isShowingContext) {
      identifyUserFromObject(user);
      captureEvent("context_chunk_viewed", {
        chunk_type: "context",
        user_id_hash: hashString(user?.clientId || ""),
        description: "User clicks on or expands a context chunk",
      });
    }

    setTimeout(() => {
      contextToggleRef.current = null;
    }, 1500);
  };

  const toggleTableView = (selectedIdx) => {
    tableViewToggleRef.current = selectedIdx;
    setChatConversation((prev) =>
      prev.map((item, idx) =>
        selectedIdx === idx ? { ...item, showTable: !item.showTable } : item,
      ),
    );
    setTimeout(() => {
      tableViewToggleRef.current = null;
    }, 1500);
  };

  const fetchChatApplication = async (chatAppId) => {
    if (chatAppId) {
      try {
        const response = await getChatAppV2ById(chatAppId, chatAppId);
        setChatAppType(response?.chat_app_type);
        setChatAppName(response?.name);
        setIsVoiceEnabled(!!response?.voice_enabled);
      } catch (error) {
        console.error(
          "Error fetching chat app:",
          error.response?.data || error.message,
        );
        setIsChatMissing(true);
      }
    }
  };

  const partialTextRef = useRef("");

  const startSpeechRecognition = () => {
    if (isChatMissing) return;
    setIsListeningActive(true);
    partialTextRef.current = "";
    let hasSubmittedQuery = false;
    let pauseTimeout = null;
    let lastSpeechTimestamp = Date.now();

    const recognizer = new SpeechSDK.SpeechRecognizer(
      speechConfig,
      recognizerConfig,
    );

    recognizer.recognizing = (_, event) => {
      lastSpeechTimestamp = Date.now();
      if (pauseTimeout) {
        clearTimeout(pauseTimeout);
      }
      setQueryInput(partialTextRef.current + " " + event.result.text);
    };

    recognizer.recognized = (_, evt) => {
      if (evt.result.reason === SpeechSDK.ResultReason.RecognizedSpeech) {
        partialTextRef.current += " " + evt.result.text;
        setQueryInput(partialTextRef.current.trim());

        if (pauseTimeout) {
          clearTimeout(pauseTimeout);
        }

        pauseTimeout = setTimeout(() => {
          if (Date.now() - lastSpeechTimestamp > 3000) {
            const finalText = partialTextRef.current.trim();
            if (finalText && !hasSubmittedQuery) {
              if (!validateInputLength(finalText)) {
                stopSpeechRecognition();
                return;
              }
              hasSubmittedQuery = true;
              recognizer.stopContinuousRecognitionAsync(
                () => {
                  fetchQueryResponse(finalText, true);
                  stopSpeechRecognition();
                },
                (error) => {
                  console.error("Error stopping recognition:", error);
                  stopSpeechRecognition();
                },
              );
            }
          }
        }, 3000);
      }
    };

    recognizer.sessionStopped = async () => {
      if (pauseTimeout) {
        clearTimeout(pauseTimeout);
      }
      const finalText = partialTextRef.current.trim();
      if (finalText && !hasSubmittedQuery) {
        hasSubmittedQuery = true;
        await fetchQueryResponse(finalText, true);
      }
      stopSpeechRecognition();
    };

    recognizer.canceled = () => {
      if (pauseTimeout) {
        clearTimeout(pauseTimeout);
      }
      setIsListeningActive(false);
      if (speechRecognizerRef.current) {
        speechRecognizerRef.current = null;
      }
    };

    recognizer.startContinuousRecognitionAsync();
    speechRecognizerRef.current = recognizer;
  };

  const stopSpeechRecognition = () => {
    if (speechRecognizerRef.current) {
      try {
        speechRecognizerRef.current.stopContinuousRecognitionAsync(
          () => {
            setIsListeningActive(false);
            if (speechRecognizerRef.current) {
              speechRecognizerRef.current = null;
            }
          },
          (error) => {
            console.error("Error stopping recognition:", error);
            setIsListeningActive(false);
            if (speechRecognizerRef.current) {
              speechRecognizerRef.current = null;
            }
          },
        );
      } catch (error) {
        console.error("Error in stopSpeechRecognition:", error);
        setIsListeningActive(false);
        if (speechRecognizerRef.current) {
          speechRecognizerRef.current = null;
        }
      }
    }
  };

  const handleTextInputChange = (event) => {
    setQueryInput(event.target.value);
  };

  const handleEnterKey = (event) => {
    if (event.key === "Enter" && queryInput.trim() !== "" && !isChatMissing) {
      event.preventDefault();

      if (!validateInputLength(queryInput)) {
        return;
      }

      stopTextToSpeech();
      fetchQueryResponse(queryInput, false);
    }
  };

  const initiateChat = () => {
    if (queryInput.trim() === "" || isChatMissing) return;

    if (!validateInputLength(queryInput)) {
      return;
    }

    stopTextToSpeech();
    fetchQueryResponse(queryInput, false);
  };

  let audioPlayerRef = useRef(null);
  const textToSpeech = (text, index, isSsml = false) => {
    if (isTTSStarting || isChatMissing) return;
    setIsTTSStarting(true);

    stopTextToSpeech();

    if (!text || typeof text !== "string" || text.trim() === "") {
      console.error("Invalid text input for TTS:", text);
      showError("Cannot read aloud: No valid text provided");
      setIsTTSStarting(false);
      return;
    }

    try {
      audioPlayerRef.current = new SpeechSDK.SpeakerAudioDestination();

      audioPlayerRef.current.onAudioStart = () => {
        setIsSpeakingActive(true);
        setCurrentSpeakingIdx(index);
        setIsTTSStarting(false);
      };

      audioPlayerRef.current.onAudioEnd = () => {
        setIsSpeakingActive(false);
        setCurrentSpeakingIdx(null);
        setIsTTSStarting(false);
        cleanupTextToSpeech();
      };

      const audioConfig = SpeechSDK.AudioConfig.fromSpeakerOutput(
        audioPlayerRef.current,
      );
      speechSynthesizerRef.current = new SpeechSDK.SpeechSynthesizer(
        speechConfig,
        audioConfig,
      );

      const speakMethod = isSsml ? "speakSsmlAsync" : "speakTextAsync";

      let finalText = text;
      if (isSsml && !text.includes("<voice")) {
        finalText = `
                    <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
                        <voice name="en-US-NovaTurboMultilingualNeural">
                            ${text}
                        </voice>
                    </speak>`;
      }

      speechSynthesizerRef.current[speakMethod](
        finalText,
        (result) => {
          if (result.reason === SpeechSDK.ResultReason.Canceled) {
            console.error("Synthesis canceled:", result.errorDetails);
            setIsTTSStarting(false);
          }
        },
        (error) => {
          console.error("Synthesis error:", error);
          setIsSpeakingActive(false);
          setCurrentSpeakingIdx(null);
          setIsTTSStarting(false);
          cleanupTextToSpeech();
        },
      );
    } catch (error) {
      console.error("Error initializing speech synthesis:", error);
      setIsSpeakingActive(false);
      setCurrentSpeakingIdx(null);
      setIsTTSStarting(false);
      cleanupTextToSpeech();
    }
  };

  const stopTextToSpeech = () => {
    try {
      if (audioPlayerRef.current) {
        audioPlayerRef.current.pause();
        audioPlayerRef.current.close();
        audioPlayerRef.current = null;
      }
      if (speechSynthesizerRef.current) {
        speechSynthesizerRef.current.close();
        speechSynthesizerRef.current = null;
      }
      setIsSpeakingActive(false);
      setCurrentSpeakingIdx(null);
      setIsTTSStarting(false);
    } catch (error) {
      console.error("Error stopping TTS:", error);
    }
  };

  const cleanupTextToSpeech = () => {
    try {
      if (audioPlayerRef.current) {
        audioPlayerRef.current.close();
        audioPlayerRef.current = null;
      }
      if (speechSynthesizerRef.current) {
        speechSynthesizerRef.current.close();
        speechSynthesizerRef.current = null;
      }
    } catch (error) {
      console.error("Error cleaning up TTS:", error);
    }
  };

  useEffect(() => {
    return () => {
      stopTextToSpeech();
    };
  }, []);

  useEffect(() => {
    const container = conversationRef.current;
    if (!container) return;

    const handleMouseMove = (e) => {
      const target = e.target;
      const isOnPlot = !!target.closest(".js-plotly-plot");
      if (isOnPlot && !plotInteractionRef.current) {
        plotInteractionRef.current = true;
      } else if (!isOnPlot && plotInteractionRef.current) {
        plotInteractionRef.current = false;
      }
    };

    container.addEventListener("mousemove", handleMouseMove);
    return () => {
      container.removeEventListener("mousemove", handleMouseMove);
    };
  }, []);

  const fetchQueryResponse = async (inputText, isAudio) => {
    if (isChatMissing) return;

    if (!validateInputLength(inputText)) {
      return;
    }

    try {
      setIsProcessing(true);
      setLastQueryInput(inputText);
      setQueryInput("");

      // Track question asked event
      identifyUserFromObject(user);
      const startTime = Date.now();
      captureEvent("question_asked", {
        chat_id_hash: hashString(chatAppId || ""),
        message_length_bucket:
          inputText.length <= 50
            ? "short"
            : inputText.length <= 200
              ? "medium"
              : "long",
        input_type: isAudio ? "voice" : "text",
        user_id_hash: hashString(user?.clientId || ""),
        description: "User sends a message in enter question",
      });

      const queryReply = await getChatResponseV2(
        { chatAppId, chatSessionID, chatAppId },
        inputText,
      );

      const responseTime = Date.now() - startTime;

      // Track response received event
      captureEvent("response_received", {
        chat_id_hash: hashString(chatAppId || ""),
        response_type: queryReply.tableData?.length > 0 ? "table" : "text",
        time_to_response_ms: responseTime,
        description: "Agent sends back a response",
      });

      setIsProcessing(false);
      setChatResponse(queryReply);
      updateChatConversation(inputText, queryReply);

      // Update chat session name if backend returned an updated title
      if (queryReply?.updatedSessionTitle) {
        setChatSessionName(queryReply.updatedSessionTitle);
      }
    } catch (error) {
      console.error("Error:", error);
      if (error.response?.status === 404) {
        setIsChatMissing(true);
      } else if (error.response?.status === 500) {
        showError(
          error.response?.data?.detail ||
            "Something went wrong. Please try again.",
        );
      } else {
        showError("An error occurred. Please try again later.");
      }
      setIsProcessing(false);
    }
  };

  const updateChatConversation = (inputText, response) => {
    if (isChatMissing) return;
    console.log("response", response);
    const newHistoryEntry = {
      query: inputText,
      response: response.response,
      isResponseCopied: false,
      showContext: false,
      contexts: response.context,
      showTable: false,
      tableData: response.tableData || [],
      csvFileId: response.csvFileId || null,
      csvFileName: response.csvFileName || null,
    };
    console.log("new history", newHistoryEntry);
    if (chatConversation.length === 0) {
      setFirstQuery(true);
    } else {
      setFirstQuery(false);
    }
    setChatConversation((prev) => [...prev, newHistoryEntry]);
  };

  if (isChatMissing) {
    return <ChatNotFoundPage />;
  }

  return (
    <div className={"flex flex-col h-full w-full overflow-hidden"}>
      <div
        ref={conversationRef}
        className="flex-1 overflow-y-auto flex flex-col gap-2.5 px-[32px] py-[16px] min-w-0"
        style={{ scrollbarWidth: "auto", scrollbarColor: "#CBD5E1 #F1F5F9" }}
      >
        {chatConversation.map((historyEntry, index) => {
          return (
            <div key={index} className={"flex flex-col gap-4 min-w-0"}>
              <div
                className={
                  "user-query flex flex-row-reverse items-start gap-2 min-w-0"
                }
              >
                <NewAvatar title={user?.fullName} />
                <div className="inline-block bg-[#374AF1] text-white py-[16px] px-[20px] rounded-[16px] ml-12 break-words whitespace-pre-wrap max-w-[85%] overflow-hidden">
                  {historyEntry.query}
                </div>
              </div>
              <div
                className={
                  "query-response flex gap-2 items-start max-w-[85%] min-w-0"
                }
              >
                <Image
                  src={"/assets/chat/AIbot.svg"}
                  className="rounded-full"
                  width={40}
                  height={40}
                  alt="Avatar"
                ></Image>
                <div className={"flex flex-col flex-1 min-w-0"}>
                  <div
                    className="bg-white text-[#475569] leading-6 text-justify py-[16px] px-[20px] break-words rounded-[16px] whitespace-pre-wrap overflow-hidden"
                    style={{
                      border: "1px solid var(--Gray-20, #E2E8F0)",
                      boxShadow: "none",
                    }}
                  >
                    {historyEntry.response?.length > 1 ? (
                      <ul className="list-disc pl-5">
                        {historyEntry.response.map((item, idx) => (
                          <li key={idx}>
                            {containsGraphData(item.response) ? (
                              <div>
                                {extractTextBeforeGraph(item.response) && (
                                  <div className="mb-4">
                                    <MarkdownViewer
                                      mdText={extractTextBeforeGraph(
                                        item.response,
                                      )}
                                    />
                                  </div>
                                )}
                                <GraphResponseComponent
                                  responseText={item.response}
                                />
                              </div>
                            ) : (
                              <MarkdownViewer mdText={item.response} />
                            )}
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <>
                        {containsGraphData(
                          historyEntry.response[0].response,
                        ) ? (
                          <div>
                            {extractTextBeforeGraph(
                              historyEntry.response[0].response,
                            ) && (
                              <div className="mb-4">
                                <MarkdownViewer
                                  mdText={extractTextBeforeGraph(
                                    historyEntry.response[0].response,
                                  )}
                                />
                              </div>
                            )}
                            <GraphResponseComponent
                              responseText={historyEntry.response[0].response}
                            />
                          </div>
                        ) : (
                          <MarkdownViewer
                            mdText={
                              historyEntry.response[0].response ||
                              "No response available"
                            }
                          />
                        )}
                      </>
                    )}
                  </div>
                  <div className={"flex gap-2 p-2 pt-3 response-icons"}>
                    {!(Array.isArray(historyEntry.response)
                      ? historyEntry.response.some((item) =>
                          containsGraphData(item.response),
                        )
                      : containsGraphData(
                          historyEntry.response?.[0]?.response ||
                            historyEntry.response,
                        )) && (
                      <>
                        {isSpeakingActive && currentSpeakingIdx === index ? (
                          <Image
                            className={"cursor-pointer"}
                            src={"/assets/chat/Stop.svg"}
                            onClick={() => stopTextToSpeech()}
                            alt={"Stop"}
                            width={"20"}
                            height={"20"}
                            title={"Stop Reading"}
                          />
                        ) : (
                          <Image
                            className={"cursor-pointer"}
                            src={"/assets/chat/speak.svg"}
                            onClick={() => {
                              const textToSpeak = Array.isArray(
                                historyEntry.response,
                              )
                                ? historyEntry.response
                                    .map((item) => item.response || "")
                                    .join(". ")
                                : historyEntry.response ||
                                  "No response available";
                              textToSpeech(textToSpeak, index, false);
                            }}
                            alt={"Read aloud"}
                            width={"20"}
                            height={"20"}
                            title={"Read Aloud"}
                          />
                        )}
                        <Image
                          className={`cursor-pointer`}
                          src={
                            historyEntry.isResponseCopied
                              ? "/assets/chat/tick.svg"
                              : "/assets/chat/copy.svg"
                          }
                          alt={"Copy"}
                          width={"20"}
                          onClick={() => handleCopyAction(index)}
                          height={"20"}
                          title={
                            historyEntry.isResponseCopied
                              ? "Copied!"
                              : "Copy to Clipboard!"
                          }
                        />
                      </>
                    )}
                    {historyEntry.contexts?.length > 0 && (
                      <Image
                        className={`cursor-pointer filter brightness-[70%] saturate-[180%] invert-[72%] sepia-[13%] hue-rotate-[180deg] ${
                          historyEntry.showContext ? "active" : ""
                        }`}
                        src={"/assets/chat/context.svg"}
                        alt={
                          historyEntry.showContext
                            ? "Hide Contexts"
                            : "View Contexts"
                        }
                        width={"20"}
                        height={"20"}
                        onClick={() => toggleContextView(index)}
                        title={
                          historyEntry.showContext
                            ? "Hide Contexts"
                            : "View Contexts"
                        }
                      />
                    )}
                    {historyEntry.tableData &&
                      historyEntry.tableData.length > 0 && (
                        <Image
                          className={`cursor-pointer ${
                            historyEntry.showTable ? "active" : ""
                          }`}
                          src={"/assets/chat/tableSvg.svg"}
                          alt={
                            historyEntry.showTable
                              ? "Hide Data Table"
                              : "View Data Table"
                          }
                          width={"20"}
                          height={"20"}
                          onClick={() => toggleTableView(index)}
                          title={
                            historyEntry.showTable
                              ? "Hide Data Table"
                              : "View Data Table"
                          }
                        />
                      )}
                  </div>
                  {historyEntry.showContext && (
                    <div ref={(el) => (contextElementRefs.current[index] = el)}>
                      <ChatSource contexts={historyEntry.contexts} />
                    </div>
                  )}
                  {historyEntry.showTable && (
                    <div ref={(el) => (tableElementRefs.current[index] = el)}>
                      <SqlDatatableComponent
                        tableData={historyEntry.tableData}
                        csvFileId={historyEntry.csvFileId}
                        csvFileName={historyEntry.csvFileName}
                      />
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
        {isProcessing && (
          <div className={"flex flex-col gap-2 w-full min-w-0"}>
            <div
              className={
                "user-query flex flex-row-reverse items-start gap-2 min-w-0"
              }
            >
              <NewAvatar title={user?.fullName} />
              <div
                className={
                  "bg-[#374AF1] text-white p-4 rounded-[16px] break-words whitespace-pre-wrap max-w-[85%] overflow-hidden"
                }
              >
                {lastQueryInput}
              </div>
            </div>
            <div
              className={
                "query-response flex gap-2 items-start max-w-[85%] min-w-0"
              }
            >
              <Image
                src={"/assets/chat/AIbot.svg"}
                width={"40"}
                height={"40"}
                alt={"Avatar"}
              ></Image>
              <div className={"flex flex-col flex-1 min-w-0"}>
                <div
                  className={
                    "bg-white text-[#475569] leading-6 text-justify p-5 rounded-3xl break-words overflow-hidden"
                  }
                >
                  <ThinkingIndicator isProcessing={isProcessing} />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
      <div className="max-w-[calc(100%-20px)] flex flex-col items-center gap-2 px-[32px] pt-4 pb-2">
        <div
          className={`border border-solid border-[#CBD5E1] relative w-full flex items-center px-2 py-1 bg-white rounded-full shadow-sm ${
            isListeningActive ? "animate-pulse border-blue-500" : ""
          }`}
        >
          <textarea
            ref={inputTextAreaRef}
            disabled={isProcessing}
            placeholder="Ask a question.."
            onChange={handleTextInputChange}
            onKeyDown={handleEnterKey}
            value={queryInput}
            className="w-full text-sm px-3 py-2 rounded-full focus:outline-none resize-none overflow-y-auto flex-grow"
            style={{
              whiteSpace: "pre-wrap",
              lineHeight: "15px",
              minHeight: "28px",
              maxHeight: "60px",
            }}
          />
          {isVoiceEnabled && (
            <button
              onClick={toggleMicrophone}
              disabled={isProcessing}
              className="flex items-center justify-center w-12 h-12"
            >
              <Image
                className={`cursor-pointer ${
                  isListeningActive ? "animate-pulse" : ""
                }`}
                src={
                  isListeningActive
                    ? "/assets/chat/Stop.svg"
                    : "/assets/chat/microphone.svg"
                }
                alt={isListeningActive ? "Stop Recording" : "Mic"}
                width={25}
                height={25}
              />
            </button>
          )}
          <button
            disabled={queryInput.trim() === "" || isProcessing}
            onClick={initiateChat}
            className={`flex items-center justify-center w-12 h-12 transition-opacity ${
              queryInput.trim() === "" ? "opacity-50 cursor-not-allowed" : ""
            }`}
          >
            <Image
              src="/assets/chat/send.svg"
              alt="Send"
              width={40}
              height={40}
            />
          </button>
        </div>
        <div className="text-[#94A3B8] text-sm text-center">
          The agent can make mistakes. Please double check the responses
        </div>
      </div>
    </div>
  );
};
export default ChatBoxComponentV2;
