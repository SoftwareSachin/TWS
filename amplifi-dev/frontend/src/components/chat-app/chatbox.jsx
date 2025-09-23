"use client";
import React, { useEffect, useRef, useState } from "react";
import Image from "next/image";
import * as SpeechSDK from "microsoft-cognitiveservices-speech-sdk";
import { showError, showSuccess } from "@/utils/toastUtils";
import { recognizerConfig, speechConfig } from "@/utils/speechConfig";
import { useChat } from "@/context_api/chatContext";
import { getChatAppById, getChatHistory, getChatResponse } from "@/api/chatApp";
import { ChatSource } from "@/components/chat-app/chatSource";
import { constants } from "@/lib/constants";
import SqlDatatable from "@/components/chat-app/sqlDatatable";
import PlotlyGraphComponent from "@/components/chat-app/plotlyGraph";
import { MarkdownViewer } from "@/components/ui/markdown-viewer";
import { MAX_QUERY_LENGTH } from "@/lib/file-constants";
import ChatNotFoundPage from "@/components/empty-screens/chatNotFound";
import { useUser } from "@/context_api/userContext";
import NewAvatar from "../ui/newAvatar";

const ChatBoxComponent = () => {
  const [query, setQuery] = useState("");
  const [lastQuery, setLastQuery] = useState("");
  const [isListening, setIsListening] = useState(false);
  const [isVoiceAllowed, setIsVoiceAllowed] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [currentSpeakingIndex, setCurrentSpeakingIndex] = useState(null);
  const [isTTSInitializing, setIsTTSInitializing] = useState(false);
  const [isChatNotFound, setIsChatNotFound] = useState(false);
  const divRef = useRef(null);
  const recognizerRef = useRef(null);
  const synthesizerRef = useRef(null);
  const isInteractingWithPlot = useRef(false);
  const observerRef = useRef(null);
  const textareaRef = useRef(null);
  const { user } = useUser();

  const {
    chatSessionID,
    chatSessionName,
    setChatResponse,
    setChatSource,
    chatAppId,
    setFirstQuery,
    setChatAppName,
    chatAppType,
    setChatAppType,
  } = useChat();

  useEffect(() => {
    if (chatAppId) {
      getChatApp(chatAppId);
    } else {
      console.log("Skipping getChatApp - missing chatAppId or workspaceId");
    }
  }, [chatAppId]);

  // Consolidated useEffect for handling chat session changes
  useEffect(() => {
    if (chatSessionID && chatAppType) {
      getChatSessionHistory();
    } else {
      // Clear chat history when no session is selected
      setChatHistory([]);
      setFirstQuery(false);
    }
  }, [chatSessionID, chatAppType]);

  // Cleanup function for component unmount
  useEffect(() => {
    return () => {
      stopTTS();
    };
  }, []);

  useEffect(() => {
    const scrollToBottom = () => {
      if (divRef.current && !isHoveringPlot.current) {
        divRef.current.scrollTo({
          top: divRef.current.scrollHeight,
          behavior: "smooth",
        });
      } else {
        console.log("🚫 Skipping scroll because mouse is on plot");
      }
    };

    scrollToBottom();

    const observer = new MutationObserver(scrollToBottom);

    if (divRef.current) {
      observer.observe(divRef.current, {
        childList: true,
        subtree: true,
      });
      observerRef.current = observer;
    }

    return () => {
      observer.disconnect();
    };
  }, [chatHistory, lastQuery]);

  // Scroll textarea to bottom when query updates during speech
  useEffect(() => {
    if (isListening && textareaRef.current) {
      textareaRef.current.scrollTop = textareaRef.current.scrollHeight;
    }
  }, [query, isListening]);

  const handleMic = () => {
    if (isListening) {
      stopSpeechRecognition();
    } else {
      stopTTS(); // Stop any playing audio before starting recognition
      startSpeechRecognition();
    }
  };
  const handleCopy = async (selectedIndex) => {
    if (chatHistory[selectedIndex].isResponseCopied) return; // Prevent multiple clicks
    try {
      const scrollPosition = divRef.current.scrollTop;
      const newChatHistory = chatHistory.map((item, index) =>
        selectedIndex === index
          ? { ...item, isResponseCopied: !item.isResponseCopied }
          : item,
      );
      setChatHistory(newChatHistory);
      await navigator.clipboard.writeText(chatHistory[selectedIndex].response);
      showSuccess("Copied to clipboard!");
      divRef.current.scrollTop = scrollPosition;
      setTimeout(() => {
        setChatHistory((prevItems) =>
          prevItems.map((item, index) =>
            selectedIndex === index
              ? { ...item, isResponseCopied: false }
              : item,
          ),
        );
      }, 6000);
    } catch (err) {
      console.log("Failed to copy!", err);
    }
  };

  const validateQueryLength = (inputText) => {
    if (inputText.length > MAX_QUERY_LENGTH) {
      showError(`The input you submitted was too long.`);
      return false;
    }
    return true;
  };

  const showContext = (selectedIndex) => {
    setChatHistory((prevItems) =>
      prevItems.map((item, index) => {
        if (selectedIndex === index) {
          return { ...item, showContext: !item.showContext };
        }
        return item;
      }),
    );
  };

  const getChatSessionHistory = async () => {
    const res = await getChatHistory(chatAppId, chatSessionID);
    const histories = res.items.map((r) => {
      if (chatAppType === constants.UNSTRUCTURED_CHAT_APP) {
        return {
          query: r.user_query,
          response: r.llm_response,
          contexts: [],
          isResponseCopied: false,
          showContext: false,
          isSql: false,
        };
      } else if (constants.SQL_CHAT_APP === chatAppType) {
        const sqlResponse = JSON.parse(r.llm_response);
        try {
          console.log(sqlResponse);
          return {
            query: r.user_query,
            tableData: JSON.parse(sqlResponse.answer) || [],
            graphPlot:
              sqlResponse?.plotly_figure &&
              JSON.parse(sqlResponse.plotly_figure),
            sqlGeneratedQuery: sqlResponse.generated_sql,
            isResponseCopied: false,
            showGraph: false,
            isSql: true,
          };
        } catch (e) {
          console.error(e);
          return {
            query: r.user_query,
            response: sqlResponse.answer,
            showContext: false,
            isSql: false,
          };
        }
      }
    });
    console.log(histories);
    setChatHistory(histories);
  };

  const getChatApp = async (chatAppId) => {
    if (chatAppId) {
      try {
        // bug on backend as no workspace id check is implemented
        const response = await getChatAppById(chatAppId, chatAppId);
        setChatAppType(response?.chat_app_type);
        setChatAppName(response?.name);
        setIsVoiceAllowed(!!response?.voice_enabled);
      } catch (error) {
        console.error(
          "Error fetching chat app:",
          error.response?.data || error.message,
        );
        setIsChatNotFound(true);
      }
    }
  };

  const partialRef = useRef("");

  const startSpeechRecognition = () => {
    if (isChatNotFound) return;
    setIsListening(true);
    partialRef.current = "";
    let hasSubmitted = false;
    let pauseTimer = null;
    let lastSpeechTime = Date.now();

    const recognizer = new SpeechSDK.SpeechRecognizer(
      speechConfig,
      recognizerConfig,
    );

    // interim results
    recognizer.recognizing = (_, event) => {
      lastSpeechTime = Date.now();
      if (pauseTimer) {
        clearTimeout(pauseTimer);
      }
      setQuery(partialRef.current + " " + event.result.text);
    };

    // final result for each phrase
    recognizer.recognized = (_, evt) => {
      if (evt.result.reason === SpeechSDK.ResultReason.RecognizedSpeech) {
        partialRef.current += " " + evt.result.text;
        setQuery(partialRef.current.trim());

        // Clear any existing timer
        if (pauseTimer) {
          clearTimeout(pauseTimer);
        }

        // Set new timer to check for end of speech
        pauseTimer = setTimeout(() => {
          // If it's been more than 3 seconds since last speech
          if (Date.now() - lastSpeechTime > 3000) {
            const finalText = partialRef.current.trim();
            if (finalText && !hasSubmitted) {
              if (!validateQueryLength(finalText)) {
                stopSpeechRecognition();
                return;
              }
              hasSubmitted = true;
              // Stop recognition before submitting to prevent further listening
              recognizer.stopContinuousRecognitionAsync(
                () => {
                  getQueryResponse(finalText, true);
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
      if (pauseTimer) {
        clearTimeout(pauseTimer);
      }
      const finalText = partialRef.current.trim();
      if (finalText && !hasSubmitted) {
        hasSubmitted = true;
        await getQueryResponse(finalText, true);
      }
      stopSpeechRecognition();
    };

    recognizer.canceled = () => {
      if (pauseTimer) {
        clearTimeout(pauseTimer);
      }
      setIsListening(false);
      if (recognizerRef.current) {
        recognizerRef.current = null;
      }
    };

    recognizer.startContinuousRecognitionAsync();
    recognizerRef.current = recognizer;
  };

  const stopSpeechRecognition = () => {
    if (recognizerRef.current) {
      try {
        recognizerRef.current.stopContinuousRecognitionAsync(
          () => {
            setIsListening(false);
            if (recognizerRef.current) {
              recognizerRef.current = null;
            }
          },
          (error) => {
            console.error("Error stopping recognition:", error);
            setIsListening(false);
            if (recognizerRef.current) {
              recognizerRef.current = null;
            }
          },
        );
      } catch (error) {
        console.error("Error in stopSpeechRecognition:", error);
        setIsListening(false);
        if (recognizerRef.current) {
          recognizerRef.current = null;
        }
      }
    }
  };

  const inputTextChangeHandler = (event) => {
    setQuery(event.target.value);
  };

  const queryEnterHandler = (event) => {
    if (event.key === "Enter" && query.trim() !== "" && !isChatNotFound) {
      event.preventDefault(); // Prevent default behavior (new line)

      if (!validateQueryLength(query)) {
        return;
      }

      stopTTS(); // Stop any playing audio before sending query
      getQueryResponse(query, false); // Call backend API
    }
  };

  const chatWithBot = () => {
    if (query.trim() === "" || isChatNotFound) return;

    if (!validateQueryLength(query)) {
      return;
    }

    stopTTS(); // Stop any playing audio before sending query
    getQueryResponse(query, false);
  };

  let audioPlayer = useRef(null);
  const ttsForResponse = (text, index, isSsml = false) => {
    if (isTTSInitializing || isChatNotFound) return; // Prevent double click
    setIsTTSInitializing(true);

    // Always stop any existing audio before starting a new one
    stopTTS();

    if (!text || typeof text !== "string" || text.trim() === "") {
      console.error("Invalid text input for TTS:", text);
      showError("Cannot read aloud: No valid text provided");
      setIsTTSInitializing(false);
      return;
    }

    try {
      audioPlayer.current = new SpeechSDK.SpeakerAudioDestination();

      audioPlayer.current.onAudioStart = () => {
        setIsSpeaking(true);
        setCurrentSpeakingIndex(index);
        setIsTTSInitializing(false);
      };

      audioPlayer.current.onAudioEnd = () => {
        setIsSpeaking(false);
        setCurrentSpeakingIndex(null);
        setIsTTSInitializing(false);
        cleanupTTS();
      };

      const audioConfig = SpeechSDK.AudioConfig.fromSpeakerOutput(
        audioPlayer.current,
      );
      synthesizerRef.current = new SpeechSDK.SpeechSynthesizer(
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

      synthesizerRef.current[speakMethod](
        finalText,
        (result) => {
          if (result.reason === SpeechSDK.ResultReason.Canceled) {
            console.error("Synthesis canceled:", result.errorDetails);
            setIsTTSInitializing(false);
          }
        },
        (error) => {
          console.error("Synthesis error:", error);
          setIsSpeaking(false);
          setCurrentSpeakingIndex(null);
          setIsTTSInitializing(false);
          cleanupTTS();
        },
      );
    } catch (error) {
      console.error("Error initializing speech synthesis:", error);
      setIsSpeaking(false);
      setCurrentSpeakingIndex(null);
      setIsTTSInitializing(false);
      cleanupTTS();
    }
  };

  const stopTTS = () => {
    try {
      if (audioPlayer.current) {
        audioPlayer.current.pause(); // Pause the audio
        audioPlayer.current.close(); // Then close it
        audioPlayer.current = null;
      }
      if (synthesizerRef.current) {
        synthesizerRef.current.close();
        synthesizerRef.current = null;
      }
      setIsSpeaking(false);
      setCurrentSpeakingIndex(null);
      setIsTTSInitializing(false);
    } catch (error) {
      console.error("Error stopping TTS:", error);
    }
  };

  const cleanupTTS = () => {
    try {
      if (audioPlayer.current) {
        audioPlayer.current.close();
        audioPlayer.current = null;
      }
      if (synthesizerRef.current) {
        synthesizerRef.current.close();
        synthesizerRef.current = null;
      }
    } catch (error) {
      console.error("Error cleaning up TTS:", error);
    }
  };
  const isHoveringPlot = useRef(false);
  useEffect(() => {
    const container = divRef.current;
    if (!container) return;

    const handleMouseMove = (e) => {
      const target = e.target;
      const isOnPlot = !!target.closest(".js-plotly-plot");

      if (isOnPlot && !isHoveringPlot.current) {
        isHoveringPlot.current = true;
      } else if (!isOnPlot && isHoveringPlot.current) {
        isHoveringPlot.current = false;
      }
    };

    container.addEventListener("mousemove", handleMouseMove);

    return () => {
      container.removeEventListener("mousemove", handleMouseMove);
    };
  }, []);

  useEffect(() => {
    return () => {
      stopTTS();
    };
  }, []);

  const resetResponse = () => {
    setQueryResponse(null);
    setChatResponse(null);
    setChatSource(null);
    setQuery("");
    setChatHistory([]);
    setFirstQuery(false);
  };

  const clearChatHistory = () => {
    setChatHistory([]);
    setFirstQuery(false);
    setQuery("");
    setLastQuery("");
  };

  // Function to call the backend API
  const getQueryResponse = async (inputText, isAudio) => {
    if (isChatNotFound) return;

    if (!validateQueryLength(inputText)) {
      return;
    }

    try {
      setIsLoading(true);
      setLastQuery(inputText);
      setQuery("");
      const queryResponse = await getChatResponse(
        { chatAppId, chatSessionID },
        inputText,
      );
      setIsLoading(false);
      setChatResponse(queryResponse);
      updateChatHistory(inputText, queryResponse);
    } catch (error) {
      console.error("Error:", error);
      if (error.response?.status === 404) {
        setIsChatNotFound(true);
      } else if (error.response?.status === 500) {
        showError(
          error.response?.data?.detail ||
            "Something went wrong. Please try again.",
        );
      } else {
        showError("An error occurred. Please try again later.");
      }
      setIsLoading(false);
    }
  };

  const updateChatHistory = (inputText, response) => {
    if (isChatNotFound) return;

    let newHistory;
    if (chatAppType === constants.UNSTRUCTURED_CHAT_APP) {
      newHistory = {
        query: inputText,
        response: response.response,
        contexts: response.contexts,
        isResponseCopied: false,
        ssml: response.ssml,
        showContext: false,
        isSql: false,
      };
    } else {
      console.log(response);
      try {
        newHistory = {
          query: inputText,
          tableData: JSON.parse(response.response) || [],
          graphPlot: response?.plotGraph && JSON.parse(response.plotGraph),
          sqlGeneratedQuery: response.sqlQuery,
          isResponseCopied: false,
          showGraph: false,
          isSql: true,
        };
      } catch (e) {
        console.error(e);
        newHistory = {
          query: inputText,
          response: response.response,
          showContext: false,
          isSql: false,
        };
      }
    }
    if (chatHistory.length === 0) {
      setFirstQuery(true);
    } else {
      setFirstQuery(false);
    }
    setChatHistory((prev) => [...prev, newHistory]);
  };

  if (isChatNotFound) {
    return <ChatNotFoundPage />;
  }
  return (
    <div className="flex flex-col h-full">
      {/* Chat Messages Scrollable Area */}
      <div
        ref={divRef}
        className="flex-1 overflow-y-auto flex flex-col gap-2.5 px-[32px] py-[16px]"
        style={{
          scrollbarWidth: "auto",
          scrollbarColor: "#CBD5E1 #F1F5F9",
        }}
      >
        {chatHistory.map((history, index) => (
          <div key={index} className="flex flex-col gap-4">
            {/* User Query */}
            <div className="user-query flex flex-row-reverse items-start gap-2">
              <NewAvatar title={user?.fullName} />
              <div className="inline-block bg-[#374AF1] text-white py-[16px] px-[20px] rounded-[16px] ml-12 break-all whitespace-pre-wrap max-w-[85%]">
                {history.query}
              </div>
            </div>

            {/* Bot Response */}
            <div className="query-response flex gap-2 items-start max-w-[85%]">
              <Image
                src="/assets/chat/AIbot.svg"
                className="rounded-full"
                width={40}
                height={40}
                alt="Avatar"
              />
              <div className="flex flex-col">
                {/* Markdown / SQL / Plotly */}
                {!history.isSql && (
                  <div
                    className="bg-white text-[#475569] leading-6 text-justify py-[16px] px-[20px] rounded-[16px] "
                    style={{
                      border: "1px solid var(--Gray-20, #E2E8F0)",
                      boxShadow: "none",
                    }}
                  >
                    <MarkdownViewer mdText={history.response} />
                  </div>
                )}

                {history.isSql && (
                  <div className="mb-2">
                    <SqlDatatable tableData={history.tableData || []} />
                  </div>
                )}

                {history.graphPlot && (
                  <PlotlyGraphComponent chartData={history.graphPlot} />
                )}

                {/* Icons */}
                <div className="flex gap-2 p-2 pt-3 response-icons">
                  {!history.isSql && (
                    <>
                      {isSpeaking && currentSpeakingIndex === index ? (
                        <Image
                          className="cursor-pointer"
                          src="/assets/chat/Stop.svg"
                          onClick={stopTTS}
                          alt="Stop"
                          width={20}
                          height={20}
                          title="Stop Reading"
                        />
                      ) : (
                        <Image
                          className="cursor-pointer"
                          src="/assets/chat/speak.svg"
                          onClick={() =>
                            ttsForResponse(
                              history.response || "No response available",
                              index,
                              false,
                            )
                          }
                          alt="Read aloud"
                          width={20}
                          height={20}
                          title="Read Aloud"
                        />
                      )}
                    </>
                  )}

                  {!history.isSql && (
                    <Image
                      className="cursor-pointer"
                      src={
                        history.isResponseCopied
                          ? "/assets/chat/tick.svg"
                          : "/assets/chat/copy.svg"
                      }
                      alt="Copy"
                      width={20}
                      height={20}
                      onClick={() => handleCopy(index)}
                      title={
                        history.isResponseCopied
                          ? "Copied!"
                          : "Copy to Clipboard!"
                      }
                    />
                  )}

                  {history.contexts?.length > 0 && (
                    <Image
                      className={`cursor-pointer filter brightness-[70%] saturate-[180%] invert-[72%] sepia-[13%] hue-rotate-[180deg] ${
                        history.showContext ? "active" : ""
                      }`}
                      src="/assets/chat/context.svg"
                      alt={
                        history.showContext ? "Hide Contexts" : "View Contexts"
                      }
                      width={20}
                      height={20}
                      onClick={() => showContext(index)}
                      title={
                        history.showContext ? "Hide Contexts" : "View Contexts"
                      }
                    />
                  )}
                </div>

                {/* Context */}
                {history.showContext && (
                  <ChatSource contexts={history.contexts} />
                )}
              </div>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex flex-col gap-2 w-full">
            <div className="user-query flex flex-row-reverse items-start gap-2">
              <NewAvatar title={user?.fullName} />
              <div className="bg-[#374AF1] text-white p-4 rounded-[16px]">
                {lastQuery}
              </div>
            </div>
            <div className="query-response flex gap-2 items-start">
              <Image
                src="/assets/chat/AIbot.svg"
                width={40}
                height={40}
                alt="Avatar"
              />
              <div className="flex flex-col">
                {/* <span className="font-semibold">System</span> */}
                <div className="bg-white text-[#475569] leading-6 text-justify p-5 rounded-3xl">
                  Loading...
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="max-w-[calc(100%-20px)] flex flex-col items-center gap-2 px-[32px] pt-4 pb-2">
        <div
          className={`border border-solid border-[#CBD5E1] relative w-full flex items-center px-2 py-1 bg-white rounded-full shadow-sm ${
            isListening ? "animate-pulse border-blue-500" : ""
          }`}
        >
          <textarea
            ref={textareaRef}
            disabled={isLoading}
            placeholder="Ask a question.."
            onChange={inputTextChangeHandler}
            onKeyDown={queryEnterHandler}
            value={query}
            className="w-full text-sm px-3 py-2 rounded-full focus:outline-none resize-none overflow-y-auto flex-grow"
            style={{
              minHeight: "28px",
              maxHeight: "60px",
              whiteSpace: "pre-wrap",
              lineHeight: "15px",
            }}
          />

          {isVoiceAllowed && (
            <button
              onClick={handleMic}
              disabled={isLoading}
              className="flex items-center justify-center w-12 h-12"
            >
              <Image
                className={`cursor-pointer ${
                  isListening ? "animate-pulse" : ""
                }`}
                src={
                  isListening
                    ? "/assets/chat/Stop.svg"
                    : "/assets/chat/microphone.svg"
                }
                alt={isListening ? "Stop Recording" : "Mic"}
                width={25}
                height={25}
              />
            </button>
          )}

          <button
            disabled={query.trim() === "" || isLoading}
            onClick={chatWithBot}
            className={`flex items-center justify-center w-12 h-12 transition-opacity ${
              query.trim() === "" ? "opacity-50 cursor-not-allowed" : ""
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
          The bot can make mistakes. Please double check the responses.
        </div>
      </div>
    </div>
  );
};

export default ChatBoxComponent;
