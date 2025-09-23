import { ChatSourceViewer } from "@/components/chat-app/chatSourceViewer";
import { useState, useEffect, useRef } from "react";
import CodeEditorViewer from "../ui/codeEditorViewer";

export const ChatSource = ({ contexts }) => {
  const [chatSource, setChatSource] = useState(null);
  const [previousChatSource, setPreviousChatSource] = useState(null);
  const chatSourceViewerRef = useRef(null);
  const sqlViewerRef = useRef(null);

  useEffect(() => {
    if (chatSource && !previousChatSource) {
      const ref =
        chatSource?.generated_sql && sqlViewerRef.current
          ? sqlViewerRef.current
          : chatSourceViewerRef.current;

      if (ref) {
        setTimeout(() => {
          ref.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }, 100);
      }
    }
    setPreviousChatSource(chatSource);
  }, [chatSource, previousChatSource]);

  const vectorContexts =
    contexts?.filter(
      (ctx) => "file_name" in ctx && !("generated_sql" in ctx),
    ) || [];
  const sqlContexts = contexts?.filter((ctx) => "generated_sql" in ctx) || [];

  console.log("Vector contexts:", vectorContexts);
  console.log("SQL contexts:", sqlContexts);

  return (
    <div className="flex flex-col flex-grow chat-source-container">
      <span className="font-bold text-[14px] mb-2">Context Retrieved</span>

      <div className="flex flex-col flex-grow">
        <div className="flex-grow flex items-start justify-start flex-wrap gap-2 mb-2">
          {vectorContexts.map((context, index) => (
            <span
              key={`vector-${context.file_id || index}`}
              onClick={() => setChatSource(context)}
              className={`cursor-pointer inline-flex items-center px-[16px] py-[12px] text-sm font-medium text-gray-700 bg-custom-contextBgColor border border-[#E2E8F0] rounded-[16px] 
                hover:bg-custom-contextHoverButtonColor ${
                  chatSource?.file_id === context?.file_id
                    ? "!bg-custom-contextHoverButtonColor"
                    : ""
                }`}
            >
              {context.file_name}
            </span>
          ))}
          {sqlContexts.map((context, index) => (
            <span
              key={`sql-${index}`}
              onClick={() => setChatSource(context)}
              className={`cursor-pointer inline-flex items-center px-[16px] py-[12px] text-sm font-medium text-gray-700 bg-custom-contextBgColor border border-[#E2E8F0] rounded-[16px] 
                hover:bg-custom-contextHoverButtonColor ${
                  chatSource === context
                    ? "!bg-custom-contextHoverButtonColor"
                    : ""
                }`}
            >
              SQL Query {index + 1}
            </span>
          ))}
        </div>
        {chatSource && chatSource.file_name && !chatSource.generated_sql && (
          <div ref={chatSourceViewerRef}>
            <ChatSourceViewer chatSource={chatSource} />
          </div>
        )}
        {chatSource && chatSource.generated_sql && (
          <div className="flex flex-col flex-grow" ref={sqlViewerRef}>
            <div className="mb-2">
              <span className="text-sm font-medium text-gray-700">
                Generated SQL Query:
              </span>
            </div>
            <CodeEditorViewer value={chatSource.generated_sql} language="sql" />
          </div>
        )}
      </div>
    </div>
  );
};
