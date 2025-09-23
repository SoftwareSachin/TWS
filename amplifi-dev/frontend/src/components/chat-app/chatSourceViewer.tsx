import { useEffect, useState, useCallback, useRef } from "react";
import { showError, showInfo } from "@/utils/toastUtils";
import { downloadFileApi } from "@/api/chatApp";
import { PreviewImage } from "@/components/ui/preview-image";
import { convertTableToText, extractTableData } from "../utility/SanitizeHtml";
import { MarkdownViewer } from "../ui/markdown-viewer";

// Module-level variable to track chunk navigation
let chunkNavigationInProgress = false;

// Export function to check if chunk navigation is in progress
export const isChunkNavigationInProgress = () => chunkNavigationInProgress;

interface FileType {
  file_id: string;
  file_name: string;
}

interface TextChunk {
  text: string;
  chunk_id: string;
  page_number: number;
  search_score: number;
  match_type: string;
  table_html?: string;
}

interface ChatSourceType {
  file_id: string;
  file_name: string;
  texts: TextChunk[];
}

interface ChatSourceViewerProps {
  chatSource: ChatSourceType;
}

export const ChatSourceViewer = ({ chatSource }: ChatSourceViewerProps) => {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isImage, setIsImage] = useState<boolean>(false);
  const [currentTextIndex, setCurrentTextIndex] = useState<number>(0);
  const chunkNavigationRef = useRef(false);

  // Set a flag to indicate chunk navigation is happening
  useEffect(() => {
    if (chunkNavigationRef.current) {
      chunkNavigationInProgress = true;
      const timer = setTimeout(() => {
        chunkNavigationInProgress = false;
        chunkNavigationRef.current = false;
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [currentTextIndex]);

  const downloadFile = async (file: FileType) => {
    showInfo("Download started");
    try {
      const success = await downloadFileApi(file);
      if (!success) {
        showError("Download error.");
      }
    } catch (error) {
      showError("Download failed.");
    }
  };

  const checkIsImage = (file_name: string): boolean => {
    return /\.(jpg|jpeg|png)$/i.test(file_name);
  };

  const handleNext = useCallback(() => {
    if (currentTextIndex < chatSource.texts.length - 1) {
      chunkNavigationRef.current = true;
      setCurrentTextIndex(currentTextIndex + 1);
    }
  }, [currentTextIndex, chatSource.texts.length]);

  const handlePrevious = useCallback(() => {
    if (currentTextIndex > 0) {
      chunkNavigationRef.current = true;
      setCurrentTextIndex(currentTextIndex - 1);
    }
  }, [currentTextIndex]);

  useEffect(() => {
    if (chatSource?.file_name) {
      if (checkIsImage(chatSource.file_name)) {
        setIsImage(true);
        setPreviewUrl(null);
      } else {
        setIsImage(false);
        setPreviewUrl(null);
      }
    }
    setCurrentTextIndex(0);
  }, [chatSource]);

  const currentText = chatSource.texts[currentTextIndex];

  const tableData = currentText?.table_html
    ? extractTableData(currentText.table_html)
    : null;

  return (
    <div className="flex flex-col flex-grow">
      <div className="bg-white shadow-[0px_1px_3px_0px_rgba(0,0,0,0.16)] text-[#475569] leading-6 text-justify p-5 rounded-3xl">
        <div className="flex flex-col gap-3">
          {isImage && (
            <PreviewImage
              file={{
                file_id: chatSource.file_id,
                file_name: chatSource.file_name,
              }}
            />
          )}
          <div className="flex items-center justify-between">
            <div
              className="text-red-700 cursor-pointer hover:underline"
              onClick={() =>
                downloadFile({
                  file_id: chatSource.file_id,
                  file_name: chatSource.file_name,
                })
              }
            >
              {chatSource.file_name}
            </div>
            {chatSource.texts.length > 1 && (
              <div className="flex items-center gap-2">
                <button
                  onClick={handlePrevious}
                  disabled={currentTextIndex === 0}
                  className={`px-3 py-1 rounded-md transition-colors ${
                    currentTextIndex === 0
                      ? "bg-gray-200 text-gray-400 cursor-not-allowed"
                      : "bg-blue-500 text-white hover:bg-blue-600"
                  }`}
                >
                  {"<"}
                </button>
                {currentText && (
                  <span className="font-semibold text-gray-500">
                    {`${currentTextIndex + 1}/${
                      chatSource.texts.length
                    } Chunks`}
                  </span>
                )}
                <button
                  onClick={handleNext}
                  disabled={currentTextIndex === chatSource.texts.length - 1}
                  className={`px-3 py-1 rounded-md transition-colors ${
                    currentTextIndex === chatSource.texts.length - 1
                      ? "bg-gray-200 text-gray-400 cursor-not-allowed"
                      : "bg-blue-500 text-white hover:bg-blue-600"
                  }`}
                >
                  {">"}
                </button>
              </div>
            )}
          </div>
          {currentText && (
            <div className="text-gray-700 text-sm" key={currentTextIndex}>
              {tableData ? (
                <div className="overflow-auto mt-4">
                  <div className="border border-slate-300 rounded-lg">
                    {tableData.map((row, rowIndex) => {
                      const isHeader = rowIndex === 0;
                      return (
                        <div
                          key={rowIndex}
                          className={`flex ${
                            isHeader
                              ? "bg-slate-100 font-semibold border-b-2 border-slate-400"
                              : rowIndex !== tableData.length - 1
                                ? "border-b border-slate-200"
                                : ""
                          }`}
                        >
                          {row.map((cell, cellIndex) => (
                            <div
                              key={cellIndex}
                              className={`flex-1 px-4 py-2 text-left ${
                                cellIndex !== row.length - 1
                                  ? "border-r border-slate-200"
                                  : ""
                              }`}
                            >
                              {cell || "-"}
                            </div>
                          ))}
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : currentText.table_html ? (
                <div className="mt-4 p-4 bg-gray-50 rounded-lg font-mono text-xs whitespace-pre-wrap">
                  {convertTableToText(currentText.table_html)}
                </div>
              ) : (
                <p className="whitespace-pre-wrap">
                  <MarkdownViewer mdText={currentText.text} />
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
