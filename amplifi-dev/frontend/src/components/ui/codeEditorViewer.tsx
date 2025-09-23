import React from "react";
import CodeEditor from "@uiw/react-textarea-code-editor";

interface CodeEditorViewerProps {
  value: string;
  language: string;
  readMode: boolean;
}

const CodeEditorViewer: React.FC<CodeEditorViewerProps> = ({
  value,
  language,
  readMode = true,
}) => {
  return (
    <div className="flex flex-col flex-grow">
      <CodeEditor
        value={value}
        language={language}
        padding={15}
        readOnly={readMode}
        data-color-mode="light"
        style={{
          fontSize: 12,
          backgroundColor: "white",
          fontFamily:
            'ui-monospace, SFMono-Regular, "SF Mono", Consolas, "Liberation Mono", Menlo, monospace',
          borderRadius: "10px",
          border: "1px solid #E2E8F0",
        }}
      />
    </div>
  );
};

export default CodeEditorViewer;
