import React, { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { MarkdownViewerProps } from "@/types/props/ChatAppProps";

export const MarkdownViewer: React.FC<MarkdownViewerProps> = ({ mdText }) => {
  const [htmlContent, setHtmlContent] = useState("");

  // const convertToHtml = async (markdownText: string): Promise<string> => {
  //   return (await remark().use(html).process(markdownText)).value as string;
  // };
  //
  // const handleMdText = async () => {
  //   setHtmlContent(await convertToHtml(mdText));
  // };
  // useEffect(() => {
  //   handleMdText();
  // }, [mdText]);

  // Custom components for table styling
  const components = {
    table: ({ children, ...props }: any) => (
      <div className="my-4 overflow-x-auto border border-gray-300 rounded-lg shadow-sm">
        <table className="min-w-full border-collapse bg-white" {...props}>
          {children}
        </table>
      </div>
    ),
    thead: ({ children, ...props }: any) => (
      <thead className="bg-gray-50" {...props}>
        {children}
      </thead>
    ),
    tbody: ({ children, ...props }: any) => (
      <tbody className="divide-y divide-gray-200" {...props}>
        {children}
      </tbody>
    ),
    tr: ({ children, ...props }: any) => (
      <tr className="hover:bg-gray-50" {...props}>
        {children}
      </tr>
    ),
    th: ({ children, ...props }: any) => (
      <th
        className="px-4 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider border-b border-gray-200"
        {...props}
      >
        {children}
      </th>
    ),
    td: ({ children, ...props }: any) => (
      <td
        className="px-4 py-3 text-sm text-gray-900 border-b border-gray-100"
        {...props}
      >
        {children}
      </td>
    ),
  };

  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
      {mdText}
    </ReactMarkdown>
  );
};
