import { http, httpV2 } from "@/api";
import { ChatAppPaginatedResponse, ChatAppResponse } from "@/types/ChatApp";
import { ListResults, Page, PaginatedResponse } from "@/types/Paginated";
import { ApiResponse } from "@/types/ApiResponse";
import { table } from "console";

export const createChatApp = async (data: any) => {
  return await httpV2.post(`/workspace/${data?.id}/chat_app`, data?.body);
};

export const getChatApp = async (workspaceId: string, page: Page) => {
  return await httpV2.get<ChatAppPaginatedResponse>(
    `/workspace/${workspaceId}/chat_apps?order=ascendent&page=${page.page}&size=${page.size}`,
  );
};

export const getLoggedInUserChatApps = async (data: any) => {
  return await http.get(
    `/my_chat_app?chat_app_type=unstructured_chat_app&order=ascendent&page=${data.page}&size=${data.size}`,
  );
};

export const deleteChatApp = async (data: any) => {
  return await http.delete(
    `/workspace/${data?.id}/chat_app/${data?.chatAppId}`,
  );
};

export const updateChatApp = async (data: any) => {
  return await httpV2.put(
    `/workspace/${data?.id}/chat_app/${data?.chatAppId}`,
    data?.body,
  );
};

export const getChatAppById = async (
  workspaceId: string,
  chatAppId: string,
) => {
  const response = await http.get<ApiResponse<ChatAppResponse>>(
    `/workspace/${workspaceId}/chat_app/${chatAppId}`,
  );
  return response?.data?.data;
};

export const getChatAppV2ById = async (
  workspaceId: string,
  chatAppId: string,
) => {
  const response = await httpV2.get<ApiResponse<ChatAppResponse>>(
    `/workspace/${workspaceId}/chat_app/${chatAppId}`,
  );
  return response?.data?.data;
};

export const createChatSession = async (chatAppId: string) => {
  try {
    const randomNum = Math.floor(Math.random() * 1000);
    const res = await http.post(`/chat_app/${chatAppId}/chat_session`, {
      title: "Untitled Chat " + randomNum,
    });
    console.log(res.data);
    return res.data?.data;
  } catch (error) {
    console.error("Error:", error);
  }
};

export const updateChatSession = (
  chatAppId: string,
  chatSessionId: string,
  payload: any,
) => {
  return http.put(
    `/chat_app/${chatAppId}/chat_session/${chatSessionId}`,
    payload,
  );
};

export const deleteChatSession = async (
  chatAppId: string,
  chatSessionId: string,
) => {
  return await http.delete(
    `/chat_app/${chatAppId}/chat_session/${chatSessionId}`,
  );
};

export const getChatResponse = async (
  { chatAppId, chatSessionID }: any,
  query: string,
) => {
  try {
    const res = await httpV2.post(
      `/chat_app/${chatAppId}/chat_session/${chatSessionID}/rag_generation`,
      {
        query,
        history: [], // Adjust this if needed
      },
    );
    const response = res.data?.data;
    // Removed sensitive chat response logging for security
    return {
      query,
      response: response?.chats?.answer || "No response received.",
      contexts: response?.chats?.file_contexts_aggregated || [],
      ssml: response?.chats?.ssml,
      plotGraph: response?.chats?.plotly_figure,
      sqlQuery: response?.chats?.generated_sql,
    };
  } catch (error) {
    console.error("Error:", error);
    return null;
  }
};

export const getChatResponseV2 = async (
  { chatAppId, chatSessionID, workspaceId }: any,
  query: string,
) => {
  try {
    const res = await httpV2.post(`/chat`, {
      chat_app_id: chatAppId,
      chat_session_id: chatSessionID,
      workspace_id: workspaceId,
      query,
    });

    const response = res.data;
    // Removed sensitive chat response logging for security

    let context: any[] = [];
    let tableData: any[] = [];
    let csvFileId: string | null = null;
    let csvFileName: string | null = null;
    let hasStructuredContext = false;
    response?.contexts?.forEach((c: any) => {
      const toolName = c?.tool_name;

      if (toolName === "perform_vector_search") {
        const unstructuredContext = c?.content?.aggregated || [];
        context = [...context, ...unstructuredContext];
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
      response: response?.responses?.length
        ? response?.responses
        : [{ response: "No response received." }],
      context,
      tableData: hasStructuredContext && tableData.length > 0 ? tableData : [],
      csvFileId,
      csvFileName,
      updatedSessionTitle: response?.updated_session_title || null,
    };
  } catch (error) {
    console.error("Error:", error);
    return null;
  }
};

export const getDataSets = async (workspaceId: string) => {
  try {
    const res = await http.get(`/workspace/${workspaceId}/dataset`);
    return res.data?.data?.items;
  } catch (error) {
    console.error("Error:", error);
    return [];
  }
};

export const downloadApi = async (
  file: any,
  setPreviewUrl: (url: string) => void,
) => {
  const response = await http.get(`files/${file.file_id}/download`, {
    responseType: "blob", // Ensures we receive binary data
    headers: {
      "Content-Type": "application/octet-stream",
    },
  });
  try {
    const blob = new Blob([response.data], { type: "image/jpeg" }); // or "image/png" depending on your image
    const blobUrl = window.URL.createObjectURL(blob);

    setPreviewUrl(blobUrl); // send it to your component's state

    return true;
  } catch (error) {
    console.error("Error previewing image:", error);
    return false;
  }
};

export const downloadFileApi = async (file: any) => {
  const response = await http.get(`files/${file.file_id}/download`, {
    responseType: "blob", // Ensures we receive binary data
    headers: {
      "Content-Type": "application/octet-stream",
    },
  });
  try {
    const blob = new Blob([response.data], { type: "application/pdf" }); // Convert response to Blob
    const blobUrl = window.URL.createObjectURL(blob); // Create object URL

    const link = document.createElement("a");
    link.href = blobUrl;
    link.download = file.file_name;
    document.body.appendChild(link);
    link.click(); // Trigger download
    document.body.removeChild(link);
    window.URL.revokeObjectURL(blobUrl);
    return true;
  } catch (error) {
    console.error("Error downloading PDF:", error);
    return false;
  }
};

export const deleteFileApi = async (file: any) => {
  return http.delete(`/workspace/${file.workspaceId}/file/${file.file_id}`);
};

export const getChatSessionsByChatAppId = async (
  chatAppId: string,
  page = 0,
) => {
  const res = await http.get<ApiResponse<PaginatedResponse<any>>>(
    `/chat_app/${chatAppId}/chat_session?order=descendent&size=8&page=${page}`,
  );
  return res?.data?.data;
};

export const getChatHistory = async (
  chatAppId: string,
  chatSessionId: string,
) => {
  const res = await http.get<ApiResponse<ListResults<any>>>(
    `/chat_app/${chatAppId}/chat_session/${chatSessionId}/chat_history`,
  );
  return res?.data?.data;
};
