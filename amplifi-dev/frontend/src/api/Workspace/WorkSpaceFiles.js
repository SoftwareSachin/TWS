import { http } from "..";

// get source connections lists ..
export const getSourceConnector = (workspaceId, page) => {
  return http.get(
    `/workspace/${workspaceId}/source?order=ascendent&page=${page?.page}&size=${page?.size}`,
  );
};

// delete source connection ..
export const deleteSourceConnector = (data) => {
  return http.delete(`/workspace/${data?.id}/source/${data?.sourceId}`);
};

//   get source details
export const getSourceConnectorDetails = (data, pagination, searchText) => {
  const searchParam = data.searchText ? `&search=${data.searchText}` : "";
  return http.get(
    `/workspace/${data.workspaceId}/source/${data.sourceId}/files_metadata?order_by=id&order=ascendent&page=${pagination.page}&size=${pagination.size}${searchParam}`,
  );
};

// get source details for edit
export const getSourceDetails = (data) => {
  return http.get(`/workspace/${data.workspaceId}/source/${data.sourceId}`);
};

// edit source details
export const editSource = (workspaceId, sourceId, payload) => {
  return http.put(`/workspace/${workspaceId}/source/${sourceId}`, payload);
};

// configure groove auto-detection
export const configureGrooveAutoDetection = (workspaceId, sourceId, config) => {
  return http.patch(
    `/workspace/${workspaceId}/source/${sourceId}/auto-detection`,
    config,
  );
};
