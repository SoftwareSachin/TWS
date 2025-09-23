import { http, httpV2 } from "..";

export const createWorkspace = (data) => {
  return http.post(`/organization/${data?.id}/workspace`, data?.body);
};

export const uploadFile = (data) => {
  return http.post(`/workspace/${data?.id}/file_upload`, data.body, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
};

export const sourceConnector = (data) => {
  return http.post(`/workspace/${data?.id}/source`, data?.body);
};

export const fetchDataSources = (workspaceId) => {
  return http.get(`/workspace/${workspaceId}/source`);
};

export const getWorkSpace = (orgId, page, searchStr = "") => {
  let queryString = `?order=ascendent&page=${page.page}&size=${page.size}`;
  if (searchStr) {
    queryString += `&search=${searchStr}`;
  }
  return http.get(`/organization/${orgId}/workspace${queryString}`);
};

export const updateWorkSpace = (data) => {
  return http.put(
    `/organization/${data.orgId}/workspace/${data.workSpaceId}`,
    data.body,
  );
};

export const deleteWorkSpace = (orgId, workspaceId) => {
  return http.delete(`/organization/${orgId}/workspace/${workspaceId}`);
};

export const addChunk = (datasetId, skip_files) => {
  const payload = {
    name: "string",
    metadata: {},
  };
  return httpV2.post(
    `/dataset/${datasetId}/ingest?skip_successful_files=${skip_files}`,
    payload,
  );
};

export const chunking_config = (payload, datasetId) => {
  return http.post(`/dataset/${datasetId}/chunking_config `, payload);
};

export const getStatus = (datasetId) => {
  return httpV2.get(`/dataset/${datasetId}/ingestion_status`);
};

export const testConnection = (data) => {
  return http.get(
    `/workspace/${data?.workSpaceId}/source/${data?.sourceId}/connection_status`,
  );
};

export const getSourceConnectorById = async (workspaceId, sourceId) => {
  console.log({ sourceId });
  const res = await http.get(`/workspace/${workspaceId}/source/${sourceId}`);
  return res.data?.data;
};
