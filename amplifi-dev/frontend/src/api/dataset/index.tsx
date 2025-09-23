import { http, httpV2 } from "..";
import { Page, PaginatedResponse } from "@/types/Paginated";
import {
  Dataset,
  DatasetPaginatedResponse,
  DatasetResponse,
  TrainSqlDataset,
  TrainingDetailsResponse,
} from "@/types/Dataset";
import { ApiResponse } from "@/types/ApiResponse";

export const createDataSet = (workspaceId: string, payload: Dataset) => {
  return httpV2.post<Partial<DatasetResponse>>(
    `/workspace/${workspaceId}/dataset`,
    payload,
  );
};

export const getDataSet = (
  workspaceId: string,
  page: Page,
  ingested?: boolean | undefined,
  type?: "sql" | "unstructured" | undefined,
  searchTerm?: string | undefined,
) => {
  const queryParams = new URLSearchParams({
    order: "ascendent",
    page: page?.page.toString(),
    size: page?.size.toString(),
  });

  if (type) {
    queryParams.append("type", type);
  }

  if (ingested !== undefined) {
    queryParams.append("ingested", String(ingested));
  }
  if (searchTerm) {
    queryParams.append("search", searchTerm);
  }

  return http.get<DatasetPaginatedResponse>(
    `/workspace/${workspaceId}/dataset?${queryParams.toString()}`,
  );
};

export const getDataSetById = async (
  workSpaceId: string,
  dataSetId: string,
) => {
  const res = await http.get<ApiResponse<DatasetResponse>>(
    `/workspace/${workSpaceId}/dataset/${dataSetId}`,
  );
  return res.data.data;
};

export const updateDataSet = (
  workspaceId: string,
  datasetId: string,
  payload: Dataset,
) => {
  return httpV2.put<Partial<DatasetResponse>>(
    `/workspace/${workspaceId}/dataset/${datasetId}`,
    payload,
  );
};

export const deleteFiles = (
  workspaceId: string,
  datasetId: string,
  fileIds: string[],
) => {
  return httpV2.delete(`/workspace/${workspaceId}/dataset/${datasetId}/files`, {
    data: { file_ids: fileIds },
  });
};

export const deleteDataSet = (workspaceId: string, datasetId: string) => {
  return httpV2.delete(`/workspace/${workspaceId}/dataset/${datasetId}`);
};

export const getDatasetChuckDetails = (
  workspaceId: string,
  datasetId: string,
  fileId?: string,
) => {
  return httpV2.get(
    `/workspace/${workspaceId}/dataset/${datasetId}/chunks?include_vectors=true&partial_vectors=true&page=1&size=10&file_id=${fileId}`,
  );
};

export const getTrainingDetails = (workspaceId: string, datasetId: string) => {
  return httpV2.get<ApiResponse<PaginatedResponse<TrainingDetailsResponse>>>(
    `/workspace/${workspaceId}/dataset/${datasetId}/trainings`,
  );
};

export const getIngestStatus = (datasetId: string) => {
  return httpV2.get(`/dataset/${datasetId}/ingestion_status`);
};

export const addGraphToDataset = (datasetId: string) => {
  return http.post(`/dataset/${datasetId}/build_graph`, {});
};

export const getGraphStatus = (datasetId: string) => {
  return http.get(`/dataset/${datasetId}/graph_status`);
};

export const trainSqlDataset = ({
  workspaceId,
  dataSetId,
  body,
}: TrainSqlDataset) => {
  return httpV2.post(
    `/workspace/${workspaceId}/dataset/${dataSetId}/train`,
    body,
  );
};

export const retryTrainSqlDataset = ({
  workspaceId,
  dataSetId,
  body,
}: TrainSqlDataset) => {
  return httpV2.post(
    `/workspace/${workspaceId}/dataset/${dataSetId}/retrain`,
    body,
  );
};

export const getDatasetsByOrganization = (orgId: string, page: Page) => {
  return http.get<DatasetPaginatedResponse>(
    `/organization/${orgId}/dataset?order=ascendent&page=${page?.page}&size=${page?.size}`,
  );
};
