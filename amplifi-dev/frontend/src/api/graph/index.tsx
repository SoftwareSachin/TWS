import { http, httpV2 } from "..";
import { ApiResponse } from "@/types/ApiResponse";
import {
  IGraphCreate,
  IGraphRead,
  IGraphUpdate,
  IEntityTypesResponse,
  IGraphEntitiesRelationships,
  IEntityExtractionPayload,
  IEntity,
} from "@/agent_schemas/graph_schema";

export const createGraph = async (dataSetId: string, payload: IGraphCreate) => {
  return httpV2.post<ApiResponse<IGraphRead>>(
    `/dataset/${dataSetId}/graph`,
    payload,
  );
};

export const getGraphById = async (
  dataSetId: string,
  graphId: string,
): Promise<IGraphRead> => {
  const res = await httpV2.get<ApiResponse<IGraphRead>>(
    `/dataset/${dataSetId}/graph/${graphId}`,
  );
  return res.data.data;
};

export const getAllGraphsByDatasetId = async (
  dataSetId: string,
): Promise<IGraphRead> => {
  const res = await httpV2.get<ApiResponse<IGraphRead>>(
    `/dataset/${dataSetId}/graph`,
  );
  return res.data.data;
};

export const updateGraph = async (
  dataSetId: string,
  graphId: string,
  payload: IGraphUpdate,
) => {
  return httpV2.put<ApiResponse<IGraphRead>>(
    `/dataset/${dataSetId}/graph/${graphId}`,
    payload,
  );
};

export const deleteGraph = async (dataSetId: string, graphId: string) => {
  const res = await httpV2.delete<ApiResponse<{ message: string }>>(
    `/dataset/${dataSetId}/graph/${graphId}`,
  );
  return res.data;
};

export const extractGraphEntities = async (
  dataSetId: string,
  graphId: string,
  entityTypes?: string[],
) => {
  const payload: IEntityExtractionPayload = {
    entity_types: entityTypes,
  };
  return httpV2.post<ApiResponse<{ message: string }>>(
    `/dataset/${dataSetId}/graph/${graphId}/entities`,
    payload,
  );
};

export const extractGraphRelationships = async (
  dataSetId: string,
  graphId: string,
) => {
  return httpV2.post<ApiResponse<{ message: string }>>(
    `/dataset/${dataSetId}/graph/${graphId}/relationships`,
  );
};

export const getGraphEntities = async (
  dataSetId: string,
  graphId: string,
): Promise<IEntity[]> => {
  const res = await httpV2.get<ApiResponse<IEntity[]>>(
    `/dataset/${dataSetId}/graph/${graphId}/entities`,
  );
  return res.data.data;
};

export const getGraphEntitiesType = async (
  dataSetId: string,
  graphId: string,
): Promise<IEntityTypesResponse> => {
  const res = await httpV2.get<ApiResponse<IEntityTypesResponse>>(
    `/dataset/${dataSetId}/graph/${graphId}/entity-types`,
  );
  return res.data.data;
};

export const getGraphEntitiesRelationships = async (
  dataSetId: string,
  graphId: string,
): Promise<IGraphEntitiesRelationships> => {
  const res = await httpV2.get<ApiResponse<IGraphEntitiesRelationships>>(
    `/dataset/${dataSetId}/graph/${graphId}/entities-relationships`,
  );
  return res.data.data;
};

export const deleteGraphEntitiesByType = async (
  dataSetId: string,
  graphId: string,
  entityTypes: string[],
) => {
  const queryParams = entityTypes
    .map((type) => `types=${encodeURIComponent(type)}`)
    .join("&");
  return httpV2.delete<ApiResponse<{ message: string }>>(
    `/dataset/${dataSetId}/graph/${graphId}/entity?${queryParams}`,
  );
};
