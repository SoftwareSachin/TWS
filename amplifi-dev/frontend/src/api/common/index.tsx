import { http } from "..";
import { FilePaginatedResponse, FileResponse } from "@/types/Files";
import { Page } from "@/types/Paginated";
import { OrganizationPaginatedResponse } from "@/types/Organization";
import { ApiResponse } from "@/types/ApiResponse";
import { DatasetResponse } from "@/types/Dataset";

export const getFiles = async (
  workspaceId: string,
  onlyUploaded: boolean,
  page?: Page,
  search?: string,
) => {
  let queryParamStr = `?order=ascendent&only_uploaded=${onlyUploaded}`;
  if (page) {
    queryParamStr += `&page=${page.page}&size=${page.size}`;
  }
  if (search) {
    queryParamStr += `&search=${search}`;
  }
  return await http.get<FilePaginatedResponse | FileResponse>(
    `/workspace/${workspaceId}/file${queryParamStr}`,
  );
};

export const getDestination = async (oId: string) => {
  return await http.get(`/organization/${oId}/destination?page=1&size=100`);
};

// get all organization
export const getOrganization = async () => {
  return await http.get<OrganizationPaginatedResponse>(
    `/organization?page=1&size=100`,
  );
};

// getDatasetDEtails
export const getDatasetDetails = async (
  workspace_id: string,
  dataset_id: string,
) => {
  const res = await http.get<ApiResponse<DatasetResponse>>(
    `/workspace/${workspace_id}/dataset/${dataset_id}`,
  );
  return res?.data?.data?.name;
};

// getDestinationDetails
export const getDestinationDetails = async (
  organizationId: string,
  destinationId: string,
) => {
  const res = await http.get(
    `/organization/${organizationId}/destination/${destinationId}`,
  );
  return res?.data?.data?.name;
};
