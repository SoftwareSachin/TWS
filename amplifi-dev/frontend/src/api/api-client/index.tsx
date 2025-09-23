import { Page } from "@/types/Paginated";
import { http } from "..";
import {
  ApiClient,
  ApiClientListResponse,
  ApiClientRegenerateResponse,
  ApiClientResponse,
} from "@/types/ApiClient";

import { AxiosResponse } from "axios";

export const getApiClientsData = (
  orgID: string,
  pagination: Page,
): Promise<AxiosResponse<ApiClientListResponse>> => {
  return http.get(
    `organization/${orgID}/api_client?page=${pagination.page}&size=${pagination.size}`,
  );
};

export const addApiClient = (
  orgID: string,
  request: ApiClient,
): Promise<AxiosResponse<ApiClientResponse>> => {
  return http.post(`organization/${orgID}/api_client`, request);
};

export const updateApiClient = (
  orgID: string,
  clientID: string,
  request: ApiClient,
): Promise<AxiosResponse<ApiClientResponse>> => {
  return http.put(`organization/${orgID}/api_client/${clientID}`, request);
};

export const deleteApiClient = (
  orgID: string,
  clientID: string,
): Promise<AxiosResponse<ApiClientResponse>> => {
  return http.delete(`organization/${orgID}/api_client/${clientID}`);
};

export const regenerateSecret = (
  orgID: string,
  clientID: string,
): Promise<AxiosResponse<ApiClientRegenerateResponse>> => {
  return http.post(
    `organization/${orgID}/api_client/${clientID}/regenerate-secret`,
  );
};
