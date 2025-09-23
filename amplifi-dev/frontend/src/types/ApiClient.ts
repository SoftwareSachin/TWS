import { z } from "zod";
import { ApiResponse } from "./ApiResponse";
import { PaginatedResponse } from "./Paginated";

export const ApiClientSchema = z.object({
  name: z.string().min(1, "Name is required"),
  expires_at: z.string().min(1, "Expires at is required"),
});

export const AddApiClientFormSchema = z.object({
  apiClient: ApiClientSchema,
});

export interface ApiClient {
  name: string;
  expires_at: string;
}

export type AddApiClientFormValues = z.infer<typeof AddApiClientFormSchema>;

export interface ApiClientItem {
  name: string;
  description: string;
  expires_at: string;
  id: string;
  client_id: string;
  organization_id: string;
  created_at: string;
  updated_at: string;
  last_used_at: any;
  client_secret?: string;
}

export type ApiClientListResponse = ApiResponse<
  PaginatedResponse<ApiClientItem>
>;

export type ApiClientResponse = ApiResponse<ApiClientItem>;

export type ApiClientRegenerateResponse = ApiResponse<{
  client_secret: string;
}>;
