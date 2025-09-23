import { ApiResponse } from "@/types/ApiResponse";
import { PaginatedResponse } from "@/types/Paginated";

export interface Organization {
  name: string;
  description?: string;
  domain?: string;
  id?: string;
}

export type OrganizationPaginatedResponse = ApiResponse<
  PaginatedResponse<Organization>
>;
