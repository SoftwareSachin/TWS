import { ApiResponse } from "@/types/ApiResponse";
import { PaginatedResponse } from "@/types/Paginated";

export interface FileObject {
  filename: string;
  mimetype: string;
  size: number;
  status: string;
  id?: string;
}

export type FilePaginatedResponse = ApiResponse<PaginatedResponse<FileObject>>;
export type FileResponse = ApiResponse<FileObject[]>;
