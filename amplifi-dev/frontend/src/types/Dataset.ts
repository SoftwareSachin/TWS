import { ApiResponse } from "@/types/ApiResponse";
import { PaginatedResponse } from "@/types/Paginated";

export interface Dataset {
  name: string;
  description: string;
  file_ids?: string[] | null;
  source_id?: string | null;
  source_type?: string | null;
}

export interface TrainSqlDataset {
  workspaceId: string;
  dataSetId: string;
  body: TrainDatasetPayload;
}

export interface TrainDatasetPayload {
  documentation: string | null;
  question_sql_pairs: TrainSqlQueryPairs[];
}

export interface TrainSqlQueryPairs {
  question: string;
  sql: string;
}

export interface DatasetResponse {
  name: string;
  description: string;
  file_ids: string[]; // Assuming it's an array of strings (update if different)
  id: string;
  source_id: string;
  source_type?: string | null;
  r2r_collection_id: string;
  knowledge_graph: boolean;
  graph_build_phase: string;
  graph_build_requested_at: string; // ISO timestamp as string
  graph_build_completed_at: string;
  last_extraction_check_at: string;
}

export type DatasetPaginatedResponse = ApiResponse<
  PaginatedResponse<DatasetResponse>
>;

export interface TrainingDetailsResponse {
  id: string;
  dataset_id: string;
  documentation: string | null;
  question_sql_pairs: Array<{
    question: string;
    sql: string;
  }> | null;
  version_id: number;
  created_at: string; // ISO timestamp as string
}
