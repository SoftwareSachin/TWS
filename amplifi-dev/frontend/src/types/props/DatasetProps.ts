import { DatasetResponse } from "@/types/Dataset";

export interface SelectDatasetProps {
  setIsOpen: (open: boolean) => void;
  parentId: string;
  setSelectedDatasetId?: (ids: string[]) => void;
  selectedDatasetId?: string[];
  setSelectedDataset?: (ids: DatasetResponse[]) => void;
  selectedDataset?: DatasetResponse[];
  type?: "sql" | "unstructured" | undefined;
  multiple?: boolean;
  datasetFrom?: "organization" | "workspace";
}
