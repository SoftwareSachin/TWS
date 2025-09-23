import { ReactNode } from "react";

export interface WorkspacePageWrapperProps {
  title: string;
  itemCount: number;
  searchTerm: string;
  onSearchChange: (value: string) => void;
  onCreateClick: () => void;
  renderItems: () => ReactNode;
  loading: boolean;
  error?: string | null;
  CreateModal?: ReactNode;
  DeleteModal?: ReactNode;
  pagination?: {
    page: number;
    size: number;
  };
  totalPages?: number;
  onPaginationChange?: (opts: { page: number; size: number }) => void;
}
