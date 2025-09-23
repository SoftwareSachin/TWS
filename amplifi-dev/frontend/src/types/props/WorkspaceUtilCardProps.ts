export interface WorkspaceUtilCardProps {
  title: string;
  description: string;
  tag: string;
  allToolNames?: string[];
  allAgentNames?: string[];
  onEdit?: () => void;
  onDelete?: () => void;
  onShowDetails?: () => void;
  actionUrl?: string;
  actionText?: string;
  openInNewTab?: boolean;
  onActionClick?: () => void;
  sourceType?: string | null;
  isEditable?: boolean;
}
