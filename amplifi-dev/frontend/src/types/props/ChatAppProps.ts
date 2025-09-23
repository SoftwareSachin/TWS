export interface CreateChatAppProps {
  onClose: () => void;
  workspaceId: string;
  chatAppId: string;
  setNewDataAdded: (newDataAdded: boolean) => void;
  newDataAdded: boolean;
  formSubmit: (submitFn: () => () => Promise<void>) => void;
}

export interface MarkdownViewerProps {
  mdText: string;
}
