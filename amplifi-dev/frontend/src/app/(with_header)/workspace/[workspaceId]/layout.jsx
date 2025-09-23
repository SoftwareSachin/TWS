import WorkspacePanelLayout from "@/components/workspace/workspacePanel";

const WorkspaceDetails = ({ children }) => {
  return (
    <div className="">
      <WorkspacePanelLayout> {children}</WorkspacePanelLayout>
    </div>
  );
};

export default WorkspaceDetails;
