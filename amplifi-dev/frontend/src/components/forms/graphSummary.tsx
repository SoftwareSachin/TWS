"use client";

import { useState } from "react";
import LargeModal from "@/components/forms/largeModal";
import { deleteGraph } from "@/api/graph";
import { showError, showSuccess } from "@/utils/toastUtils";
import { useGraph } from "@/context_api/graphContext";
import { Button } from "../ui/button";
import {
  Accordion,
  AccordionItem,
  AccordionTrigger,
  AccordionContent,
} from "@/components/ui/accordian";
import DeleteModal from "./deleteModal";
import WarningModal from "./warningModal";

type GraphSummaryModalProps = {
  isOpen: boolean;
  onClose: () => void;
  onRegenerate: () => void;
  onGraphDeleted?: () => void;
  datasetId: string;
  graphId: string;
  relationships: {
    entities: Array<{
      name: string;
      type: string;
      description: string;
    }>;
    relationships: Array<{
      source_entity: string;
      target_entity: string;
      relationship_type: string;
      relationship_description: string;
    }>;
    total_entities: number;
    total_relationships: number;
  } | null;
};

const GraphSummaryModal: React.FC<GraphSummaryModalProps> = ({
  isOpen,
  onClose,
  onRegenerate,
  onGraphDeleted,
  datasetId,
  graphId,
  relationships,
}) => {
  const [accordionValue, setAccordionValue] = useState<string | null>(null);
  const [selectedTab, setSelectedTab] = useState<"entities" | "relationships">(
    "entities",
  );
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [showRegenerateModal, setShowRegeneratModal] = useState(false);

  // Use Graph Context
  const { resetGraphState } = useGraph();

  const isEntities = selectedTab === "entities";

  // Process entities data
  const entityTypes =
    relationships?.entities.reduce(
      (acc, entity) => {
        const type = entity.type.charAt(0).toUpperCase() + entity.type.slice(1); // Capitalize type
        if (!acc[type]) {
          acc[type] = [];
        }
        acc[type].push(entity.name);
        return acc;
      },
      {} as Record<string, string[]>,
    ) ?? {};

  // Process relationships data
  const relationshipTypes =
    relationships?.relationships.reduce(
      (acc, rel) => {
        const type = rel.relationship_type.toUpperCase().replace(/ /g, "_");
        if (!acc[type]) {
          acc[type] = [];
        }
        acc[type].push(`${rel.source_entity} → ${rel.target_entity}`);
        return acc;
      },
      {} as Record<string, string[]>,
    ) ?? {};

  const currentData = isEntities ? entityTypes : relationshipTypes;
  const totalCount = isEntities
    ? (relationships?.total_entities ?? 0)
    : (relationships?.total_relationships ?? 0);
  const typeCount = Object.keys(currentData).length;
  const handleDeleteClick = () => {
    setShowDeleteModal(true);
  };

  const handleDeleteConfirm = async () => {
    try {
      console.log(
        "Deleting graph with ID:",
        graphId,
        "from dataset:",
        datasetId,
      );
      const response = await deleteGraph(datasetId, graphId);
      console.log("Graph deleted successfully:", response);
      showSuccess("Knowledge graph deleted successfully");

      // Reset graph state in context
      resetGraphState();

      // Clear any existing deletion flag to allow fresh start
      localStorage.removeItem(`graph_deleted_${datasetId}`);

      // Call the dataset page refresh callback
      if (onGraphDeleted) {
        console.log("GRAPH DELETED: Calling dataset page refresh callback");
        await onGraphDeleted();
      }

      setShowDeleteModal(false);
      onClose();
    } catch (error) {
      console.error("Failed to delete graph:", error);
      showError("Failed to delete knowledge graph");
    }
  };

  const handleRegenerateClick = () => {
    setShowRegeneratModal(true);
  };

  const handleRegenerateConfirm = async () => {
    try {
      // Delete the existing graph first
      await deleteGraph(datasetId, graphId);
      // Reset graph state in context
      resetGraphState();

      // Call the dataset page refresh callback to update the UI state
      if (onGraphDeleted) {
        onGraphDeleted();
      }

      showSuccess("Previous graph deleted. Starting regeneration...");

      setShowRegeneratModal(false);
      onClose();
      onRegenerate();
    } catch (error) {
      console.error("Failed to delete graph for regeneration:", error);
      showError("Failed to delete existing graph. Please try again.");
    }
  };

  return (
    <>
      <LargeModal
        isOpen={isOpen}
        onClose={onClose}
        onSubmit={onClose}
        showDelete={true}
        showRegenerate={true}
        title="Knowledge Graph Summary"
        actionButton="Close"
        type="graph"
        fullWidth={false}
        fullHeight={false}
        hideCancelButton={true}
        hideSubmitButton={true}
        deleteText="Delete Knowledge Graph"
        onDelete={handleDeleteClick}
        onRegenerate={handleRegenerateClick}
        regenerateButtonClassName="bg-custom-warning text-white border px-4 py-2 rounded text-sm hover:bg-custom-warning/80 rounded-md"
      >
        {/* Tabs */}
        <div className="flex gap-4 mt-4 px-6">
          <Button
            className={`px-3 py-1 text-sm font-medium ${
              isEntities
                ? "bg-custom-numberColor text-custom-customBlue"
                : "bg-white text-black"
            }`}
            onClick={() => setSelectedTab("entities")}
          >
            Entity Types
          </Button>
          <Button
            className={`px-3 py-1 text-sm font-medium ${
              !isEntities
                ? "bg-custom-numberColor text-custom-customBlue"
                : "bg-white text-black"
            }`}
            onClick={() => setSelectedTab("relationships")}
          >
            Graph Relationships
          </Button>
        </div>

        {/* Divider */}
        <div className="w-full flex justify-between items-end mt-3 pt-3 border-t border-[#eeeeee]" />

        {/* Summary Count */}
        <div className="flex gap-6 px-4 pt-2 pb-4 text-sm text-muted-foreground font-medium">
          {isEntities ? (
            <>
              <div className="flex items-center gap-2">
                <span>Total Types</span>
                <span className="px-2 py-0.5 rounded-full bg-custom-numberColor text-blue-700 text-xs font-semibold">
                  {typeCount}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span>Total Entities</span>
                <span className="px-2 py-0.5 rounded-full bg-custom-numberColor text-blue-700 text-xs font-semibold">
                  {totalCount}
                </span>
              </div>
            </>
          ) : (
            <div className="flex items-center gap-2">
              <span>Total Relationships</span>
              <span className="px-2 py-0.5 rounded-full bg-custom-numberColor text-blue-700 text-xs font-semibold">
                {totalCount}
              </span>
            </div>
          )}
        </div>

        {/* Accordion */}
        <div className="px-4 pb-6">
          <Accordion
            type="single"
            collapsible
            value={accordionValue ?? undefined}
            onValueChange={setAccordionValue}
            className="space-y-2"
          >
            {Object.entries(currentData).map(([label, items]) => (
              <div
                key={label}
                className="bg-white rounded-xl border border-gray-200 shadow-sm"
              >
                <AccordionItem
                  value={label}
                  className="transition-all duration-200 ease-in-out"
                >
                  <AccordionTrigger className="px-4 py-3 flex justify-between items-center text-left w-full transition-all duration-200 ease-in-out">
                    <span className="text-sm font-medium text-left w-full text-start">
                      {label}
                    </span>
                    <span className="text-sm px-2 py-1 rounded-full bg-custom-numberColor text-blue-700 font-semibold w-10 text-center">
                      {String(items.length).padStart(2, "0")}
                    </span>
                  </AccordionTrigger>
                  <AccordionContent className="border-t p-3 text-gray-600 transition-all duration-200 ease-in-out overflow-hidden">
                    <div className="animate-accordion-down">
                      {items.length > 0 ? (
                        <ul className="list-disc list-inside space-y-1 max-h-32 overflow-y-auto pr-1 text-sm">
                          {items.map((item, i) => (
                            <li
                              key={i}
                              className="transition-opacity duration-200 ease-in-out"
                            >
                              {item}
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p>No data found.</p>
                      )}
                    </div>
                  </AccordionContent>
                </AccordionItem>
              </div>
            ))}
          </Accordion>
        </div>
      </LargeModal>
      <DeleteModal
        title="Delete Knowledge Graph"
        subTitle="This action will permanently remove the graph."
        isOpen={showDeleteModal}
        onDelete={handleDeleteConfirm}
        onClose={() => setShowDeleteModal(false)}
      />
      <WarningModal
        title="Regenerate Knowledge Graph"
        subTitle="This action will start a new graph generation process."
        isOpen={showRegenerateModal}
        onRegenerate={handleRegenerateConfirm}
        onClose={() => setShowRegeneratModal(false)}
        submitText="Regenerate"
        cancelText="Discard"
      />
    </>
  );
};

export default GraphSummaryModal;
