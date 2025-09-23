"use client";

import React, { useState } from "react";
import LargeModal from "@/components/forms/largeModal";
import { Button } from "@/components/ui/button";
import { RadioButton } from "@/design_components/radio/radio-button";
import { FormLabel } from "../ui/FormLabel";
import { Input } from "../ui/input";
import { X } from "lucide-react";
import Image from "next/image";
import play from "@/assets/icons/play.svg";
import { showError, showSuccess, showWarning } from "@/utils/toastUtils";
import {
  createGraph,
  extractGraphEntities,
  extractGraphRelationships,
  getGraphById,
  getGraphEntitiesType,
  getGraphEntitiesRelationships,
  deleteGraphEntitiesByType,
  deleteGraph,
} from "@/api/graph";
import { useGraph } from "@/context_api/graphContext";

type AddGraphModalProps = {
  isOpen: boolean;
  onClose: () => void;
  onGenerateGraph: (relationships: any, graphId: string) => void;
  onSetOriginalSelectionMethod?: (method: "auto" | "manual") => void;
  datasetId: string;
  existingGraphId?: string | null;
  entitiesExtracted?: boolean;
  originalSelectionMethod?: "auto" | "manual" | null;
};

// Reusable component for entity details view
const EntityDetailsView: React.FC<{
  extractedEntities: any[];
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  setExtractedEntities: (entities: any[]) => void;
}> = ({
  extractedEntities,
  searchQuery,
  setSearchQuery,
  setExtractedEntities,
}) => (
  <div className="mt-6">
    {/* Search Bar */}
    <div className="mb-4">
      <div className="relative">
        <input
          type="text"
          placeholder="Search"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
        <svg
          className="absolute left-3 top-2.5 h-5 w-5 text-gray-400"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
          />
        </svg>
      </div>
    </div>

    {/* Entity List */}
    <div className="max-h-64 overflow-y-auto border border-gray-200 rounded-md">
      {extractedEntities
        .filter((entity) =>
          entity.name.toLowerCase().includes(searchQuery.toLowerCase()),
        )
        .map((entity) => (
          <div
            key={entity.id}
            className="flex items-center justify-between p-3 border-b border-gray-100 hover:bg-gray-50"
          >
            <div className="flex items-center space-x-3">
              <input
                type="checkbox"
                checked={entity.checked}
                onChange={(e) => {
                  const updatedEntities = extractedEntities.map((ent) =>
                    ent.id === entity.id
                      ? { ...ent, checked: e.target.checked }
                      : ent,
                  );
                  setExtractedEntities(updatedEntities);
                }}
                className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 focus:ring-2"
              />
              <div className="flex flex-col">
                <span className="text-sm font-medium text-gray-900">
                  {entity.name}
                </span>
                <span className="text-xs text-gray-500">
                  {entity.instances} entities found
                </span>
              </div>
            </div>
          </div>
        ))}
    </div>
  </div>
);

// Reusable component for loading skeleton
const LoadingSkeleton: React.FC = () => (
  <div className="mt-6 space-y-4">
    <div className="flex items-center justify-between">
      <div className="h-4 bg-gray-200 rounded animate-pulse w-48"></div>
      <div className="h-4 bg-gray-200 rounded animate-pulse w-16"></div>
    </div>
    <div className="border border-gray-200 rounded-md p-4 space-y-3">
      {[...Array(4)].map((_, i) => (
        <div key={i} className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-4 h-4 bg-gray-200 rounded animate-pulse"></div>
            <div className="h-4 bg-gray-200 rounded animate-pulse w-24"></div>
          </div>
          <div className="h-4 bg-gray-200 rounded animate-pulse w-20"></div>
        </div>
      ))}
    </div>
  </div>
);

// Reusable component for extraction button
const ExtractionButton: React.FC<{
  isExtractingEntities: boolean;
  entitiesGenerated: boolean;
  onClick: () => void;
  disabled?: boolean;
}> = ({ isExtractingEntities, entitiesGenerated, onClick, disabled }) => (
  <Button
    type="button"
    onClick={onClick}
    disabled={disabled || isExtractingEntities}
    className={`inline-flex items-center px-4 py-2 text-xs font-medium transition ${
      disabled || isExtractingEntities
        ? "bg-gray-300 text-gray-500 cursor-not-allowed"
        : "text-black bg-white border border-gray-300 hover:bg-gray-50"
    }`}
  >
    {isExtractingEntities ? (
      <>
        <div className="animate-spin h-3 w-3 border-2 border-gray-400 border-t-gray-600 rounded-full mr-2" />
        Extracting Entities
      </>
    ) : entitiesGenerated ? (
      <>
        <svg
          className="w-3 h-3 mr-2"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
          />
        </svg>
        Generate Again
      </>
    ) : (
      <>
        <Image
          className="mr-2"
          src={play}
          alt="Extract Entities"
          width={12}
          height={12}
        />
        Extract Entities
      </>
    )}
  </Button>
);

const AddGraphModal: React.FC<AddGraphModalProps> = ({
  isOpen,
  onClose,
  onGenerateGraph,
  onSetOriginalSelectionMethod,
  datasetId,
  existingGraphId = null,
  entitiesExtracted = false,
  originalSelectionMethod = null,
}) => {
  // Determine initial radio button state based on original selection method
  const getInitialAssociationType = () => {
    if (entitiesExtracted && originalSelectionMethod === "manual") {
      return "Manually";
    } else if (entitiesExtracted && originalSelectionMethod === "auto") {
      return "Auto Extract";
    }
    return undefined;
  };

  const [associationType, setAssociationType] = useState<string | undefined>(
    getInitialAssociationType(),
  );
  const [customEntities, setCustomEntities] = useState<string[]>([]);
  const [currentInput, setCurrentInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [graphPolling, setGraphPolling] = useState(false);
  const [entitiesGenerated, setEntitiesGenerated] = useState(entitiesExtracted);
  const [isExtractingEntities, setIsExtractingEntities] = useState(false);
  const [currentGraphId, setCurrentGraphId] = useState<string | null>(
    existingGraphId,
  );
  const [showEntityDetails, setShowEntityDetails] = useState(entitiesExtracted);
  const [extractedEntities, setExtractedEntities] = useState<any[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [hasStartedExtraction, setHasStartedExtraction] =
    useState(entitiesExtracted); // Track if extraction has started

  // Use Graph Context
  const {
    setIsGeneratingEntities,
    setIsGeneratingRelationships,
    updateGraphStatus,
  } = useGraph();

  const handleRadioChange = (value: string) => {
    // Only allow radio change if extraction hasn't started
    if (!hasStartedExtraction) {
      setAssociationType(value);
      setCustomEntities([]);
      setCurrentInput("");
      setEntitiesGenerated(false);
      setIsExtractingEntities(false);
      setCurrentGraphId(null);
      setShowEntityDetails(false);
      setExtractedEntities([]);
      setSearchQuery("");
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && currentInput.trim() !== "") {
      e.preventDefault();
      if (!customEntities.includes(currentInput.trim())) {
        setCustomEntities([...customEntities, currentInput.trim()]);
        setCurrentInput("");
      }
    }
  };

  const handleRemoveEntity = (index: number) => {
    setCustomEntities(customEntities.filter((_, i) => i !== index));
  };

  const fetchExtractedEntities = async (graphId: string) => {
    try {
      // Always use getGraphEntitiesType to group by type for consistency
      const response = await getGraphEntitiesType(datasetId, graphId);
      // Transform the entity types to match the expected format
      const transformedEntities = response.entity_types.map(
        (entityType: { entity_type: string; count: number }) => ({
          id: entityType.entity_type,
          name: entityType.entity_type,
          type: "Entity Type",
          description: `${entityType.count} entities found`,
          checked: true,
          instances: entityType.count,
        }),
      );
      setExtractedEntities(transformedEntities);
    } catch (error) {
      console.error("Failed to fetch extracted entities:", error);
      showError("Failed to fetch extracted entities");
    }
  };

  const fetchOriginalEntityTypes = async (graphId: string) => {
    try {
      // Get the graph details to check if it has original entity_types
      const graphDetails = await getGraphById(datasetId, graphId);

      if (
        graphDetails &&
        graphDetails.entity_types &&
        Array.isArray(graphDetails.entity_types)
      ) {
        setCustomEntities(graphDetails.entity_types);
      }
    } catch (error) {
      console.error("Failed to fetch original entity types:", error);
      showError("Failed to fetch original entity types");
    }
  };

  const pollEntitiesStatus = async (graphId: string) => {
    const startTimestampMs = Date.now();
    const maxPollDuration = 60 * 1000; // 1 minute
    const pollInterval = 15 * 1000; // 15 seconds
    let timeoutId: number | undefined;
    let pollingStopped = false;

    // Stop polling after 1 minute
    const stopPollingTimeout = window.setTimeout(async () => {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
      pollingStopped = true;
      setIsExtractingEntities(false);

      // Check if status is still pending when polling stops
      try {
        const graph = await getGraphById(datasetId, graphId);
        if (graph?.entities_status === "pending") {
          showWarning(
            "Polling stopped, fetch the status from fetch status button",
          );
          onClose(); // Close the modal
        }
      } catch (error) {
        console.error("Error checking final status:", error);
      }
    }, maxPollDuration);

    const poll = async () => {
      if (pollingStopped) return;

      try {
        const graph = await getGraphById(datasetId, graphId);
        const status = graph?.entities_status;
        if (!status || status === "pending") {
          const elapsedMs = Date.now() - startTimestampMs;
          if (elapsedMs < maxPollDuration) {
            timeoutId = window.setTimeout(poll, pollInterval);
          }
          return;
        }
        // terminal-ish state
        clearTimeout(stopPollingTimeout);
        setIsExtractingEntities(false);
        setIsGeneratingEntities(false); // Update context

        // Update context with the latest status
        updateGraphStatus(
          graphId,
          status,
          graph?.relationships_status || "pending",
        );

        if (status === "success") {
          setEntitiesGenerated(true);
          await fetchExtractedEntities(graphId);
          setShowEntityDetails(true);
          showSuccess("Entities extracted successfully");
        } else if (status === "failed") {
          showError("Entity extraction failed");
        } else {
          await fetchExtractedEntities(graphId);
          setShowEntityDetails(true);
        }
      } catch (error) {
        console.error("Polling entities status failed", error);
        const elapsedMs = Date.now() - startTimestampMs;
        if (elapsedMs < maxPollDuration && !pollingStopped) {
          timeoutId = window.setTimeout(poll, pollInterval);
        }
      }
    };
    poll();
  };

  const handleExtractEntities = async () => {
    // Immediately disable the button to prevent multiple clicks
    setIsExtractingEntities(true);
    setIsGeneratingEntities(true); // Update context

    // Mark that extraction has started - this will disable the other radio button
    setHasStartedExtraction(true);

    // Save the selection method when user starts entity extraction
    if (onSetOriginalSelectionMethod && associationType) {
      const method = associationType === "Auto Extract" ? "auto" : "manual";
      onSetOriginalSelectionMethod(method);
    }

    // If entities are already generated and we have an existing graph, delete it first
    if (entitiesGenerated && currentGraphId) {
      try {
        console.log(
          "Deleting existing graph before regenerating entities:",
          currentGraphId,
        );
        await deleteGraph(datasetId, currentGraphId);
        showSuccess(
          "Previous graph deleted. Starting new entity extraction...",
        );

        // Reset states after deletion
        setCurrentGraphId(null);
        setEntitiesGenerated(false);
        setShowEntityDetails(false);
        setExtractedEntities([]);
        setSearchQuery("");
      } catch (error) {
        console.error("Failed to delete existing graph:", error);
        showError("Failed to delete existing graph. Please try again.");
        setIsExtractingEntities(false);
        setIsGeneratingEntities(false);
        setHasStartedExtraction(false);
        return;
      }
    }

    if (associationType === "Auto Extract") {
      try {
        // Create a new graph
        const graphResponse = await createGraph(datasetId, {
          status: "pending",
          error_message: "",
        });
        const graphId = graphResponse.data.data.id;
        setCurrentGraphId(graphId);
        // Kick off extraction
        await extractGraphEntities(datasetId, graphId);
        // Poll until extraction complete
        await pollEntitiesStatus(graphId);
      } catch (error) {
        setIsExtractingEntities(false);
        setIsGeneratingEntities(false);
        setHasStartedExtraction(false);
        console.error("Entity extraction error:", error);
        showError("Failed to start entity extraction");
      }
    } else if (associationType === "Manually") {
      if (customEntities.length === 0) {
        showError("Please define at least one custom entity");
        setIsExtractingEntities(false);
        setIsGeneratingEntities(false);
        setHasStartedExtraction(false);
        return;
      }
      try {
        // Create a new graph
        const graphResponse = await createGraph(datasetId, {
          status: "pending",
          error_message: "",
          entity_types: customEntities,
        });
        const graphId = graphResponse.data.data.id;
        setCurrentGraphId(graphId);
        // Kick off extraction with custom entities
        await extractGraphEntities(datasetId, graphId, customEntities);
        // Poll until extraction complete
        await pollEntitiesStatus(graphId);
      } catch (error) {
        setIsExtractingEntities(false);
        setIsGeneratingEntities(false);
        setHasStartedExtraction(false);
        console.error("Custom entity extraction error:", error);
        showError("Failed to start custom entity extraction");
      }
    }
  };

  const pollGraphStatus = async () => {
    if (!currentGraphId) return;
    setGraphPolling(true);
    setIsGeneratingRelationships(true); // Update context
    const startTimestampMs = Date.now();
    const maxPollDuration = 60 * 2000; // 1 minute
    const pollInterval = 15 * 1000; // 15 seconds
    let timeoutId: number | undefined;
    let pollingStopped = false;

    // Stop polling after 1 minute
    const stopPollingTimeout = window.setTimeout(async () => {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
      pollingStopped = true;
      setGraphPolling(false);
      setIsGenerating(false);

      // Check if status is still pending when polling stops
      try {
        const graph = await getGraphById(datasetId, currentGraphId);
        if (graph?.relationships_status === "pending") {
          showWarning(
            "Polling stopped, fetch the status from fetch status button",
          );
          onClose(); // Close the modal
        } else if (
          graph?.relationships_status === "success" &&
          graph?.entities_status === "success"
        ) {
          // If polling stopped but graph is actually complete, show success
          try {
            const relationships = await getGraphEntitiesRelationships(
              datasetId,
              currentGraphId,
            );

            // Update context states to reflect completion
            setIsGeneratingEntities(false);
            setIsGeneratingRelationships(false);
            updateGraphStatus(currentGraphId, "success", "success");

            onGenerateGraph(relationships, currentGraphId);
            onClose();
          } catch (error) {
            console.error("Failed to fetch graph relationships:", error);
            showError("Failed to fetch graph relationships");
            onClose();
          }
        }
      } catch (error) {
        console.error("Error checking final status:", error);
      }
    }, maxPollDuration);

    const poll = async () => {
      if (pollingStopped) return;

      try {
        const graph = await getGraphById(datasetId, currentGraphId);
        if (graph && graph.relationships_status !== "pending") {
          clearTimeout(stopPollingTimeout);
          setGraphPolling(false);
          setIsGenerating(false);
          setIsGeneratingRelationships(false); // Update context

          // Update context with the latest status
          updateGraphStatus(
            currentGraphId,
            graph.entities_status,
            graph.relationships_status,
          );
          // Check if both entities and relationships are successful
          if (
            graph.relationships_status === "success" &&
            graph.entities_status === "success"
          ) {
            try {
              const relationships = await getGraphEntitiesRelationships(
                datasetId,
                currentGraphId,
              );

              // Update context states to reflect completion
              setIsGeneratingEntities(false);
              setIsGeneratingRelationships(false);
              updateGraphStatus(currentGraphId, "success", "success");

              // Close modal and immediately open summary modal
              onGenerateGraph(relationships, currentGraphId);
              onClose();
            } catch (error) {
              console.error("Failed to fetch graph relationships:", error);
              showError("Failed to fetch graph relationships");
            }
          } else if (graph.relationships_status === "failed") {
            // Update context states for failure
            setIsGeneratingRelationships(false);
            updateGraphStatus(currentGraphId, graph.entities_status, "failed");
            showError("Graph generation failed");
          } else {
            // Handle other non-pending states if needed
            console.log("Graph status:", graph.relationships_status);
          }
          return;
        }
        const elapsedMs = Date.now() - startTimestampMs;
        if (elapsedMs < maxPollDuration) {
          timeoutId = window.setTimeout(poll, pollInterval);
        }
      } catch (error: any) {
        if (error?.response?.status === 404) {
          const elapsedMs = Date.now() - startTimestampMs;
          if (elapsedMs < maxPollDuration && !pollingStopped) {
            timeoutId = window.setTimeout(poll, pollInterval);
          }
        } else {
          console.error("Polling failed", error);
          clearTimeout(stopPollingTimeout);
          setIsGenerating(false);
          setGraphPolling(false);
          setIsGeneratingRelationships(false); // Update context
          showError("Something went wrong while polling the graph.");
        }
      }
    };
    poll();
  };

  const handleSubmit = async () => {
    if (!currentGraphId) {
      showError("No graph available. Please extract entities first.");
      return;
    }

    try {
      setIsGenerating(true);

      // Delete unselected entity types before generating relationships
      const unselectedEntities = extractedEntities.filter(
        (entity) => !entity.checked,
      );
      if (unselectedEntities.length > 0) {
        const unselectedEntityTypes = unselectedEntities.map((e) => e.name);

        try {
          // Delete all unselected entity types in a single API call
          const deleteResponse = await deleteGraphEntitiesByType(
            datasetId,
            currentGraphId,
            unselectedEntityTypes,
          );
          console.log("Bulk delete response:", deleteResponse);
          showSuccess(
            `Deleted ${unselectedEntities.length} unselected entity types`,
          );
        } catch (error) {
          console.error("Failed to delete unselected entity types:", error);
          showError(
            "Failed to delete some entity types, but continuing with graph generation",
          );
        }
      }

      // Extract relationships for the existing graph
      await extractGraphRelationships(datasetId, currentGraphId);
      showSuccess("Graph generation started");
      pollGraphStatus();
    } catch (err) {
      setIsGenerating(false);
      console.error("Graph generation error:", err);
      showError("Failed to start graph generation");
    } finally {
      setLoading(false);
    }
  };

  // Check if Generate Graph button should be enabled
  const canGenerateGraph =
    showEntityDetails &&
    !isGenerating &&
    !graphPolling &&
    extractedEntities.some((entity) => entity.checked);

  // Fetch entities on component mount if entities are already extracted
  React.useEffect(() => {
    if (entitiesExtracted && existingGraphId) {
      // Always fetch the extracted entities for display
      fetchExtractedEntities(existingGraphId);

      // If this was originally manual selection, also fetch the original entity types
      if (originalSelectionMethod === "manual") {
        fetchOriginalEntityTypes(existingGraphId);
      }
    }
  }, [entitiesExtracted, existingGraphId, originalSelectionMethod]);

  // Cleanup effect when modal closes
  React.useEffect(() => {
    if (!isOpen) {
    }
  }, [isOpen]);

  if (isGenerating || graphPolling) {
    return (
      <LargeModal
        isOpen={isOpen}
        onClose={onClose}
        title="Create Graph from Dataset"
        type="graph"
        fullWidth={false}
        fullHeight={false}
        hideSubmitButton={true}
        cancelText="Close Window"
      >
        <div className="flex flex-col items-center justify-center py-10 space-y-4">
          <div className="animate-spin h-8 w-8 border-4 border-gray-200 border-t-blue-600 rounded-full" />
          <h3 className="text-md font-semibold">Generating Graph...</h3>
          <p className="text-sm text-muted-foreground text-center max-w-xs">
            We&apos;re processing your entities and building the graph.
          </p>
        </div>
      </LargeModal>
    );
  }

  return (
    <LargeModal
      isOpen={isOpen}
      onClose={onClose}
      onSubmit={handleSubmit}
      title="Create Graph from Dataset"
      actionButton="Generate Graph"
      type="graph"
      fullWidth={false}
      fullHeight={false}
      disabled={!canGenerateGraph}
    >
      <div className="p-4 text-sm space-y-6">
        {/* Radio group section - always show, disable non-selected after extraction starts */}
        <div>
          <FormLabel>Select Entities Generation Mode</FormLabel>

          {/* Show normal radio buttons if extraction hasn't started */}
          {!hasStartedExtraction && (
            <RadioButton
              value={associationType}
              onChange={handleRadioChange}
              options={[
                { value: "Auto Extract", label: "Auto Extract from Dataset" },
                { value: "Manually", label: "Define Manually" },
              ]}
            />
          )}

          {/* Show custom radio buttons with disabled states after extraction starts */}
          {hasStartedExtraction && (
            <div className="flex gap-4">
              {/* Auto Extract option */}
              <div className="px-3 py-1 space-y-2 rounded-lg">
                <label
                  className={`flex items-center space-x-2 ${
                    associationType === "Auto Extract"
                      ? "cursor-pointer"
                      : "cursor-not-allowed"
                  }`}
                >
                  <div
                    className={`w-4 h-4 border-2 rounded-full flex items-center justify-center ${
                      associationType === "Auto Extract"
                        ? "border-blue-600 bg-blue-600"
                        : "border-gray-300 bg-gray-100"
                    }`}
                  >
                    {associationType === "Auto Extract" && (
                      <div className="w-2 h-2 bg-white rounded-full"></div>
                    )}
                  </div>
                  <span
                    className={`text-sm ml-[10px] ${
                      associationType === "Auto Extract"
                        ? "text-blue-600 font-medium"
                        : "text-gray-400"
                    }`}
                  >
                    Auto Extract from Dataset
                  </span>
                </label>
              </div>

              {/* Manual option */}
              <div className="px-3 py-1 space-y-2 rounded-lg">
                <label
                  className={`flex items-center space-x-2 ${
                    associationType === "Manually"
                      ? "cursor-pointer"
                      : "cursor-not-allowed"
                  }`}
                >
                  <div
                    className={`w-4 h-4 border-2 rounded-full flex items-center justify-center ${
                      associationType === "Manually"
                        ? "border-blue-600 bg-blue-600"
                        : "border-gray-300 bg-gray-100"
                    }`}
                  >
                    {associationType === "Manually" && (
                      <div className="w-2 h-2 bg-white rounded-full"></div>
                    )}
                  </div>
                  <span
                    className={`text-sm ml-[10px] ${
                      associationType === "Manually"
                        ? "text-blue-600 font-medium"
                        : "text-gray-400"
                    }`}
                  >
                    Define Manually
                  </span>
                </label>
              </div>
            </div>
          )}
        </div>

        {/* Auto Extract Section */}
        {associationType === "Auto Extract" && (
          <div className="w-full flex flex-col pt-3 border-t border-[#eeeeee]">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-medium text-black mb-1">
                  {isExtractingEntities
                    ? "Entities Generation Started"
                    : entitiesGenerated
                      ? "Successfully Generated Entities"
                      : "Start entities generation"}
                </h3>
                <p className="text-sm text-muted-foreground">
                  {isExtractingEntities
                    ? "You can either stay or close, we will notify you once done"
                    : entitiesGenerated
                      ? `Found ${extractedEntities.length} entity types with their counts`
                      : "This may take time depending on the dataset size"}
                </p>
              </div>

              <ExtractionButton
                isExtractingEntities={isExtractingEntities}
                entitiesGenerated={entitiesGenerated}
                onClick={handleExtractEntities}
              />
            </div>

            {/* Loading and Entity Display */}
            {isExtractingEntities && <LoadingSkeleton />}
            {showEntityDetails && !isExtractingEntities && (
              <EntityDetailsView
                extractedEntities={extractedEntities}
                searchQuery={searchQuery}
                setSearchQuery={setSearchQuery}
                setExtractedEntities={setExtractedEntities}
              />
            )}
          </div>
        )}

        {/* Manual Entity Definition */}
        {associationType === "Manually" && (
          <div>
            <FormLabel htmlFor="customEntities">
              Define Custom Entities
            </FormLabel>
            <Input
              id="customEntities"
              value={currentInput}
              onChange={(e) => setCurrentInput(e.target.value)}
              onKeyDown={handleKeyPress}
            />
            <div className="mt-2 flex flex-wrap gap-2">
              {customEntities.map((entity, index) => (
                <div
                  key={index}
                  className="flex items-center gap-1 bg-[#D7DBFC] px-2 py-1 h-[28px] rounded-[22px] text-sm"
                >
                  {entity}
                  <button onClick={() => handleRemoveEntity(index)}>
                    <X size={12} className="text-[#292929] mt-1 " />
                  </button>
                </div>
              ))}
            </div>
            <div className="flex space-x-1 space-y-2 text-sm text-muted-foreground">
              <span className="mt-2 text-red-500">*</span>
              <p>
                If an entered entity is not found in the dataset, it will be
                ignored in the final graph.
              </p>
            </div>

            {/* Start entities generation section for manual flow */}
            <div className="w-full flex flex-col pt-3 border-t border-[#eeeeee] mt-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-medium text-black mb-1">
                    {isExtractingEntities
                      ? "Entities Generation Started"
                      : entitiesGenerated
                        ? "Successfully Generated Entities"
                        : "Start entities generation"}
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    {isExtractingEntities
                      ? "You can either stay or close, we will notify you once done"
                      : entitiesGenerated
                        ? `Found ${extractedEntities.length} entity types with their counts`
                        : "This may take time depending on the dataset size"}
                  </p>
                </div>

                <ExtractionButton
                  isExtractingEntities={isExtractingEntities}
                  entitiesGenerated={entitiesGenerated}
                  onClick={handleExtractEntities}
                  disabled={
                    !entitiesGenerated &&
                    (customEntities.length === 0 || isExtractingEntities)
                  }
                />
              </div>

              {/* Loading and Entity Display */}
              {isExtractingEntities && <LoadingSkeleton />}
              {showEntityDetails && !isExtractingEntities && (
                <EntityDetailsView
                  extractedEntities={extractedEntities}
                  searchQuery={searchQuery}
                  setSearchQuery={setSearchQuery}
                  setExtractedEntities={setExtractedEntities}
                />
              )}
            </div>
          </div>
        )}
      </div>
    </LargeModal>
  );
};

export default AddGraphModal;
