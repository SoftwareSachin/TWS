/* This code snippet is a React component named `Datasets`. It fetches chunk details for a dataset
using an API call and displays the data in a table format. Here's a breakdown of what the code does: */

"use client";
import React, { useEffect, useRef, useState } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  addGraphToDataset,
  getDataSetById,
  getDatasetChuckDetails,
  getGraphStatus,
  deleteFiles,
  getTrainingDetails,
} from "@/api/dataset";
import {
  getAllGraphsByDatasetId,
  getGraphEntitiesRelationships,
  deleteGraph,
} from "@/api/graph";
import { useGraph } from "@/context_api/graphContext";
import {
  getSourceConnectorDetails,
  getSourceDetails,
} from "@/api/Workspace/WorkSpaceFiles";
import { useParams } from "next/navigation";
import { showError, showSuccess } from "@/utils/toastUtils";
import { Button } from "@/components/ui/button";
import {
  Loader,
  ArrowLeft,
  RefreshCw,
  Plus,
  BarChart3,
  ArrowUp,
  ArrowDown,
} from "lucide-react";
import { constants, GraphStatus, SortDirection } from "@/lib/constants";
import Paginator from "@/components/utility/paginator";
import { getStatus } from "@/api/Workspace/workspace";
import { IngestionStatus } from "@/lib/constants";
import Modal from "@/components/forms/modal";
import IngestionForm from "@/components/forms/ingestionForm";
import { SqlTrainingForm } from "@/components/forms/sqlTrainingForm";
import LargeModal from "@/components/forms/largeModal";
import CreateDatasetForm from "@/components/forms/createDatasetForm";
import DeleteModal from "@/components/forms/deleteModal";
import { addChunk } from "@/api/Workspace/workspace";
import { useRouter } from "next/navigation";
import Image from "next/image";
import {
  identifyUserFromObject,
  hashString,
  captureEvent,
} from "@/utils/posthogUtils";
import { useUser } from "@/context_api/userContext";
import processIcon from "@/assets/icons/processing.svg";
import successIcon from "@/assets/icons/success-state.svg";
import failedIcon from "@/assets/icons/alert-triangle.svg";
import arrowIcon from "@/assets/icons/arrows.svg";
import trashIcon from "@/assets/icons/trash-icon.svg";
import viewChunks from "@/assets/icons/view-chunks.svg";

import AddGraphModal from "@/components/forms/addGraphModal";
import GraphSummaryModal from "@/components/forms/graphSummary";

import { getNextSortState, getSortedData } from "@/components/utility/sorting";
const Datasets = () => {
  const { user } = useUser();
  const [chunkDetails, setChunkDetails] = useState(null);
  const [loader, setLoader] = useState(true); // Initialize as true
  const [showModal, setShowModal] = useState(false);
  const [addGraphEnable, setAddGraphEnable] = useState(false);
  const [fileMetaData, setFileMetaData] = useState([]);
  const [sourceId, setSourceId] = useState(null);
  const [sourceType, setSourceType] = useState(null);
  const [isStructured, setIsStructured] = useState(false);
  const [pagination, setPagination] = useState({
    page: 1,
    size: 25,
  });
  const [totalPages, setTotalPages] = useState(1);
  const [ingestionUIState, setIngestionUIState] = useState(
    IngestionStatus.SUCCESS,
  );
  const [statusCheckInterval, setStatusCheckInterval] = useState(null);
  const [showReingestModal, setShowReingestModal] = useState(false);
  const [showSqlTrainingModal, setShowSqlTrainingModal] = useState(false);
  const [newDataAdded, setNewDataAdded] = useState(false);
  const [datasetName, setDatasetName] = useState("");
  const [ingestionStatusData, setIngestionStatusData] = useState([]);
  const [showChunksModal, setShowChunksModal] = useState(false);
  const [selectedFileData, setSelectedFileData] = useState(null);
  const [isLoadingChunks, setIsLoadingChunks] = useState(false);
  const [showAddMoreFilesModal, setShowAddMoreFilesModal] = useState(false);
  const [sortField, setSortField] = useState(null);
  const [sortDirection, setSortDirection] = useState(null);
  const [showAddGraphModal, setShowAddGraphModal] = useState(false);
  const [showGraphSummaryModal, setShowGraphSummaryModal] = useState(false);
  const [handleNextClick, setHandleNextClick] = useState(null);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [fileToDelete, setFileToDelete] = useState(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [trainingDetails, setTrainingDetails] = useState([]);
  const [filesCount, setFilesCount] = useState(0);
  const [lastDeletedFileId, setLastDeletedFileId] = useState(null);

  // Use Graph Context
  const {
    currentGraphId,
    graphEntitiesStatus,
    graphRelationshipsStatus,
    isGeneratingEntities,
    isGeneratingRelationships,
    isFetchingStatus,
    graphRelationships,
    setCurrentGraphId,
    setGraphEntitiesStatus,
    setGraphRelationshipsStatus,
    setIsFetchingStatus,
    setGraphRelationships,
    setIsGeneratingEntities,
    setIsGeneratingRelationships,
    updateGraphStatus,
  } = useGraph();

  // Track previous statuses to toast only on transition to FAILED
  const prevEntitiesStatusRef = useRef(null);
  const prevRelationshipsStatusRef = useRef(null);

  // Add state to track original selection method
  const [originalSelectionMethod, setOriginalSelectionMethod] = useState(null);

  const params = useParams();
  const workspaceId = params.workspaceId || "";
  const datasetId = params.sourceId || "";
  const router = useRouter();

  //function to get dataset and source information

  const handleSort = (field) => {
    const { nextField, nextDirection } = getNextSortState(
      field,
      sortField,
      sortDirection,
    );
    setSortField(nextField);
    setSortDirection(nextDirection);
  };
  const fetchDatasetAndSourceInfo = async () => {
    try {
      const datasetRes = await getDataSetById(workspaceId, datasetId);
      const currentFilesCount = datasetRes?.file_ids?.length || 0;
      console.log("filesCount", currentFilesCount);
      setFilesCount(currentFilesCount);
      const source_id = datasetRes?.source_id;
      setSourceId(source_id);
      setDatasetName(datasetRes?.name || "Dataset");

      let fetchedSourceType = null;
      if (source_id) {
        const sourceRes = await getSourceDetails({
          workspaceId: workspaceId,
          sourceId: source_id,
        });
        fetchedSourceType = sourceRes?.data?.data?.source_type;
        setSourceType(fetchedSourceType);
      }

      return {
        source_id,
        sourceType: fetchedSourceType,
        filesCount: currentFilesCount,
      };
    } catch (err) {
      console.error("Error fetching dataset or source info:", err);
      throw err;
    }
  };

  // function to get POSTGRESQL data
  const fetchStructuredData = async (source_id) => {
    setIsStructured(true);
    const data = {
      workspaceId: workspaceId,
      sourceId: source_id,
    };
    console.log("fetchStructuredData - data:", data);
    const fileMeta = await getSourceConnectorDetails(data, pagination);
    console.log("fetchStructuredData - API response:", fileMeta);
    console.log("fetchStructuredData - items:", fileMeta?.data?.data?.items);
    setFileMetaData(fileMeta?.data?.data?.items || []);
    const totalFiles = fileMeta?.data?.data?.total || 0;
    const calculatedTotalPages = Math.ceil(totalFiles / pagination.size);
    setTotalPages(calculatedTotalPages);
  };

  // function to get Unstructured data
  const fetchUnstructuredData = async () => {
    setIsStructured(false);
    const chunkRes = await getDatasetChuckDetails(workspaceId, datasetId);
    setChunkDetails(chunkRes?.data?.data?.items || []);
  };

  const fetchTrainingDetails = async () => {
    const trainingRes = await getTrainingDetails(workspaceId, datasetId);
    setTrainingDetails(trainingRes?.data?.data?.items || []);
    return trainingRes; // Return the response
  };
  // Main function to load data based on source type
  const loadDataBySourceType = async () => {
    try {
      setFileMetaData([]);
      setChunkDetails(null);

      const { source_id, sourceType: fetchedSourceType } =
        await fetchDatasetAndSourceInfo();

      if (
        fetchedSourceType === constants.SOURCE_TYPE.POSTGRES ||
        fetchedSourceType === constants.SOURCE_TYPE.MYSQL
      ) {
        await fetchStructuredData(source_id);
      } else {
        setIsStructured(false);
      }
    } catch (err) {
      setChunkDetails([]);
      setFileMetaData([]);
      showError(`${err?.response?.data?.detail}`);
      console.error("Error in loadDataBySourceType:", err);
    } finally {
      setLoader(false); // Set loader to false only after complete data loading
    }
  };

  const getGraphStatusFirst = async () => {
    try {
      const statusResponse = await getGraphStatus(params.sourceId);

      const has_knowledge_graph =
        statusResponse?.data?.data?.has_knowledge_graph;
      if (!has_knowledge_graph) {
        setAddGraphEnable(true);
      }
    } catch (statusError) {
      console.error("Error fetching status:", statusError);
      setShowModal(false);
    }
  };

  const checkIngestionStatus = async () => {
    try {
      const statusResponse = await getStatus(datasetId);
      const statusList = statusResponse?.data?.data || [];

      // Store the ingestion status data for the table
      setIngestionStatusData(statusList);

      // If no status data, check if we have chunk details (which means ingestion completed)
      if (statusList.length === 0) {
        // Try to load data to see if ingestion actually completed
        try {
          await loadDataBySourceType();
          setIngestionUIState(IngestionStatus.SUCCESS);
          return IngestionStatus.SUCCESS;
        } catch (err) {
          setIngestionUIState(IngestionStatus.NOT_STARTED);
          return IngestionStatus.NOT_STARTED;
        }
      }

      if (
        statusList.some(
          (item) =>
            item.status?.toLowerCase() === IngestionStatus.PROCESSING ||
            item.status?.toLowerCase() ===
              IngestionStatus.EXTRACTION_COMPLETED ||
            item.status?.toLowerCase() === IngestionStatus.SPLITTING,
        )
      ) {
        setIngestionUIState(IngestionStatus.SUCCESS); // Show table even when processing
        return IngestionStatus.PROCESSING;
      }

      if (
        statusList.some((item) =>
          [IngestionStatus.FAILED, IngestionStatus.EXCEPTION].includes(
            item.status?.toLowerCase(),
          ),
        )
      ) {
        setIngestionUIState(IngestionStatus.SUCCESS); // Show table even when failed
        return IngestionStatus.FAILED;
      }

      if (
        statusList.length > 0 &&
        statusList.every(
          (item) => item.status?.toLowerCase() === IngestionStatus.SUCCESS,
        )
      ) {
        setIngestionUIState(IngestionStatus.SUCCESS);
        return IngestionStatus.SUCCESS;
      }

      // If we have some data but status is unclear, try to load data
      try {
        await loadDataBySourceType();
        setIngestionUIState(IngestionStatus.SUCCESS);
        return IngestionStatus.SUCCESS;
      } catch (err) {
        setIngestionUIState(IngestionStatus.PROCESSING);
        return IngestionStatus.PROCESSING;
      }
    } catch (err) {
      if (err?.response?.status === 404) {
        setIngestionUIState(IngestionStatus.NOT_STARTED);
        return IngestionStatus.NOT_STARTED;
      }
      console.error("Error in status check:", err);
      setIngestionUIState(IngestionStatus.FAILED);
      return IngestionStatus.FAILED;
    }
  };

  const handleReingestFiles = async () => {
    try {
      identifyUserFromObject(user);
      const startTime = Date.now();

      // Track re-ingestion start
      captureEvent("ingestion_started", {
        workspace_id_hash: hashString(workspaceId || ""),
        dataset_id_hash: hashString(datasetId || ""),
        is_retry: true,
        description: "User starts re-ingestion process",
      });

      await invalidateGraphIfAny();
      await addChunk(datasetId, true);

      const duration_ms = Date.now() - startTime;

      // Track successful re-ingestion
      captureEvent("file_ingested", {
        workspace_id_hash: hashString(workspaceId || ""),
        dataset_id_hash: hashString(datasetId || ""),
        status: "re-ingestion_initiated",
        duration_ms: duration_ms,
        retry_count: 1,
        description: "Re-ingestion completes via backend",
      });

      showSuccess("Re-ingestion started successfully!");
      await checkIngestionStatus();
      router.push(`/workspace/${workspaceId}/datasets/${datasetId}`);
    } catch (err) {
      console.error("Error re-ingesting files:", err);

      // Track failed re-ingestion
      captureEvent("file_ingested", {
        workspace_id_hash: hashString(workspaceId || ""),
        dataset_id_hash: hashString(datasetId || ""),
        status: "failed",
        duration_ms: Date.now(),
        retry_count: 1,
        description: "Re-ingestion failed during backend processing",
      });

      showError("Failed to start re-ingestion. Please try again.");
    }
  };

  const handleRetryTraining = async () => {
    try {
      const train = await fetchTrainingDetails();
      console.log("Training Details", train.data.data);

      // Extract training data including documentation and question_sql_pairs
      const trainingData = train.data.data;
      const trainingInfo =
        trainingData && trainingData.length > 0 ? trainingData[0] : null;

      // Set the training details state to pass to the modal
      setTrainingDetails(trainingInfo);
      setShowSqlTrainingModal(true);
    } catch (error) {
      console.error("Error fetching training details:", error);
      setShowSqlTrainingModal(true);
    }
  };

  const isPreviewDisabled = (status) => {
    if (!status) return true;
    return status.toLowerCase() !== IngestionStatus.SUCCESS.toLowerCase();
  };

  const isDeleteDisabled = (status) => {
    if (!status) return true;
    const lowerStatus = status.toLowerCase();
    return ![
      IngestionStatus.SUCCESS.toLowerCase(),
      IngestionStatus.FAILED.toLowerCase(),
    ].includes(lowerStatus);
  };

  const areAllFilesCompleted = () => {
    const files = getMergedFileData();
    if (!files || files.length === 0) return true;

    return files.every((file) => {
      const status = file?.status?.toLowerCase();
      return status === IngestionStatus.SUCCESS.toLowerCase();
    });
  };

  const hasAnyFailedFiles = () => {
    const files = getMergedFileData();
    if (!files || files.length === 0) return false;

    return files.some((file) => {
      const status = file?.status?.toLowerCase();
      return status === IngestionStatus.FAILED.toLowerCase();
    });
  };

  const handleDeleteFile = (fileId) => {
    // Find the file data to show filename in confirmation
    const fileData = getMergedFileData().find(
      (file) => file?.file_id === fileId,
    );

    if (!fileData) {
      showError("File not found in this dataset");
      return;
    }

    // Check if file can be deleted
    if (isDeleteDisabled(fileData.status)) {
      showError(
        "File cannot be deleted at this time. Please wait for processing to complete.",
      );
      return;
    }

    setFileToDelete({
      id: fileId,
      filename: fileData?.filename || "this file",
    });
    setShowDeleteModal(true);
  };

  const confirmDeleteFile = async () => {
    if (!fileToDelete) return;

    try {
      setIsDeleting(true);
      // Store the deleted file ID for verification in refresh logic
      setLastDeletedFileId(fileToDelete.id);

      // Call the delete files API
      await deleteFiles(workspaceId, datasetId, [fileToDelete.id]);

      // Track dataset file deleted event
      identifyUserFromObject(user);
      captureEvent("dataset_file_deleted", {
        dataset_id: datasetId,
        file_id: fileToDelete.id,
        file_name: fileToDelete.filename,
        user_id: hashString(user?.clientId || ""),
        description: "User clicks delete (trash icon)",
      });

      showSuccess("File deleted successfully");
      // Immediately reflect local files count so UI toggles to Add Files if last one
      setFilesCount((prev) => Math.max((prev || 1) - 1, 0));
      // Invalidate graph since dataset content changed
      const graphDeleted = await invalidateGraphIfAny();
      if (graphDeleted) {
        showSuccess(
          "Dataset files modified - previous graph deleted, generate new graph",
        );
      }
      // Refresh the data after successful deletion - exactly like dataset deletion

      // Trigger data refresh using useEffect pattern (same as dataset page)
      setRefreshTrigger((prev) => prev + 1);
    } catch (err) {
      showError(
        "Failed to delete file: " +
          (err.response?.data?.detail ?? err.detail ?? "Unknown error"),
      );
      setLastDeletedFileId(null); // Clear on error
    } finally {
      setShowDeleteModal(false);
      setFileToDelete(null);
      setIsDeleting(false);
    }
  };

  const handlePreviewChunks = async (fileId) => {
    try {
      // Find the file data from merged data
      const fileData = getMergedFileData().find(
        (file) => file?.file_id === fileId,
      );

      if (!fileData) {
        showError("File not found in this dataset");
        return;
      }

      // Check if file is successful before fetching chunks
      if (
        fileData.status?.toLowerCase() !== IngestionStatus.SUCCESS.toLowerCase()
      ) {
        showError(
          "Chunk details are only available for successfully ingested files",
        );
        return;
      }

      // Set initial data and show modal with loading state
      setSelectedFileData({
        ...fileData,
        chunks: [],
      });
      setIsLoadingChunks(true);
      setShowChunksModal(true);

      // Fetch chunk details for this specific file
      const chunkRes = await getDatasetChuckDetails(
        workspaceId,
        datasetId,
        fileId,
      );
      const chunkData = chunkRes?.data?.data?.items?.[0] || {};

      setSelectedFileData({
        ...fileData,
        chunks: chunkData?.chunks || [],
      });
    } catch (err) {
      console.error("Error previewing chunks:", err);
      showError("Failed to load chunk details. Please try again.");
      setShowChunksModal(false);
    } finally {
      setIsLoadingChunks(false);
    }
  };

  // Delete existing graph (if any) and reset graph-related UI state
  const invalidateGraphIfAny = async () => {
    try {
      if (currentGraphId) {
        await deleteGraph(datasetId, currentGraphId);
        updateGraphStatus(null, null, null);
        setCurrentGraphId(null);
        setGraphRelationships(null);
        setIsGeneratingEntities(false);
        setIsGeneratingRelationships(false);
        setIsFetchingStatus(false);
        setShowAddGraphModal(false);
        setShowGraphSummaryModal(false);
        setOriginalSelectionMethod(null);
        // Reset previous status refs since graph is gone
        prevEntitiesStatusRef.current = null;
        prevRelationshipsStatusRef.current = null;
        return true;
      }
    } catch (e) {
      console.error("Failed to invalidate graph:", e);
    }
    return false;
  };
  const getMergedFileData = () => {
    if (!ingestionStatusData || ingestionStatusData.length === 0) {
      // If no ingestion status data, try to use chunk details directly
      if (chunkDetails && chunkDetails.length > 0) {
        return chunkDetails.map((chunkItem) => ({
          file_id: chunkItem.file_id || chunkItem.filename,
          filename: chunkItem.filename,
          status: "processing", // Default status
          created_at: new Date().toISOString(), // Default date
          chunks_count: chunkItem.total_chunks || 0,
          file_size: chunkItem.file_size || null,
          type:
            chunkItem.filename?.split(".").pop()?.toLowerCase() || "unknown", // Add type field
        }));
      }
      return [];
    }

    return ingestionStatusData.map((statusItem) => {
      // Find corresponding chunk details for this file
      const chunkItem = chunkDetails?.find(
        (chunk) =>
          chunk.file_id === statusItem.file_id ||
          chunk.filename === statusItem.filename,
      );

      return {
        ...statusItem,
        chunks_count: chunkItem?.total_chunks || 0,
        file_size: chunkItem?.file_size || null,
        type: statusItem.filename?.split(".").pop()?.toLowerCase() || "unknown", // Add type field
      };
    });
  };

  const dataWithComputedFields = (fileMetaData || []).map((file) => ({
    ...file,
    type: file?.filename?.split(".").pop()?.toLowerCase() || "",
  }));

  console.log("fileMetaData:", fileMetaData);
  console.log("dataWithComputedFields:", dataWithComputedFields);
  console.log("isStructured:", isStructured);

  // Use different data sources for structured vs unstructured
  const structuredSortedData = getSortedData(
    dataWithComputedFields,
    sortField,
    sortDirection,
  );

  const mergedFileData = getMergedFileData();
  console.log("getMergedFileData():", mergedFileData);
  console.log("ingestionStatusData:", ingestionStatusData);
  console.log("chunkDetails:", chunkDetails);

  const unstructuredSortedData = getSortedData(
    mergedFileData,
    sortField,
    sortDirection,
  );

  const toggleDropdown = (fileId) => {
    setOpenDropdown(openDropdown === fileId ? null : fileId);
  };

  const getStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case "processing":
        return "bg-blue-100 text-custom-Processing";
      case "failed":
        return "bg-red-100 text-red-700";
      case "injested":
      case IngestionStatus.SUCCESS:
        return "bg-green-100 text-custom-Success";
      default:
        return "bg-gray-100 text-gray-700";
    }
  };

  const getStatusIcon = (status) => {
    switch (status?.toLowerCase()) {
      case "processing":
        return (
          <Image
            src={processIcon}
            alt="Processing"
            width={16}
            height={16}
            className="animate-spin"
          />
        );
      case "failed":
        return <Image src={failedIcon} alt="Failed" width={16} height={16} />;
      case "injested":
      case IngestionStatus.SUCCESS:
        return <Image src={successIcon} alt="Success" width={16} height={16} />;
      default:
        return null;
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return "-";
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", {
      month: "2-digit",
      day: "2-digit",
      year: "numeric",
    });
  };

  const handleGenerateGraphFromAddModal = (relationships, graphId) => {
    setGraphRelationships(relationships);
    setCurrentGraphId(graphId);
    setShowAddGraphModal(false);
    setShowGraphSummaryModal(true);
  };

  // Function to store the selection method when user starts entity extraction
  const handleSetOriginalSelectionMethod = (method) => {
    setOriginalSelectionMethod(method);
  };

  // Function to handle graph deletion and refresh the page
  const handleGraphDeleted = async () => {
    console.log("GRAPH DELETED: Refreshing dataset page...");
    try {
      // Refresh the existing graphs to get the updated state
      await fetchExistingGraphs();
      console.log("GRAPH DELETED: Dataset page refreshed successfully");
    } catch (error) {
      console.error("GRAPH DELETED: Error refreshing dataset page:", error);
    }
  };

  // Function to fetch existing graphs for the dataset
  const fetchExistingGraphs = async () => {
    try {
      const graphsResponse = await getAllGraphsByDatasetId(datasetId);

      // If graph exists, set the graph ID and status
      if (graphsResponse && graphsResponse.id) {
        // Detect original selection method based on entity_types field
        const hasEntityTypes =
          graphsResponse.entity_types &&
          Array.isArray(graphsResponse.entity_types) &&
          graphsResponse.entity_types.length > 0;
        const detectedSelectionMethod = hasEntityTypes ? "manual" : "auto";

        setOriginalSelectionMethod(detectedSelectionMethod);

        updateGraphStatus(
          graphsResponse.id,
          graphsResponse.entities_status,
          graphsResponse.relationships_status,
        );

        // Toast on transition to FAILED for entities/relationships
        const entitiesStatus = graphsResponse.entities_status;
        const relationshipsStatus = graphsResponse.relationships_status;

        if (
          entitiesStatus === GraphStatus.FAILED &&
          prevEntitiesStatusRef.current !== GraphStatus.FAILED
        ) {
          showError("Entities extraction failed");
        }
        if (
          relationshipsStatus === GraphStatus.FAILED &&
          prevRelationshipsStatusRef.current !== GraphStatus.FAILED
        ) {
          showError("Relationship extraction failed");
        }

        prevEntitiesStatusRef.current = entitiesStatus;
        prevRelationshipsStatusRef.current = relationshipsStatus;

        // Detect if generation is in progress and set context flags accordingly
        if (entitiesStatus === GraphStatus.PENDING) {
          // Entities are still being generated
          setIsGeneratingEntities(true);
          setIsGeneratingRelationships(false);
          console.log("Detected ongoing entity generation");
        } else if (
          entitiesStatus === GraphStatus.SUCCESS &&
          relationshipsStatus === GraphStatus.PENDING
        ) {
          // Entities complete, relationships being generated
          setIsGeneratingEntities(false);
          setIsGeneratingRelationships(true);
          console.log("Detected ongoing relationship generation");
        } else if (
          entitiesStatus === GraphStatus.SUCCESS &&
          (relationshipsStatus === GraphStatus.NOT_STARTED ||
            relationshipsStatus === GraphStatus.FAILED)
        ) {
          // Entities complete, relationships not started or failed
          setIsGeneratingEntities(false);
          setIsGeneratingRelationships(false);
          console.log("Entities extracted, relationships not started/failed");
        } else if (
          entitiesStatus === GraphStatus.SUCCESS &&
          relationshipsStatus === GraphStatus.SUCCESS
        ) {
          // Both entities and relationships complete
          setIsGeneratingEntities(false);
          setIsGeneratingRelationships(false);
          console.log("Both entities and relationships complete");
        } else if (entitiesStatus === GraphStatus.FAILED) {
          // Entity generation failed
          setIsGeneratingEntities(false);
          setIsGeneratingRelationships(false);
          console.log("Entity generation failed");
        } else {
          // No generation in progress or completed states
          setIsGeneratingEntities(false);
          setIsGeneratingRelationships(false);
        }
      } else {
        updateGraphStatus(null, null, null);
        setOriginalSelectionMethod(null);
        setIsGeneratingEntities(false);
        setIsGeneratingRelationships(false);
        prevEntitiesStatusRef.current = null;
        prevRelationshipsStatusRef.current = null;
      }
    } catch (error) {
      // Handle 404 or other errors
      if (error?.response?.status === 404) {
        updateGraphStatus(null, null, null);
        setOriginalSelectionMethod(null);
        setIsGeneratingEntities(false);
        setIsGeneratingRelationships(false);
        prevEntitiesStatusRef.current = null;
        prevRelationshipsStatusRef.current = null;
      } else {
        console.error("Error fetching existing graphs:", error);
        updateGraphStatus(null, null, null);
        setOriginalSelectionMethod(null);
        setIsGeneratingEntities(false);
        setIsGeneratingRelationships(false);
        prevEntitiesStatusRef.current = null;
        prevRelationshipsStatusRef.current = null;
      }
    }
  };

  // Function to fetch graph relationships and show summary modal
  const handleShowGraph = async () => {
    if (!currentGraphId) {
      showError("No graph available to display");
      return;
    }

    try {
      const relationships = await getGraphEntitiesRelationships(
        datasetId,
        currentGraphId,
      );

      setGraphRelationships(relationships);
      setShowGraphSummaryModal(true);
      showSuccess("Graph loaded successfully");
    } catch (error) {
      console.error("Failed to fetch graph relationships:", error);
      showError("Failed to load graph. Please try again.");
    }
  };

  // Function to handle fetch status button click
  const handleFetchStatus = async () => {
    if (isFetchingStatus) return; // Prevent multiple simultaneous calls

    try {
      setIsFetchingStatus(true);

      await fetchExistingGraphs();
      // Show success only when no failures are present.
      if (
        graphEntitiesStatus !== GraphStatus.FAILED &&
        graphRelationshipsStatus !== GraphStatus.FAILED
      ) {
        showSuccess("Graph status updated successfully");
      }
    } catch (error) {
      console.error("Failed to fetch graph status:", error);
      showError("Failed to fetch graph status. Please try again.");
    } finally {
      setIsFetchingStatus(false);
    }
  };

  useEffect(() => {
    let intervalId;

    const fetchIfSource = async () => {
      try {
        // Reset graph context states at the beginning of each fetch
        console.log(
          "Resetting graph context states on component mount/refresh",
        );
        setIsGeneratingEntities(false);
        setIsGeneratingRelationships(false);
        setIsFetchingStatus(false);

        const { sourceType: fetchedSourceType } =
          await fetchDatasetAndSourceInfo();

        if (
          fetchedSourceType === constants.SOURCE_TYPE.POSTGRES ||
          fetchedSourceType === constants.SOURCE_TYPE.MYSQL
        ) {
          await loadDataBySourceType();
          setIngestionUIState(IngestionStatus.SUCCESS);
        } else {
          // checkIngestionStatus will handle selective chunk fetching for successful files
          const status = await checkIngestionStatus();

          if (status === IngestionStatus.PROCESSING) {
            // Set state to success to show the table even when processing
            setIngestionUIState(IngestionStatus.SUCCESS);
            intervalId = setInterval(async () => {
              const nextStatus = await checkIngestionStatus();
              if (
                nextStatus === IngestionStatus.SUCCESS ||
                nextStatus === IngestionStatus.FAILED
              ) {
                clearInterval(intervalId);
                // Refresh the data when status changes
                await checkIngestionStatus();
              }
            }, 20000);
          }

          if (status === IngestionStatus.SUCCESS) {
            setIngestionUIState(IngestionStatus.SUCCESS);
          }
        }

        // Fetch existing graphs after data is loaded
        await fetchExistingGraphs();
      } catch (err) {
        setChunkDetails([]);
        setFileMetaData([]);
        showError(`${err?.response?.data?.detail}`);
        console.error("Error loading component", err);
      } finally {
        setLoader(false); // Ensure loader is set to false even on error
      }
    };

    fetchIfSource();

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [params, pagination]);

  // Cleanup delete state when component unmounts
  useEffect(() => {
    return () => {
      setShowDeleteModal(false);
      setFileToDelete(null);
      setIsDeleting(false);
      setLastDeletedFileId(null);
      // Reset graph context states on unmount
      console.log("Component unmounting - resetting graph context states");
      setIsGeneratingEntities(false);
      setIsGeneratingRelationships(false);
      setIsFetchingStatus(false);
    };
  }, []);

  // Refresh data when refreshTrigger changes (after file deletion)
  useEffect(() => {
    if (refreshTrigger > 0) {
      // Call all the necessary APIs to refresh the data (same as dataset page)
      const refreshData = async () => {
        try {
          // First refresh dataset info to update filesCount
          const { filesCount: currentFilesCount } =
            await fetchDatasetAndSourceInfo();

          // Check if we have any files left after deletion
          if (currentFilesCount === 0) {
            // No files left - clear ingestion data and skip API call
            console.log("No files remaining, clearing ingestion status data");
            setIngestionStatusData([]);
            setLastDeletedFileId(null);
            return;
          }

          // Get updated ingestion status
          try {
            const statusResponse = await getStatus(datasetId);
            const statusList = statusResponse?.data?.data || [];
            setIngestionStatusData(statusList);
          } catch (error) {
            // Handle 404 errors gracefully - they're expected when no files exist
            if (error?.response?.status === 404) {
              console.log("Ingestion status API returned 404 - no files exist");
              setIngestionStatusData([]);
            } else {
              throw error;
            }
          }

          // Clear the deleted file ID after successful refresh
          setLastDeletedFileId(null);
        } catch (error) {
          console.error("Error refreshing data:", error);
          // Fallback to the original method
          loadDataBySourceType();
        }
      };

      refreshData();
    }
  }, [refreshTrigger, datasetId]);

  // Invalidate graph when new files are added via the edit/add-more flow
  const newDataAddedInitRef = useRef(true);
  const [shouldInvalidateGraph, setShouldInvalidateGraph] = useState(false);

  useEffect(() => {
    if (newDataAddedInitRef.current) {
      newDataAddedInitRef.current = false;
      return;
    }
    // Only invalidate graph if explicitly flagged to do so
    if (shouldInvalidateGraph) {
      (async () => {
        console.log("Invalidating graph due to new files being added");
        await invalidateGraphIfAny();
        setShouldInvalidateGraph(false); // Reset the flag
      })();
    }
  }, [newDataAdded, shouldInvalidateGraph]);

  return (
    <div className="relative p-8 min-h-screen">
      {/* Header - Always visible */}
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Button
              variant="dataset"
              onClick={() => router.back()}
              className="p-2 rounded-lg transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-gray-600" />
            </Button>
            <div>
              <h1 className="text-2xl font-semibold text-gray-900">
                {datasetName}
              </h1>
            </div>
          </div>

          {/* Show buttons only when not loading and not structured */}
          {!loader && (
            <div className="flex items-center space-x-3">
              {/* Only show retry button if not all files are successful or if it's structured data */}
              {(isStructured || !areAllFilesCompleted()) && (
                <Button
                  variant="outline"
                  className="flex items-center space-x-2 border-gray-300 text-gray-700"
                  onClick={
                    isStructured ? handleRetryTraining : handleReingestFiles
                  }
                >
                  <RefreshCw className="w-4 h-4" />
                  <span>
                    {isStructured ? "Retrain Dataset" : "Retry Ingestion"}
                  </span>
                </Button>
              )}

              {(sourceType === constants.SOURCE_TYPE.AZURE ||
                sourceType === null) &&
                filesCount > 0 && (
                  <>
                    {/* Show "Add Graph to Dataset" button when files are displayed in table */}
                    <Button
                      className={`bg-blue-600 hover:bg-blue-700 text-white ${
                        !areAllFilesCompleted() ||
                        isGeneratingEntities ||
                        isGeneratingRelationships ||
                        (currentGraphId &&
                          graphEntitiesStatus === GraphStatus.SUCCESS &&
                          graphRelationshipsStatus === GraphStatus.PENDING)
                          ? "opacity-50 cursor-not-allowed"
                          : ""
                      }`}
                      onClick={() => {
                        if (
                          areAllFilesCompleted() &&
                          !isGeneratingEntities &&
                          !isGeneratingRelationships &&
                          !(
                            currentGraphId &&
                            graphEntitiesStatus === GraphStatus.SUCCESS &&
                            graphRelationshipsStatus === GraphStatus.PENDING
                          )
                        ) {
                          // Check if graph exists with both statuses as success
                          if (
                            currentGraphId &&
                            graphEntitiesStatus === GraphStatus.SUCCESS &&
                            graphRelationshipsStatus === GraphStatus.SUCCESS
                          ) {
                            handleShowGraph();
                          } else {
                            // Track add graph clicked event
                            identifyUserFromObject(user);
                            captureEvent("add_graph_clicked", {
                              dataset_id: datasetId,
                              user_id: hashString(user?.clientId || ""),
                              description:
                                "User clicks 'Add Graph to Dataset' button",
                            });

                            // Open AddGraphModal for new graph or entities extracted case
                            setShowAddGraphModal(true);
                          }
                        }
                      }}
                      disabled={
                        !areAllFilesCompleted() ||
                        isGeneratingEntities ||
                        isGeneratingRelationships ||
                        (currentGraphId &&
                          graphEntitiesStatus === GraphStatus.SUCCESS &&
                          graphRelationshipsStatus === GraphStatus.PENDING)
                      }
                    >
                      <BarChart3 className="w-4 h-4 mr-2" />
                      {(() => {
                        const buttonText = isGeneratingEntities
                          ? "Generating Entities..."
                          : isGeneratingRelationships
                            ? "Generating Relationships..."
                            : currentGraphId &&
                                graphEntitiesStatus === GraphStatus.SUCCESS &&
                                graphRelationshipsStatus === GraphStatus.PENDING
                              ? "Generating Relationships..."
                              : currentGraphId &&
                                  graphEntitiesStatus === GraphStatus.SUCCESS &&
                                  graphRelationshipsStatus ===
                                    GraphStatus.SUCCESS
                                ? "Show Graph"
                                : currentGraphId &&
                                    graphEntitiesStatus ===
                                      GraphStatus.SUCCESS &&
                                    (graphRelationshipsStatus ===
                                      GraphStatus.NOT_STARTED ||
                                      graphRelationshipsStatus ===
                                        GraphStatus.FAILED)
                                  ? "Entities Extracted Build Graph"
                                  : "Add Graph to Dataset";

                        return buttonText;
                      })()}
                    </Button>
                  </>
                )}

              {(sourceType === constants.SOURCE_TYPE.AZURE ||
                sourceType === null) &&
                !hasAnyFailedFiles() &&
                areAllFilesCompleted() &&
                (isGeneratingEntities ||
                  isGeneratingRelationships ||
                  (currentGraphId &&
                    graphEntitiesStatus === GraphStatus.SUCCESS &&
                    graphRelationshipsStatus === GraphStatus.PENDING)) && (
                  <Button
                    className="bg-blue-600 hover:bg-blue-700 text-white"
                    onClick={handleFetchStatus}
                  >
                    {isFetchingStatus ? (
                      <>
                        <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                        Fetching...
                      </>
                    ) : (
                      "Fetch Status"
                    )}
                  </Button>
                )}
            </div>
          )}

          {/* Remove the loading state for buttons area completely */}
        </div>

        {/* Content based on ingestion state */}
        {loader ? (
          // Show the earlier, more pleasing loading state
          <div className="text-center py-12 text-gray-500">
            <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <BarChart3 className="w-8 h-8 text-gray-400" />
            </div>
            <p className="text-lg font-medium text-gray-900 mb-2">
              Loading dataset information...
            </p>
            <p className="text-gray-500">
              Please wait while we fetch your data...
            </p>
          </div>
        ) : (
          <>
            {/* Data Table */}
            {isStructured ? (
              <div className="bg-white rounded-lg border border-gray-200 w-full">
                <div className="w-full overflow-x-auto">
                  <Table className="w-full">
                    <TableHeader>
                      <TableRow className="bg-custom-tableHeader border-b border-gray-200">
                        <TableHead className="text-xs font-medium text-custom-tableColumn px-6 py-3 text-center">
                          <div className="flex items-center justify-center space-x-1">
                            <span></span>
                          </div>
                        </TableHead>
                        <TableHead className="text-s font-medium text-custom-tableColumn px-6 py-3 text-left">
                          <div className="flex items-center justify-start space-x-1">
                            <span className="font-bold">File Name</span>
                            <Button
                              variant="dataset"
                              onClick={() => handleSort("filename")}
                              className="ml-1 p-1 rounded "
                            >
                              <Image
                                src={arrowIcon}
                                alt="Arrow"
                                width={14}
                                height={14}
                                className={`transition-transform ${
                                  sortField === "filename" &&
                                  sortDirection === SortDirection.ASCENDING
                                    ? "rotate-180"
                                    : ""
                                }`}
                              />
                            </Button>
                          </div>
                        </TableHead>
                        <TableHead className="text-s font-medium px-6 py-3 text-center">
                          <div className="flex items-center justify-center space-x-1">
                            <span className="font-bold">Rows</span>
                            <Button
                              variant="dataset"
                              onClick={() => handleSort("rows")}
                              className="ml-1 p-1 rounded"
                            >
                              <Image
                                src={arrowIcon}
                                alt="Arrow"
                                width={14}
                                height={14}
                                className={`transition-transform ${
                                  sortField === "rows" &&
                                  sortDirection === SortDirection.ASCENDING
                                    ? "rotate-180"
                                    : ""
                                }`}
                              />
                            </Button>
                          </div>
                        </TableHead>
                        <TableHead className="text-s font-medium px-6 py-3 text-center">
                          <div className="flex items-center justify-center space-x-1">
                            <span className="font-bold">Columns</span>
                            <Button
                              variant="dataset"
                              onClick={() => handleSort("columns")}
                              className="ml-1 p-1 rounded"
                            >
                              <Image
                                src={arrowIcon}
                                alt="Arrow"
                                width={14}
                                height={14}
                                className={`transition-transform ${
                                  sortField === "columns" &&
                                  sortDirection === SortDirection.ASCENDING
                                    ? "rotate-180"
                                    : ""
                                }`}
                              />
                            </Button>
                          </div>
                        </TableHead>
                        <TableHead className="text-s font-medium px-6 py-3 text-center">
                          <div className="flex items-center justify-center space-x-1">
                            <span className="font-bold">Type</span>
                            <Button
                              variant="dataset"
                              onClick={() => handleSort("type")}
                              className="ml-1 p-1 rounded"
                            >
                              <Image
                                src={arrowIcon}
                                alt="Arrow"
                                width={14}
                                height={14}
                                className={`transition-transform ${
                                  sortField === "type" &&
                                  sortDirection === SortDirection.ASCENDING
                                    ? "rotate-180"
                                    : ""
                                }`}
                              />
                            </Button>
                          </div>
                        </TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {structuredSortedData.map((file, index) => (
                        <TableRow
                          key={file?.file_id || index}
                          className="border-b border-gray-100"
                        >
                          <TableCell className="px-6 py-4 text-sm text-gray-900 font-medium text-center border-r border-gray-300">
                            {index + 1}
                          </TableCell>
                          <TableCell className="px-6 py-4 text-sm text-gray-900 font-medium text-left">
                            {file?.filename}
                          </TableCell>
                          <TableCell className="px-6 py-4 text-sm text-gray-700 text-center">
                            {file?.rows || "-"}
                          </TableCell>
                          <TableCell className="px-6 py-4 text-sm text-gray-700 text-center">
                            {file?.columns || "-"}
                          </TableCell>
                          <TableCell className="px-6 py-4 text-sm text-gray-700 text-center">
                            {file?.filename?.split(".").pop()?.toUpperCase() ||
                              "-"}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </div>
            ) : ingestionStatusData?.length > 0 || chunkDetails?.length > 0 ? (
              <div className="bg-white rounded-lg border border-gray-200 w-full">
                <div className="w-full overflow-x-auto">
                  <Table className="w-full">
                    <TableHeader>
                      <TableRow className="bg-custom-tableHeader border-b border-gray-200">
                        <TableHead className="text-xs font-medium text-custom-tableColumn px-6 py-3 text-center">
                          <div className="flex items-center justify-center space-x-1">
                            <span></span>
                          </div>
                        </TableHead>
                        <TableHead className="text-s font-medium text-custom-tableColumn px-6 py-3 text-left">
                          <div className="flex items-center justify-start space-x-1">
                            <span className="font-bold">File Name</span>
                            <Button
                              variant="dataset"
                              onClick={() => handleSort("filename")}
                              className="ml-1 p-1 rounded "
                            >
                              <Image
                                src={arrowIcon}
                                alt="Arrow"
                                width={14}
                                height={14}
                                className={`transition-transform ${
                                  sortField === "filename" &&
                                  sortDirection === SortDirection.ASCENDING
                                    ? "rotate-180"
                                    : ""
                                }`}
                              />
                            </Button>
                          </div>
                        </TableHead>

                        <TableHead className="text-s font-medium px-6 py-3 text-center">
                          <div className="flex items-center justify-center space-x-1">
                            <span className="font-bold">Type</span>
                            <Button
                              variant="dataset"
                              onClick={() => handleSort("type")}
                              className="ml-1 p-1 rounded"
                            >
                              <Image
                                src={arrowIcon}
                                alt="Arrow"
                                width={14}
                                height={14}
                                className={`transition-transform ${
                                  sortField === "type" &&
                                  sortDirection === SortDirection.ASCENDING
                                    ? "rotate-180"
                                    : ""
                                }`}
                              />
                            </Button>
                          </div>
                        </TableHead>
                        <TableHead className="text-s font-medium px-6 py-3 text-center">
                          <div className="flex items-center justify-center space-x-1">
                            <span className="font-bold">Status</span>
                            <Button
                              variant="dataset"
                              onClick={() => handleSort("status")}
                              className="ml-1 p-1 rounded"
                            >
                              <Image
                                src={arrowIcon}
                                alt="Arrow"
                                width={14}
                                height={14}
                                className={`transition-transform ${
                                  sortField === "status" &&
                                  sortDirection === SortDirection.ASCENDING
                                    ? "rotate-180"
                                    : ""
                                }`}
                              />
                            </Button>
                          </div>
                        </TableHead>
                        <TableHead className="text-s font-medium px-6 py-3">
                          <div className="flex items-center justify-center space-x-1">
                            <span className="font-bold">Added</span>
                            <Button
                              variant="dataset"
                              onClick={() => handleSort("created_at")}
                              className="ml-1 p-1 rounded"
                            >
                              <Image
                                src={arrowIcon}
                                alt="Arrow"
                                width={14}
                                height={14}
                                className={`transition-transform ${
                                  sortField === "created_at" &&
                                  sortDirection === SortDirection.ASCENDING
                                    ? "rotate-180"
                                    : ""
                                }`}
                              />
                            </Button>
                          </div>
                        </TableHead>
                        <TableHead className="text-s font-medium px-6 py-3 text-center hover:bg-none">
                          <span className="font-bold">Preview</span>
                        </TableHead>
                        <TableHead className="text-s font-medium px-6 py-3 text-center hover:bg-none">
                          <span className="font-bold">Delete</span>{" "}
                        </TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {unstructuredSortedData.map((file, index) => (
                        <TableRow
                          key={file?.file_id || index}
                          className="border-b border-gray-100"
                        >
                          <TableCell className="px-6 py-4 text-sm text-gray-900 font-medium text-center border-r border-gray-300">
                            {index + 1}
                          </TableCell>
                          <TableCell className="px-6 py-4 text-sm text-gray-900 font-medium text-left">
                            {file?.filename}
                          </TableCell>
                          <TableCell className="px-6 py-4 text-sm text-gray-700 text-center">
                            {file?.type || "-"}
                          </TableCell>
                          <TableCell className="px-6 py-4 text-sm text-center">
                            <span
                              className={`inline-flex items-center w-23 h-7 justify-center px-2.5 py-0.5 rounded-[6px] text-xs font-medium ${getStatusColor(
                                file?.status,
                              )}`}
                            >
                              {getStatusIcon(file?.status)}
                              <span className="ml-1">{file?.status}</span>
                            </span>
                          </TableCell>
                          <TableCell className="px-6 py-4 text-sm text-gray-700 text-center">
                            {file?.created_at
                              ? formatDate(file.created_at)
                              : "-"}
                          </TableCell>
                          <TableCell className="px-6 py-4 text-sm text-gray-700 text-center">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handlePreviewChunks(file?.file_id)}
                              disabled={isPreviewDisabled(file?.status)}
                              className={`p-2 h-8 w-8 ${
                                isPreviewDisabled(file?.status)
                                  ? "opacity-50 cursor-not-allowed"
                                  : "hover:bg-blue-50"
                              }`}
                              title="Preview Chunks"
                            >
                              <Image
                                src={viewChunks}
                                alt="Preview"
                                width={16}
                                height={16}
                              />
                            </Button>
                          </TableCell>
                          <TableCell className="px-6 py-4 text-sm text-gray-700 text-center">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleDeleteFile(file?.file_id)}
                              disabled={isDeleteDisabled(file?.status)}
                              className={`p-2 h-8 w-8 ${
                                isDeleteDisabled(file?.status)
                                  ? "opacity-50 cursor-not-allowed"
                                  : "hover:bg-red-50"
                              }`}
                              title="Delete File"
                            >
                              <Image
                                src={trashIcon}
                                alt="Delete"
                                width={16}
                                height={16}
                              />
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </div>
            ) : (
              <div className="text-center py-12 text-gray-500">
                <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <BarChart3 className="w-8 h-8 text-gray-400" />
                </div>
                <p className="text-lg font-medium text-gray-900 mb-2">
                  No data available
                </p>
                <p className="text-gray-500 mb-6">
                  {filesCount === 0
                    ? "No files in this dataset."
                    : "Please wait while chunks are being fetched..."}
                </p>
                {/* Show "Add New File to Dataset" button in center when no files */}
                {filesCount === 0 &&
                  (sourceType === constants.SOURCE_TYPE.AZURE ||
                    sourceType === null) && (
                    <Button
                      className="bg-blue-600 hover:bg-blue-700 text-white"
                      onClick={() => setShowAddMoreFilesModal(true)}
                    >
                      <Plus className="w-4 h-4 mr-2" />
                      Add New File to Dataset
                    </Button>
                  )}
              </div>
            )}

            {/* Pagination */}
            {isStructured && totalPages > 1 && (
              <div className="flex justify-center">
                <Paginator
                  page={pagination}
                  size={"full"}
                  totalPages={totalPages}
                  showPageSize={true}
                  onChange={(opts) => setPagination(opts)}
                />
              </div>
            )}
          </>
        )}
      </div>

      {/* Modal for adding graph */}
      {showModal && (
        <div className="absolute inset-x-0 top-[250px] transform -translate-y flex justify-center bg-black bg-opacity-50 z-50">
          <div className="flex flex-col items-center space-y-4 p-6 bg-white rounded shadow-lg">
            <Loader className="animate-spin text-blue-500" size={48} />
            <p className="text-gray-700">
              Adding Graph in Dataset. Please wait...
            </p>
          </div>
        </div>
      )}

      {/* Modal for previewing chunks */}
      <LargeModal
        isOpen={showChunksModal}
        onClose={() => {
          setShowChunksModal(false);
          setIsLoadingChunks(false);
          setSelectedFileData(null);
        }}
        title="File Chunks Preview"
        fullWidth={true}
        fullHeight={true}
        hideSubmitButton={true}
      >
        <div className="p-6">
          {isLoadingChunks ? (
            <div className="flex flex-col items-center justify-center py-12">
              <div className="animate-spin h-8 w-8 border-4 border-gray-200 border-t-blue-600 rounded-full mb-4"></div>
              <p className="text-gray-600">Loading chunks...</p>
            </div>
          ) : (
            selectedFileData && (
              <div key={selectedFileData?.file_id}>
                <div className="text-base font-semibold flex items-end justify-between mb-2">
                  <div>{selectedFileData?.filename}</div>
                </div>
                {selectedFileData?.chunks?.length > 0 ? (
                  <Table className="border border-gray-300 rounded-2xl mh-[70vh] mb-3">
                    <TableHeader>
                      <TableRow className="border-b-2 border-gray-300">
                        <TableHead className="text-xs font-semibold bg-gray-200 ps-2" />
                        <TableHead className="text-xs font-semibold bg-gray-200 ps-6">
                          Chunks
                        </TableHead>
                        <TableHead className="text-xs font-semibold bg-gray-200 ps-2">
                          Vector Embeddings
                        </TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {selectedFileData?.chunks?.map((chuckData, i) => (
                        <TableRow
                          key={chuckData?.document_chunk_id || i}
                          className="border-b-2 border-gray-300 bg-white"
                        >
                          <TableCell className="py-3 px-4">{i + 1}</TableCell>
                          <TableCell className="py-3 px-4 truncate max-w-[300px]">
                            {chuckData?.text}
                          </TableCell>
                          <TableCell className="py-3 px-4 truncate max-w-[200px]">
                            {chuckData?.vector?.join(", ")}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    <p>No chunks found for this file.</p>
                  </div>
                )}
              </div>
            )
          )}
        </div>
      </LargeModal>

      {/* Modal for adding more files */}
      <LargeModal
        isOpen={showAddMoreFilesModal}
        onClose={() => {
          setShowAddMoreFilesModal(false);
          setHandleNextClick(null);
        }}
        title={"Add More Files"}
        fullWidth={true}
        fullHeight={true}
        actionButton="Next"
        onSubmit={() => {
          if (handleNextClick) {
            handleNextClick();
          }
        }}
        type="dataset"
      >
        <CreateDatasetForm
          onClose={() => {
            setShowAddMoreFilesModal(false);
            setHandleNextClick(null);
            // Refresh data when modal is closed (after ingestion is complete)
            loadDataBySourceType();
          }}
          workspaceId={workspaceId}
          datasetId={datasetId}
          setDatasetId={() => {}}
          setNewDataAdded={(value) => {
            setNewDataAdded(value);
            // Set flag to invalidate graph when new files are added
            setShouldInvalidateGraph(true);
            // Don't refresh data immediately when adding more files
            // Data will be refreshed after ingestion is complete
          }}
          newDataAdded={newDataAdded}
          formSubmit={setHandleNextClick}
          isEditMode={true}
          isAddMoreFilesMode={true}
        />
      </LargeModal>

      {showAddGraphModal && (
        <AddGraphModal
          isOpen={showAddGraphModal}
          onClose={() => setShowAddGraphModal(false)}
          onGenerateGraph={handleGenerateGraphFromAddModal}
          onSetOriginalSelectionMethod={handleSetOriginalSelectionMethod}
          datasetId={datasetId}
          existingGraphId={
            currentGraphId &&
            graphEntitiesStatus === GraphStatus.SUCCESS &&
            (graphRelationshipsStatus === GraphStatus.PENDING ||
              graphRelationshipsStatus === GraphStatus.NOT_STARTED ||
              graphRelationshipsStatus === GraphStatus.FAILED)
              ? currentGraphId
              : null
          }
          entitiesExtracted={
            currentGraphId &&
            graphEntitiesStatus === GraphStatus.SUCCESS &&
            (graphRelationshipsStatus === GraphStatus.PENDING ||
              graphRelationshipsStatus === GraphStatus.NOT_STARTED ||
              graphRelationshipsStatus === GraphStatus.FAILED)
          }
          originalSelectionMethod={originalSelectionMethod}
        />
      )}
      {showGraphSummaryModal && (
        <GraphSummaryModal
          isOpen={showGraphSummaryModal}
          onClose={() => setShowGraphSummaryModal(false)}
          onRegenerate={() => {
            setShowAddGraphModal(true);
            setGraphRelationships(null);
          }}
          onGraphDeleted={handleGraphDeleted}
          datasetId={datasetId}
          graphId={currentGraphId}
          relationships={graphRelationships}
        />
      )}

      {/* Delete Confirmation Modal */}
      <DeleteModal
        isOpen={showDeleteModal}
        onClose={() => setShowDeleteModal(false)}
        onDelete={confirmDeleteFile}
        title={`Delete "${fileToDelete?.filename}"?`}
      />

      {/* Ingestion Form Modal for Retry Training */}
      <Modal
        isOpen={showReingestModal}
        onClose={() => setShowReingestModal(false)}
        title="Retry Training Configuration"
        size="lg"
      >
        <IngestionForm
          setIsOpen={setShowReingestModal}
          dataSetId={datasetId}
          workspaceId={workspaceId}
          datasetId={datasetId}
          onClose={() => setShowReingestModal(false)}
          setNewDataAdded={(value) => {
            setNewDataAdded(value);
            // Don't set shouldInvalidateGraph for retry flows
            // They handle graph invalidation directly in handleReingestFiles
          }}
          newDataAdded={newDataAdded}
          isFirstTime={false}
          isEditFlow={true}
          isUpdateFlow={false}
        />
      </Modal>

      {/* SQL Training Form Modal for Retry Training */}
      <Modal
        isOpen={showSqlTrainingModal}
        onClose={() => setShowSqlTrainingModal(false)}
        title="SQL Training Configuration"
        size="lg"
      >
        <SqlTrainingForm
          setIsOpen={setShowSqlTrainingModal}
          dataSetId={datasetId}
          workspaceId={workspaceId}
          datasetId={datasetId}
          onClose={() => setShowSqlTrainingModal(false)}
          setNewDataAdded={(value) => {
            setNewDataAdded(value);
            // Don't set shouldInvalidateGraph for SQL training retry flows
            // They don't need graph invalidation
          }}
          newDataAdded={newDataAdded}
          submitButton={"Retry Training"}
          initialTrainingData={trainingDetails}
        />
      </Modal>
    </div>
  );
};

export default Datasets;
