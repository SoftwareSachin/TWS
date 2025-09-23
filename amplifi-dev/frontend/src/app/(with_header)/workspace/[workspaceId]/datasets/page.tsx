"use client";

import React, { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  createDataSet,
  deleteDataSet,
  getDataSet,
  updateDataSet,
} from "@/api/dataset";
import { showError, showSuccess } from "@/utils/toastUtils";
import DeleteModal from "@/components/forms/deleteModal";
import LargeModal from "@/components/forms/largeModal";

import { Dataset, DatasetResponse } from "@/types/Dataset";
import CreateDatasetForm from "@/components/forms/createDatasetForm";
import WorkspaceUtilCard from "@/components/ui/WorkspaceUtilCard";
import WorkspacePageWrapper from "@/components/Agentic/layout";
import { getStatus } from "@/api/Workspace/workspace";
import { IngestionStatus } from "@/lib/constants";
import {
  identifyUserFromObject,
  captureEvent,
  hashString,
} from "@/utils/posthogUtils";
import { useUser } from "@/context_api/userContext";

const Datasets = () => {
  const { user } = useUser();
  const [datasets, setDatasets] = useState<DatasetResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [pageError, setPageError] = useState<string | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(
    null,
  );
  const [datasetToEdit, setDatasetToEdit] = useState<DatasetResponse | null>(
    null,
  );
  const [searchTerm, setSearchTerm] = useState("");
  const [isReadOnlyMode, setIsReadOnlyMode] = useState(false);
  const [handleNextClick, setHandleNextClick] = useState<(() => void) | null>(
    null,
  );

  const [pagination, setPagination] = useState({
    page: 1,
    size: 50,
  });

  const [totalPages, setTotalPages] = useState(1);
  const [datasetStatuses, setDatasetStatuses] = useState<{
    [key: string]: boolean;
  }>({});

  const params = useParams();

  const isDatasetEditable = async (datasetId: string) => {
    try {
      const statusResponse = await getStatus(datasetId);
      const statusList = statusResponse?.data?.data || [];

      // If no files, consider it editable
      if (statusList.length === 0) return true;

      // Check if all files are in final state (success or failed)
      const allFilesCompleted = statusList.every((item: { status: string }) => {
        const status = item.status?.toLowerCase();
        return (
          status === IngestionStatus.SUCCESS.toLowerCase() ||
          status === IngestionStatus.FAILED.toLowerCase()
        );
      });

      return allFilesCompleted;
    } catch (err: any) {
      if (err?.response?.status === 404) {
        // If 404, it means no ingestion status - consider it editable
        return true;
      }
      console.error("Error checking dataset status:", err);
      return false;
    }
  };
  const router = useRouter();
  const workspaceId = Array.isArray(params.workspaceId)
    ? params.workspaceId[0]
    : params.workspaceId;

  const fetchDatasets = async () => {
    setLoading(true);
    try {
      const res = await getDataSet(
        workspaceId,
        pagination,
        undefined,
        undefined,
        searchTerm,
      );
      identifyUserFromObject(user);

      if (res.status === 200) {
        const items: DatasetResponse[] = res.data.data.items;
        setDatasets(items);
        const totalDatasets = res.data?.data?.total ?? 0;
        const calculatedTotalPages = Math.ceil(totalDatasets / pagination.size);
        setTotalPages(calculatedTotalPages);

        // Check status for each dataset
        const statuses: { [key: string]: boolean } = {};
        await Promise.all(
          items.map(async (dataset) => {
            statuses[dataset.id] = await isDatasetEditable(dataset.id);
          }),
        );
        setDatasetStatuses(statuses);
      }
    } catch (err: any) {
      setPageError(err.response?.data?.detail ?? err.detail ?? "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  // Function to check if any dataset is still processing
  const hasProcessingDatasets = () => {
    return Object.values(datasetStatuses).some((status) => !status);
  };

  useEffect(() => {
    let intervalId: NodeJS.Timeout;

    const fetchAndPoll = async () => {
      await fetchDatasets();

      // Track dataset list viewed event
      if (datasets.length > 0) {
        identifyUserFromObject(user);
        captureEvent("dataset_list_viewed", {
          dataset_count: datasets.length,
          user_id: hashString(user?.clientId || ""),
          description: "When user opens dataset list page",
        });
      }

      // If any dataset is still processing, start polling
      if (hasProcessingDatasets()) {
        intervalId = setInterval(async () => {
          await fetchDatasets();
          // If no more processing datasets, stop polling
          if (!hasProcessingDatasets()) {
            clearInterval(intervalId);
          }
        }, 20000); // Poll every 20 seconds
      }
    };

    if (workspaceId) {
      fetchAndPoll();
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [workspaceId, pagination]);

  const handleCreateOrUpdateDataset = async (data: any) => {
    try {
      const datasetPayload = {
        name: data.name,
        description: data.description,
        file_ids: data.file_ids || [],
        source_id: data.source_id,
        source_type: data.source_type,
      };

      if (datasetToEdit) {
        // Update existing dataset
        const res = await updateDataSet(
          workspaceId,
          datasetToEdit.id,
          datasetPayload,
        );
        if (res.status === 200) {
          showSuccess("Dataset updated successfully");
        }
      } else {
        // Create new dataset
        const res = await createDataSet(workspaceId, datasetPayload);
        if (res.status === 200) {
          // Track dataset created event
          identifyUserFromObject(user);
          captureEvent("dataset_created", {
            dataset_name: data.name,
            user_id: hashString(user?.clientId || ""),
            description: "Create Dataset",
          });

          showSuccess("Dataset created successfully");
        }
      }
      setIsModalOpen(false);
      setDatasetToEdit(null);
      await fetchDatasets();
    } catch (err: any) {
      showError(err.response?.data?.detail ?? err.detail ?? "Unknown error");
    }
  };

  const handleOpenEditModal = async (dataset: DatasetResponse) => {
    setDatasetToEdit(dataset);
    setIsModalOpen(true);
  };

  const handleShowDetails = async (dataset: DatasetResponse) => {
    identifyUserFromObject(user);
    captureEvent("dataset_viewed", {
      dataset_id: hashString(dataset.id),
      source_id: hashString(dataset.source_type || ""),
      created_by_id: hashString(user?.clientId || ""),
      description: "User opens a dataset page",
    });
    router.push(`/workspace/${workspaceId}/datasets/${dataset.id}`);
  };

  const handleOpenDeleteModal = (datasetId: string) => {
    setSelectedDatasetId(datasetId);
    setIsDeleteModalOpen(true);
  };

  const handleDeleteDataset = async () => {
    setLoading(true);
    if (!selectedDatasetId) return;
    try {
      await deleteDataSet(workspaceId, selectedDatasetId);
      showSuccess("Dataset deleted successfully");
      await fetchDatasets();
    } catch (err: any) {
      showError(
        "Failed to delete dataset: " +
          (err.response?.data?.detail ?? err.detail),
      );
    } finally {
      setIsDeleteModalOpen(false);
      setSelectedDatasetId(null);
      setLoading(false);
    }
  };

  // Calculate filtered datasets
  const filteredDatasets = datasets.filter((dataset) =>
    dataset.name.toLowerCase().includes(searchTerm.toLowerCase()),
  );

  return (
    <WorkspacePageWrapper
      title="Datasets"
      itemCount={filteredDatasets.length}
      searchTerm={searchTerm}
      onSearchChange={(term) => {
        setSearchTerm(term);
        if (term) {
          identifyUserFromObject(user);
          const filteredCount = datasets.filter((dataset) =>
            dataset.name.toLowerCase().includes(term.toLowerCase()),
          ).length;
          captureEvent("dataset_search", {
            result_count: filteredCount,
            user_id: hashString(user?.clientId || ""),
            description: "Search bar query is submitted",
          });
        }
      }}
      onCreateClick={() => {
        setDatasetToEdit(null);
        setIsModalOpen(true);
      }}
      renderItems={() =>
        filteredDatasets.map((dataset, idx) => (
          <WorkspaceUtilCard
            key={idx}
            title={dataset.name}
            description={`${dataset.description || ""}`}
            tag="files"
            allToolNames={[`${dataset.file_ids?.length || 0} files`]}
            actionText="Show Files"
            onDelete={
              datasetStatuses[dataset.id]
                ? () => handleOpenDeleteModal(dataset.id)
                : undefined
            }
            onEdit={
              datasetStatuses[dataset.id]
                ? () => handleOpenEditModal(dataset)
                : undefined
            }
            onShowDetails={() => {
              // Track dataset opened event
              identifyUserFromObject(user);
              captureEvent("dataset_opened", {
                dataset_id: dataset.id,
                dataset_name: dataset.name,
                user_id: hashString(user?.clientId || ""),
                description: "Triggered when user clicks and opens a dataset",
              });
              handleShowDetails(dataset);
            }}
            sourceType={dataset.source_type}
            isEditable={datasetStatuses[dataset.id]}
          />
        ))
      }
      loading={loading}
      error={pageError}
      CreateModal={
        <LargeModal
          isOpen={isModalOpen}
          onClose={() => {
            setIsModalOpen(false);
            setDatasetToEdit(null);
            setIsReadOnlyMode(false);
          }}
          onSubmit={() => {
            if (handleNextClick) {
              handleNextClick();
            }
          }}
          title={
            isReadOnlyMode
              ? "Dataset Details"
              : datasetToEdit
                ? "Edit Dataset"
                : "Create New Dataset"
          }
          actionButton={isReadOnlyMode ? undefined : "Next"}
          type="dataset"
          fullWidth={true}
          fullHeight={true}
        >
          <CreateDatasetForm
            onClose={() => {
              setIsModalOpen(false);
              setDatasetToEdit(null);
              setIsReadOnlyMode(false);
            }}
            workspaceId={workspaceId}
            datasetId={datasetToEdit?.id}
            setDatasetId={() => {}}
            setNewDataAdded={() => {
              fetchDatasets();
            }}
            newDataAdded={false}
            formSubmit={setHandleNextClick}
            isEditMode={!!datasetToEdit}
          />
        </LargeModal>
      }
      DeleteModal={
        <DeleteModal
          isOpen={isDeleteModalOpen}
          onClose={() => setIsDeleteModalOpen(false)}
          onDelete={handleDeleteDataset}
          title="Are you sure you want to delete this dataset?"
        />
      }
      pagination={pagination}
      totalPages={totalPages}
      onPaginationChange={setPagination}
    />
  );
};

export default Datasets;
