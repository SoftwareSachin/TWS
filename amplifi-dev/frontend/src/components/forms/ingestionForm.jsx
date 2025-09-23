import React, { useState, useEffect } from "react";
import { Input } from "../ui/input";
import {
  addChunk,
  getStatus,
  chunking_config,
} from "@/api/Workspace/workspace";
import { dismissToast, showError, showSuccess } from "@/utils/toastUtils";
import { useRouter } from "next/navigation";
import { z } from "zod";
import {
  identifyUserFromObject,
  hashString,
  captureEvent,
} from "@/utils/posthogUtils";
import { useUser } from "@/context_api/userContext";

const IngestionFormSchema = z
  .object({
    name: z
      .string()
      .min(1, "Name is required")
      .max(100, "Name must be 100 characters or less"),
    chunkSize: z
      .number()
      .min(100, "Chunk size must be at least 100")
      .max(5000, "Chunk size must be at most 5000"),
    chunkOverlap: z
      .number()
      .min(0, "Chunk overlap must be at least 0")
      .max(500, "Chunk overlap must be at most 500"),
  })
  .refine((data) => data.chunkOverlap < data.chunkSize, {
    message: "Chunk overlap must be less than chunk size",
    path: ["chunkOverlap"],
  });

const IngestionForm = ({
  setIsOpen,
  dataSetId,
  workspaceId,
  datasetId,
  onClose,
  setNewDataAdded,
  newDataAdded,
  isFirstTime = false,
  isEditFlow = false,
  isUpdateFlow = false,
}) => {
  const { user } = useUser();
  const [formData, setFormData] = useState({
    name: "",
    chunkingStrategy: "by_title",
    chunkSize: 2500,
    chunkOverlap: 250,
    embeddingModel: "gpt-3",
  });

  const [loading, setLoading] = useState(false); // Loading state
  const [error, setError] = useState({}); // Error state
  const router = useRouter();

  const isInvalidChunkConfig =
    parseInt(formData.chunkOverlap) >= parseInt(formData.chunkSize);

  const handleChange = (e) => {
    const { name, value } = e.target;

    const parsedValue =
      name === "chunkSize" || name === "chunkOverlap" ? parseInt(value) : value;

    setFormData((prevState) => ({
      ...prevState,
      [name]: parsedValue,
    }));

    setError((prev) => ({ ...prev, [name]: undefined }));
    if (
      (name === "chunkSize" || name === "chunkOverlap") &&
      (name === "chunkOverlap" ? parsedValue : formData.chunkOverlap) <
        (name === "chunkSize" ? parsedValue : formData.chunkSize)
    ) {
      setError((prev) => ({ ...prev, chunkOverlap: undefined }));
    }
  };

  const validateForm = (data) => {
    try {
      IngestionFormSchema.parse(data);
      setError({});
      return true;
    } catch (err) {
      const fieldErrors = {};
      if (err.errors) {
        err.errors.forEach((error) => {
          fieldErrors[error.path[0]] = { message: error.message };
        });
      }
      setError(fieldErrors);
      // showError("Please validate form details.");
      return false;
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateForm(formData)) return;

    setLoading(true);
    const payload = {
      name: formData.name,
      provider: "unstructured_local",
      overlap: parseInt(formData.chunkOverlap),
      strategy: "auto",
      chunking_strategy: formData.chunkingStrategy,
      max_characters: parseInt(formData.chunkSize),
    };

    // Track start ingestion event
    identifyUserFromObject(user);
    const startTime = Date.now();

    captureEvent("ingestion_started", {
      workspace_id_hash: hashString(workspaceId || ""),
      dataset_id_hash: hashString(dataSetId || ""),
      chunk_size: parseInt(formData.chunkSize),
      chunk_overlap: parseInt(formData.chunkOverlap),
      chunking_strategy: formData.chunkingStrategy,
      description: "User starts ingestion process",
    });

    try {
      const chunkingConfigResponse = await chunking_config(payload, dataSetId);
      if (chunkingConfigResponse?.status === 200) {
        let skipSuccessfulFiles = false;

        if (isFirstTime) {
          // First time after adding files - skip successful files
          skipSuccessfulFiles = true;
        } else if (isUpdateFlow) {
          // Coming from "Update" button in existing modal - don't skip successful files
          skipSuccessfulFiles = false;
        } else if (isEditFlow) {
          // Regular edit/retry flow - don't skip successful files
          skipSuccessfulFiles = false;
        }

        const response = await addChunk(dataSetId, skipSuccessfulFiles);
        if (response?.status === 200) {
          const duration_ms = Date.now() - startTime;

          // Track successful ingestion initiation
          captureEvent("file_ingested", {
            workspace_id_hash: hashString(workspaceId || ""),
            dataset_id_hash: hashString(dataSetId || ""),
            status: "initiated",
            duration_ms: duration_ms,
            retry_count: 0,
            description: "Ingestion completes via backend",
          });

          showSuccess("Ingestion initiated successfully!");
          setNewDataAdded(!newDataAdded);
          setIsOpen(false);
          onClose();
          router.push(`/workspace/${workspaceId}/datasets/${dataSetId}`);
        }
      }
    } catch (err) {
      console.error("API Error:", err);
      const duration_ms = Date.now() - startTime;

      // Track failed ingestion
      captureEvent("file_ingested", {
        workspace_id_hash: hashString(workspaceId || ""),
        dataset_id_hash: hashString(dataSetId || ""),
        status: "failed",
        duration_ms: duration_ms,
        retry_count: 0,
        description: "Ingestion failed during backend processing",
      });

      setError({
        form: {
          message: "Failed to add chunking configuration. Please try again.",
        },
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <form onSubmit={handleSubmit}>
        <div className="p-4 pt-0 space-y-4">
          <div>
            <label
              htmlFor="name"
              className="block mb-2 text-sm font-medium text-gray-900"
            >
              Name
            </label>
            <Input
              id="name"
              name="name"
              value={formData.name}
              onChange={handleChange}
              placeholder="Enter configuration name"
              required
              className="w-full"
            />
            {error.name && (
              <p className="text-red-500 text-sm">{error.name.message}</p>
            )}
          </div>

          {/* <div>
            <label
              htmlFor="chunkingStrategy"
              className="block mb-2 text-sm font-medium text-gray-900"
            >
              File Splitting Strategy{" "}
              <span className="text-gray-400 text-xs">
                {"(We recommend 'By Title' for most PDFs)"}
              </span>
            </label>
            <select
              id="chunkingStrategy"
              name="chunkingStrategy"
              className="w-full p-2 border border-gray-300 rounded-md focus:outline-none"
              value={formData.chunkingStrategy}
              onChange={handleChange}
              required
            >
              <option value="by_title">By Title</option>
              <option value="basic">Basic</option>
            </select>
          </div> */}

          <div>
            <label
              htmlFor="chunkSize"
              className="block mb-2 text-sm font-medium text-gray-900"
            >
              Chunk Size{" "}
              <span className="text-gray-400 text-xs">
                {"(Default setting of 2500 works best)"}
              </span>
            </label>
            <div className="flex items-center space-x-2">
              <div className="p-2 border rounded-md w-20">
                {formData.chunkSize}
              </div>
              <Input
                type="range"
                id="chunkSize"
                name="chunkSize"
                value={formData.chunkSize}
                onChange={handleChange}
                min="100"
                max="5000"
                step="100"
                required
                className="w-full !shadow-none"
              />
            </div>
            {error.chunkSize && (
              <p className="text-red-500 text-sm">{error.chunkSize.message}</p>
            )}
          </div>

          <div>
            <label
              htmlFor="chunkOverlap"
              className="block mb-2 text-sm font-medium text-gray-900"
            >
              Chunk Overlap{" "}
              <span className="text-gray-400 text-xs">
                {"(Default setting of 250 works best)"}
              </span>
            </label>
            <div className="flex items-center space-x-2">
              <div className="p-2 border rounded-md w-20">
                {formData.chunkOverlap}
              </div>
              <Input
                type="range"
                id="chunkOverlap"
                name="chunkOverlap"
                value={formData.chunkOverlap}
                onChange={handleChange}
                min="0"
                max="500"
                step="10"
                required
                className="w-full !shadow-none"
              />
            </div>
            {error.chunkOverlap && (
              <p className="text-red-500 text-sm">
                {error.chunkOverlap.message}
              </p>
            )}
          </div>

          {/* <div>
            <label
              htmlFor="embeddingModel"
              className="block mb-2 text-sm font-medium text-gray-900"
            >
              Embedding Model{" "}
              <span className="text-gray-400 text-xs">
                {"(OpenAI Embedding is recommended for faster processing)"}
              </span>
            </label>
            <select
              id="embeddingModel"
              name="embeddingModel"
              className="w-full p-2 border border-gray-300 rounded-md focus:outline-none"
              value={formData.embeddingModel}
              onChange={handleChange}
              required
            >
              <option value="unstructured_local">Local Embedding Mode</option>
              <option value="gpt-3">OpenAI Text Embedding 3 Large</option>
            </select>
          </div> */}
        </div>
        {error.form && (
          <p className="text-red-500 text-sm px-4">{error.form.message}</p>
        )}

        <div className="border-t flex justify-end gap-4 p-4">
          <button
            type="button"
            className="border px-4 py-2 rounded text-sm"
            onClick={() => setIsOpen(false)}
          >
            Cancel
          </button>
          <button
            type="submit"
            className="bg-blue-500 text-white px-4 py-2 rounded text-sm"
            disabled={loading}
          >
            {loading ? "Submitting..." : "Start Ingestion"}
          </button>
        </div>
      </form>
    </>
  );
};

export default IngestionForm;
