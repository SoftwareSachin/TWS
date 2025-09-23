"use client";
import React, { useEffect, useState } from "react";
import { Input } from "@/components/ui/input";
import {
  RadioGroup,
  RadioGroupItem,
} from "@/design_components/radio/radio-group";
import { Textarea } from "../ui/textarea";
import { ChevronRight } from "lucide-react";
import DrawerVertical from "./drawervertical";
import DatasetSelectFileForm from "./datasetSelectFileForm";
import Select from "react-select"; // Import react-select
import { showError, showSuccess } from "@/utils/toastUtils";
import { createDataSet, getDataSetById, updateDataSet } from "@/api/dataset";
import IngestionForm from "./ingestionForm";
import Modal from "./modal";
import { fetchDataSources } from "@/api/Workspace/workspace";
import { constants } from "@/lib/constants";
import { SqlTrainingForm } from "@/components/forms/sqlTrainingForm";
import { validateForm } from "@/lib/form-validation";
import { DatasetSchema } from "@/form_schemas/dataset";
import { addChunk } from "@/api/Workspace/workspace";
import { useRouter } from "next/navigation";
import { Button } from "../ui/button";
import { set } from "date-fns";
import {
  identifyUserFromObject,
  hashString,
  captureEvent,
} from "@/utils/posthogUtils";
import { useUser } from "@/context_api/userContext";

const CreateDatasetForm = ({
  onClose,
  workspaceId,
  setNewDataAdded,
  newDataAdded,
  datasetId,
  setDatasetId,
  formSubmit,
  isEditMode,
  isAddMoreFilesMode = false,
}) => {
  const { user } = useUser();
  const [selectModal, setSelectModal] = useState(false);
  const [dataSetId, setDataSetId] = useState(datasetId);
  const [submitted, setSubmitted] = useState(false);
  const [formData, setFormData] = useState({
    name: "",
    description: "",
    fileOption: "selectFile", // default option
    selectedFile: [], // for select input
    sourceId: "",
  });
  const [configureModal, setConfigureModal] = useState(false);
  const [dataSourceOptions, setDataSourceOptions] = useState([]);
  const [isSqlDatasource, setIsSqlDatasource] = useState(false);
  const [formErrors, setFormErrors] = useState({});
  const [nameLength, setNameLength] = useState(0);
  const [showChunkConfirmModal, setShowChunkConfirmModal] = useState(false);
  const [loading, setLoading] = useState(false);
  const NAME_MAX_LENGTH = 25;
  const router = useRouter();

  useEffect(() => {
    if (submitted) {
      validateFormDetails();
    }
  }, [formData, submitted]);

  // Update name length when form data changes
  useEffect(() => {
    setNameLength(formData.name.length);
  }, [formData.name]);
  useEffect(() => {
    if (formSubmit) {
      formSubmit(() => handleSubmit);
    }
  }, [formSubmit, formData]);

  useEffect(() => {
    if (datasetId) {
      handleEditDataSetForm();
    }
  }, [datasetId, isAddMoreFilesMode]);

  useEffect(() => {
    if (formData.sourceId && dataSourceOptions.length) {
      const selectedSource = dataSourceOptions
        .flatMap((group) => group.options)
        .find((option) => option.value === formData.sourceId);

      if (
        selectedSource?.type === constants.SOURCE_TYPE_LABEL.POSTGRES ||
        selectedSource?.type === constants.SOURCE_TYPE_LABEL.MYSQL
      ) {
        setIsSqlDatasource(true);
      } else {
        setIsSqlDatasource(false);
      }
    }
  }, [formData.sourceId, dataSourceOptions]);

  const handleEditDataSetForm = async () => {
    const datasetData = await getDataSetById(workspaceId, datasetId);
    setFormData({
      name: datasetData.name,
      description: datasetData.description,
      fileOption: isAddMoreFilesMode
        ? "selectFile"
        : datasetData.source_id
          ? "ingestAll"
          : "selectFile",
      selectedFile: isAddMoreFilesMode
        ? datasetData.file_ids || [] // Show existing files as pre-selected for add more files mode
        : datasetData.source_id
          ? []
          : datasetData.file_ids,
      sourceId: isAddMoreFilesMode ? "" : datasetData.source_id,
    });
  };

  const handleSetSelectedFile = (selectedFile) => {
    setFormData((prev) => ({
      ...prev,
      selectedFile,
    }));
    setSelectModal(false); // Close modal after selection

    // Track file upload initiated event
    if (selectedFile && selectedFile.length > 0) {
      identifyUserFromObject(user);
      captureEvent("file_upload_initiated", {
        workspace_id_hash: hashString(workspaceId || ""),
        file_count: selectedFile.length,
        dataset_id_hash: hashString(datasetId || dataSetId || ""),
        description: "User initiates file upload for dataset",
      });
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
    if (name === "name") {
      setNameLength(value.length);
    }
  };

  const handleRadioChange = (value) => {
    const updatedForm = {
      ...formData,
      fileOption: value,
    };

    if (value === "ingestAll") {
      updatedForm.selectedFile = [];
    }

    if (value === "selectFile") {
      updatedForm.sourceId = "";
      setIsSqlDatasource(false);
    }
    setFormData(updatedForm);
  };

  const handleSelectChange = async (selectedOption) => {
    setFormData({
      ...formData,
      sourceId: selectedOption.value,
    });
  };

  const validateFormDetails = () => {
    const formValidationResponse = validateForm(DatasetSchema, formData);
    setFormErrors({});
    if (!formValidationResponse.success) {
      setFormErrors(formValidationResponse.errors);
      return false;
    }
    return true;
  };

  const handleSubmit = async () => {
    setSubmitted(true);
    if (validateFormDetails()) {
      try {
        const payload = {
          name: formData.name,
          description: formData.description, //  ...(formData.selectedFile && {file_ids:formData.selectedFile} )
        };
        if (formData.fileOption === "ingestAll") {
          payload["source_id"] = formData.sourceId;
        } else {
          payload["file_ids"] = formData.selectedFile;
        }
        let response;
        if (datasetId) {
          response = await updateDataSet(workspaceId, datasetId, payload);
        } else {
          response = await createDataSet(workspaceId, payload);
        }

        if (response.status === 200) {
          if (isAddMoreFilesMode) {
            showSuccess("Files added to dataset successfully!");
          } else {
            showSuccess(`${response.data.message}`);
          }
          setDatasetId(response.data.data.id);
          setDataSetId(response.data.data.id);
        }

        if (isEditMode && formData.fileOption === "selectFile") {
          // For both edit mode and add more files mode, show chunk confirmation
          setShowChunkConfirmModal(true);
        } else {
          setConfigureModal(true);
        }

        if (isEditMode) {
          setNewDataAdded(!newDataAdded);
        }
      } catch (error) {
        showError(`${error.response.data.detail}`);
      }
    }
  };

  const handleChunkValueUpdate = async () => {
    try {
      setLoading(true);
      const res = await addChunk(dataSetId, true);
      if (res.status === 200) {
        const successMessage = isAddMoreFilesMode
          ? "New files ingestion started with previous config!"
          : "Ingestion re‑started with previous config!";
        showSuccess(successMessage);
        setNewDataAdded(!newDataAdded);
        onClose();
        router.push(`/workspace/${workspaceId}/datasets/${dataSetId}`);
      } else {
        showError("Failed to reuse old config.");
        setLoading(false);
      }
    } catch (err) {
      showError(err?.message || "Something went wrong");
      setLoading(false);
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => {
    // Fetch data source options from API
    const getSourceTypeLabel = (sourceType) => {
      switch (sourceType) {
        case constants.SOURCE_TYPE.AZURE:
          return constants.SOURCE_TYPE_LABEL.AZURE;
        case constants.SOURCE_TYPE.AWS:
          return constants.SOURCE_TYPE_LABEL.AWS;
        case constants.SOURCE_TYPE.POSTGRES:
          return constants.SOURCE_TYPE_LABEL.POSTGRES;
        case constants.SOURCE_TYPE.MYSQL:
          return constants.SOURCE_TYPE_LABEL.MYSQL;
        case constants.SOURCE_TYPE.GROOVE:
          return constants.SOURCE_TYPE_LABEL.GROOVE;
        default:
          return;
      }
    };
    const fetchData = async () => {
      try {
        const response = await fetchDataSources(workspaceId);
        const options = response.data.data.items.map((item) => {
          let label;

          // Handle different source types for label generation
          if (
            item.sources.source_type === constants.SOURCE_TYPE.POSTGRES ||
            item.sources.source_type === constants.SOURCE_TYPE.MYSQL
          ) {
            label = `${item.sources.host}/${item.sources.database_name}`;
          } else if (
            item.sources.source_type === constants.SOURCE_TYPE.GROOVE
          ) {
            // Handle the new groove_source type
            label = item.sources.source_name;
          } else {
            // Default case for other source types
            label = item.sources.container_name;
          }

          return {
            label: label,
            value: item.sources.source_id,
            type: getSourceTypeLabel(item.sources.source_type),
          };
        });
        const dataOption = Object.values(
          options.reduce((acc, item) => {
            acc[item.type] = acc[item.type] || {
              label: item.type,
              options: [],
            };
            acc[item.type].options.push(item);
            return acc;
          }, {}),
        );
        setDataSourceOptions(dataOption);
      } catch (error) {
        showError("Failed to load data sources");
      }
    };

    fetchData();
  }, []);

  return (
    <>
      <form className="space-y-6 p-4 bg-white rounded h-full flex flex-col items-center">
        <div className="flex flex-col items-center w-full">
          <div className="w-2/5">
            <label
              htmlFor="name"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Name
            </label>
            <Input
              type="text"
              id="name"
              name="name"
              value={formData.name}
              onChange={handleChange}
              placeholder="Enter dataset name"
              maxLength={NAME_MAX_LENGTH}
              required
            />
            <div className="text-xs text-gray-500 mt-1">
              {nameLength}/{NAME_MAX_LENGTH} characters
            </div>
            {formErrors?.name && (
              <div className="text-red-600 text-sm">{formErrors.name}</div>
            )}
          </div>
        </div>

        <div className="flex flex-col items-center w-full">
          <div className="w-2/5">
            <label
              htmlFor="description"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Description
            </label>
            <Textarea
              id="description"
              name="description"
              value={formData.description}
              onChange={handleChange}
              placeholder="Enter dataset description"
              rows="3"
            />
          </div>
        </div>

        <div className="flex flex-col items-center w-full">
          <div className="w-2/5">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              File Option
            </label>
            <RadioGroup
              value={formData.fileOption}
              onValueChange={handleRadioChange}
              className="mt-2 space-y-2"
            >
              <div
                className={`${
                  formData.fileOption === "selectFile" ? "bg-gray-100" : ""
                } p-3 space-y-2 rounded-lg`}
              >
                <div className="flex items-center space-x-2">
                  <RadioGroupItem
                    value="selectFile"
                    id="selectFile"
                    className={`${
                      formData.fileOption === "selectFile"
                        ? "bg-blue-10 border-0"
                        : ""
                    } text-white`}
                  />
                  <label htmlFor="selectFile" className="text-sm">
                    Select file
                  </label>
                </div>

                {formData.fileOption === "selectFile" && (
                  <>
                    <div
                      className="border rounded-md px-3 py-2 text-sm bg-white flex justify-between items-center text-gray-400"
                      onClick={() => {
                        setSelectModal(true);
                      }}
                    >
                      {formData?.selectedFile?.length
                        ? "File selected"
                        : "Select files from options"}{" "}
                      <ChevronRight className="text-sm w-4 h-4" />
                    </div>
                    {formErrors?.selectedFile && (
                      <div className="text-red-600 text-sm">
                        {formErrors.selectedFile}
                      </div>
                    )}
                  </>
                )}
              </div>

              <div
                className={`${
                  formData.fileOption === "ingestAll" ? "bg-gray-100" : ""
                } p-3 space-y-2 rounded-lg`}
              >
                <div className="flex items-center space-x-2">
                  <RadioGroupItem
                    value="ingestAll"
                    id="ingestAll"
                    className={`${
                      formData.fileOption === "ingestAll"
                        ? "bg-blue-10 border-0"
                        : ""
                    } text-white`}
                  />
                  <label htmlFor="ingestAll" className="text-sm">
                    Ingest all new files from source
                  </label>
                </div>

                {formData.fileOption === "ingestAll" && (
                  <div className="flex flex-col">
                    <label htmlFor="selectFile" className="text-sm font-medium">
                      Data source
                    </label>

                    <Select
                      id="selectFile"
                      name="selectFile"
                      value={dataSourceOptions
                        .flatMap((group) => group.options)
                        .find((option) => option.value === formData.sourceId)}
                      onChange={handleSelectChange}
                      options={dataSourceOptions}
                      className="mt-2 rounded focus:border-none text-sm"
                      placeholder="Select a source"
                    />
                    {formErrors?.sourceId && (
                      <div className="text-red-600 text-sm">
                        {formErrors.sourceId}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </RadioGroup>
          </div>
        </div>

        <DrawerVertical
          width={"500px"}
          isOpen={selectModal}
          onClose={() => {
            setSelectModal(false);
          }}
          title="Select Files"
        >
          <DatasetSelectFileForm
            setIsOpen={setSelectModal}
            workspaceId={workspaceId}
            setSelectedFile={handleSetSelectedFile}
            selectedFile={formData?.selectedFile}
          />
        </DrawerVertical>

        {/*<div className={'absolute bottom-0 flex justify-end border-t p-4 w-full gap-4 text-sm '}>*/}
        {/*  <button type="button" onClick={onClose} className="bg-white border font-medium py-2 px-4 rounded-md">*/}
        {/*    Cancel*/}
        {/*  </button>*/}
        {/*  <button type="submit" className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md">*/}
        {/*    Next*/}
        {/*  </button>*/}
        {/*</div>*/}
      </form>

      <Modal
        isOpen={configureModal}
        onClose={() => setConfigureModal(false)}
        title={`${
          isSqlDatasource ? "Configure Training" : "Configure File Ingestion"
        }`}
        size="false"
      >
        {!isSqlDatasource && (
          <IngestionForm
            setIsOpen={() => setConfigureModal(false)}
            dataSetId={dataSetId}
            workspaceId={workspaceId}
            onClose={onClose}
            setNewDataAdded={setNewDataAdded}
            newDataAdded={newDataAdded}
            isFirstTime={!isEditMode && !isAddMoreFilesMode}
            isEditFlow={false}
            isUpdateFlow={isEditMode || isAddMoreFilesMode}
          />
        )}
        {isSqlDatasource && (
          <SqlTrainingForm
            setIsOpen={() => setConfigureModal(false)}
            dataSetId={dataSetId}
            workspaceId={workspaceId}
            onClose={onClose}
            setNewDataAdded={setNewDataAdded}
            newDataAdded={newDataAdded}
          />
        )}
      </Modal>
      {showChunkConfirmModal && (
        <Modal
          isOpen={showChunkConfirmModal}
          onClose={() => setShowChunkConfirmModal(false)}
          title="Update Chunking Configuration for New Files"
          size="false"
        >
          <div className="space-y-4 p-4">
            <p>
              {isAddMoreFilesMode
                ? "Update chunking Configuration now? Skipping will reuse the previous setup."
                : "Update chunking Configuration?"}
            </p>
            <div className="flex justify-end gap-4">
              <Button
                variant={"primary"}
                isLoading={loading}
                className="bg-blue-500 text-white px-4 py-2 rounded text-sm hover:bg-blue-600"
                onClick={() => {
                  setShowChunkConfirmModal(false);
                  setConfigureModal(true);
                }}
              >
                {"Update"}
              </Button>
              <Button
                variant={"outline"}
                isLoading={loading}
                // className="border px-4 py-2 rounded text-sm"
                onClick={async () => {
                  setShowChunkConfirmModal(false);
                  handleChunkValueUpdate();
                }}
              >
                {"Skip"}
              </Button>
            </div>
          </div>
        </Modal>
      )}
    </>
  );
};

export default CreateDatasetForm;
