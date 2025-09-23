/* The above code is a React component called `FileUploadComponent` that allows users to upload files
to a workspace. Here is a breakdown of what the code is doing: */
import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import Image from "next/image";
import uploadIcon from "@/assets/icons/upload-files.svg";
import AwsForm from "../forms/awsForm";
import Modal from "../forms/modal";
import { useRouter } from "next/navigation";
import { getSourceConnectorById, uploadFile } from "@/api/Workspace/workspace";
import { showError, showSuccess } from "@/utils/toastUtils";
import { PostgresForm } from "@/components/forms/postgresForm";
import GrooveForm from "@/components/forms/grooveForm";
import { constants } from "@/lib/constants";
import {
  ALLOWED_FILE_EXTENSIONS,
  ALLOWED_IMG_EXTENSIONS,
  ALLOWED_AUDIO_EXTENSIONS,
  ALLOWED_VIDEO_EXTENSIONS,
  ALLOWED_MIME_TYPES,
  MAX_FILE_SIZE_UPLOADED_FILES,
  MAX_IMG_SIZE_UPLOADED_FILES,
  MAX_AUDIO_SIZE_UPLOADED_FILES,
  MAX_VIDEO_SIZE_UPLOADED_FILES,
} from "@/lib/file-constants";
import { initFlagsmith } from "@/lib/flagsmith";

export default function FileUploadComponent({ WorkSpaceId, sourceId, oId }) {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [dragging, setDragging] = useState(false);
  const [openModal, setOpenModal] = useState(false);
  const [type, setType] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [sourceConnector, setSourceConnector] = useState(null);
  const router = useRouter();
  const orgId = oId;
  const source = {
    aws: {
      type: constants.SOURCE.AWS,
      title: "Connect to AWS S3",
    },
    azure: {
      type: constants.SOURCE.AZURE,
      title: "Connect to Azure Blob",
    },
    archive: {
      type: "archive",
      title: "Connect to Archive",
    },
    sftp: {
      type: "sftp",
      title: "Connect to SFTP",
    },
    postgres: {
      type: constants.SOURCE.POSTGRES,
      title: "Connect to PostgresSQL DB",
    },
    mysql: {
      type: constants.SOURCE.MYSQL,
      title: "Connect to MySQL DB",
    },
    groove: {
      type: constants.SOURCE.GROOVE,
      title: "Connect to Groove Support",
    },
  };

  useEffect(() => {
    if (sourceId) {
      getSourceConnectorById(WorkSpaceId, sourceId).then((connector) => {
        connector["id"] = sourceId;
        setSourceConnector(connector);
        if (connector.source_type === constants.SOURCE_TYPE.AZURE) {
          setType(constants.SOURCE.AZURE);
        } else if (connector.source_type === constants.SOURCE_TYPE.AWS) {
          setType(constants.SOURCE.AWS);
        } else if (connector.source_type === constants.SOURCE_TYPE.POSTGRES) {
          setType(constants.SOURCE.POSTGRES);
        } else if (connector.source_type === constants.SOURCE_TYPE.MYSQL) {
          setType(constants.SOURCE.MYSQL);
        } else if (connector.source_type === constants.SOURCE_TYPE.GROOVE) {
          setType(constants.SOURCE.GROOVE);
        }
        setOpenModal(true);
      });
    }
  }, [sourceId]);

  const dataSources = [
    {
      title: "Azure Blob",
      type: constants.SOURCE.AZURE,
    },
    {
      title: "AWS S3",
      type: constants.SOURCE.AWS,
    },
    {
      title: "PostgresSQL",
      type: constants.SOURCE.POSTGRES,
    },
    {
      title: "MySQL",
      type: constants.SOURCE.MYSQL,
    },
    {
      title: "Groove Support",
      type: constants.SOURCE.GROOVE,
    },
  ];
  const [availableSources, setAvailableSources] = useState(dataSources);

  useEffect(() => {
    const loadFlags = async () => {
      const flagsmith = await initFlagsmith();

      const isGrooveEnabled = flagsmith
        ? flagsmith.hasFeature(constants.GROOVE_CONNECTOR_FEATURE)
        : false;

      const filtered = dataSources.filter((ds) =>
        ds.type === constants.SOURCE.GROOVE ? isGrooveEnabled : true,
      );

      setAvailableSources(filtered);
    };

    loadFlags();
  }, []);

  const checkIsImage = (fileExtension) => {
    return ALLOWED_IMG_EXTENSIONS.includes(fileExtension);
  };

  const checkIsAudio = (fileExtension) => {
    return ALLOWED_AUDIO_EXTENSIONS.includes(fileExtension);
  };

  const checkIsVideo = (fileExtension) => {
    return ALLOWED_VIDEO_EXTENSIONS.includes(fileExtension);
  };

  const validateFile = (file) => {
    const fileExtension = "." + file.name.split(".").pop().toLowerCase();
    const fileSize = file.size;
    const fileMimeType = file.type;
    // Check file extension
    console.log(file);
    if (!ALLOWED_FILE_EXTENSIONS.includes(fileExtension)) {
      showError("unsupported file");
      return false;
    }

    // Check MIME type
    if (!ALLOWED_MIME_TYPES.includes(fileMimeType) && fileMimeType) {
      showError("unsupported file");
      return false;
    }

    let maxFileSize = MAX_FILE_SIZE_UPLOADED_FILES;
    if (checkIsImage(fileExtension)) {
      maxFileSize = MAX_IMG_SIZE_UPLOADED_FILES;
    } else if (checkIsAudio(fileExtension)) {
      maxFileSize = MAX_AUDIO_SIZE_UPLOADED_FILES;
    } else if (checkIsVideo(fileExtension)) {
      maxFileSize = MAX_VIDEO_SIZE_UPLOADED_FILES;
    }
    // Check file size
    if (fileSize > maxFileSize) {
      showError(
        `${file.name} exceeds maximum file size of ${
          maxFileSize / (1024 * 1024)
        }MB`,
      );
      return false;
    }
    return true;
  };

  // Handle file selection
  const handleFileChange = (event) => {
    const files = Array.from(event.target.files || []);
    const validFiles = files.filter(validateFile);
    setSelectedFiles((prev) => [...prev, ...validFiles]);
  };

  // Trigger file input click when the button is clicked
  const handleButtonClick = (event) => {
    event.preventDefault();
    document.getElementById("fileUpload").click();
  };

  // Handle file removal
  const handleRemoveFile = (event, index) => {
    event.preventDefault();
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  // Handle drag events
  const handleDragOver = (event) => {
    event.preventDefault();
    setDragging(true);
  };

  const handleDragLeave = () => {
    setDragging(false);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    const files = Array.from(event.dataTransfer.files || []);
    const validFiles = files.filter(validateFile);
    setSelectedFiles((prev) => [...prev, ...validFiles]);
  };

  const handleNext = async () => {
    if (isUploading) return; // Prevent multiple clicks
    setIsUploading(true); // Disable button
    try {
      const formData = new FormData();
      selectedFiles.forEach((file) => {
        formData.append("files", file);
      });
      const payload = {
        id: WorkSpaceId,
        body: formData,
      };

      const response = await uploadFile(payload);
      if (response.status === 200) {
        showSuccess("Uploading files this may take a moment.");
        router.push(`/workspace/${WorkSpaceId}/files/0`);
      }
    } catch (error) {
      showError(error.response.data.detail || "File upload failed.");
    } finally {
      setIsUploading(false);
      setSelectedFiles([]); // Clear selected files after upload
    }
  };

  return (
    <>
      <div className="p-6 flex flex-col items-center justify-center space-y-6">
        {/* Upload Card */}
        <Card
          className={`w-full border-gray-200 border-4 rounded ${
            dragging ? "border-blue-500" : ""
          }`}
        >
          <CardContent
            className="p-12 flex flex-col gap-4 items-center justify-center"
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <Image src={uploadIcon} alt="upload icon" />
            <p className="text-gray-700 text-sm font-normal">
              Drag or click here to upload a file
            </p>
            <div className={"flex flex-col justify-center items-center"}>
              <p className="text-gray-800 text-xs ">
                Accepted Image Formats: .jpg, .png, .jpeg (max{" "}
                {MAX_IMG_SIZE_UPLOADED_FILES / (1024 * 1024)} mb per file)
              </p>
              <p className="text-gray-800 text-xs ">
                Accepted Text Formats: .docx, .pptx, .pdf, .html, .xlsx, .csv,
                .md (max {MAX_FILE_SIZE_UPLOADED_FILES / (1024 * 1024)} mb per
                file)
              </p>
              <p className="text-gray-800 text-xs ">
                Accepted Audio Formats: .mp3, .wav, .aac (max{" "}
                {MAX_AUDIO_SIZE_UPLOADED_FILES / (1024 * 1024)} mb per file)
              </p>
            </div>
            {/* File Input */}
            <input
              key={selectedFiles.length}
              multiple
              type="file"
              id="fileUpload"
              className="hidden"
              onChange={(e) => handleFileChange(e)}
              accept={ALLOWED_FILE_EXTENSIONS.join(",")}
            />

            {/* Upload Button */}
            <Button
              className="bg-blue-500 text-white"
              onClick={handleButtonClick}
              disabled={isUploading}
            >
              Upload file
            </Button>

            {/* Display Selected File */}
            {selectedFiles.length > 0 && (
              <>
                <div className="mt-4 w-full">
                  <h4 className="text-gray-800">Selected File:</h4>
                  {selectedFiles?.map((item, idx) => (
                    <div
                      key={idx}
                      className="flex justify-between items-center"
                    >
                      <span className="text-sm">
                        {item.name} ({(item.size / (1024 * 1024)).toFixed(2)}MB)
                      </span>
                      <Button
                        variant="outline"
                        color="red"
                        onClick={(event) => handleRemoveFile(event, idx)}
                        className="ml-2 mb-1"
                        disabled={isUploading}
                      >
                        Remove
                      </Button>
                    </div>
                  ))}
                </div>
                <Button
                  type="button"
                  className="bg-blue-500 text-white"
                  onClick={handleNext}
                  disabled={isUploading || selectedFiles.length === 0}
                >
                  {isUploading ? "Uploading..." : "Next"}
                </Button>
              </>
            )}
          </CardContent>
        </Card>

        {/* Cloud Storage Options */}
        <Card className="flex justify-around space-x-4 w-full px-8 py-6 !mt-2 rounded">
          {availableSources.map((datasource, index) => (
            <div
              key={index}
              className="flex flex-col items-center cursor-pointer"
              onClick={() => {
                setType(datasource.type);
                setOpenModal(true);
              }}
              tabIndex={0}
              role="button"
              onKeyDown={(e) => e.key === "Enter" && setOpenModal(true)}
            >
              <div className="">
                <Image
                  src={`assets/icons/${datasource.type}.svg`}
                  alt={datasource.title}
                  width={50}
                  height={50}
                />
              </div>
              <p className="mt-2 text-sm">{datasource.title}</p>
            </div>
          ))}
        </Card>
      </div>

      {/* Modal Component */}
      <Modal
        isOpen={openModal}
        onClose={() => setOpenModal(false)}
        title={source[type]?.title}
      >
        {type !== constants.SOURCE.POSTGRES &&
          type !== constants.SOURCE.MYSQL &&
          type !== constants.SOURCE.GROOVE && (
            <AwsForm workSpaceId={WorkSpaceId} type={type} sId={sourceId} />
          )}
        {(type === constants.SOURCE.POSTGRES ||
          type === constants.SOURCE.MYSQL) && (
          <PostgresForm
            workSpaceId={WorkSpaceId}
            source={sourceConnector}
            sourceType={
              type === constants.SOURCE.POSTGRES
                ? constants.SOURCE_TYPE.POSTGRES
                : constants.SOURCE_TYPE.MYSQL
            }
          ></PostgresForm>
        )}
        {type === constants.SOURCE.GROOVE && (
          <GrooveForm workSpaceId={WorkSpaceId} type={type} sId={sourceId} />
        )}
      </Modal>
    </>
  );
}
