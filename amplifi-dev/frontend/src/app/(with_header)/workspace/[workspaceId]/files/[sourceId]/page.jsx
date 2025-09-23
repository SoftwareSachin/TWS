/* The above code is a React component named `Files` that displays a table of files based on the source
selected. It fetches file data from an API using `getFiles` and `getSourceConnectorDetails`
functions. The component uses state variables to store file and source data, as well as a loader
state to indicate when data is being fetched. */

"use client";
import React, { useState, useEffect } from "react";
import Image from "next/image"; // Import Image component from Next.js
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useParams } from "next/navigation";
import tickCircle from "@/assets/icons/tick-circle.svg";
import alert from "@/assets/icons/alert-triangle.svg";
import processing from "@/assets/icons/processing.svg";
import alertUploading from "@/assets/icons/alert-octagon.svg";
import downloadIcon from "@/assets/icons/download.svg";
import trashIcon from "@/assets/icons/trash-icon.svg";
import { getFiles } from "@/api/common";
import { downloadFileApi, deleteFileApi } from "@/api/chatApp";
import {
  getSourceConnectorDetails,
  getSourceDetails,
} from "@/api/Workspace/WorkSpaceFiles";
import { showError, showInfo, showSuccess } from "@/utils/toastUtils";
import DeleteModal from "@/components/forms/deleteModal";
import Paginator from "@/components/utility/paginator";

import { useUpdate } from "@/context_api/updateContext";
import frame from "@/assets/icons/Frame.svg";

import { RefreshCw, MoreHorizontal, Pencil, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import SearchBox from "@/design_components/utility/search-box";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import arrowIcon from "@/assets/icons/arrows.svg";
import { getNextSortState, getSortedData } from "@/components/utility/sorting";
import { constants, SortDirection } from "@/lib/constants";

const Files = () => {
  const [fileList, setFileList] = useState([]);
  const [sourceList, setSourceList] = useState([]);
  const [loader, setLoader] = useState(false);
  const [sourceType, setSourceType] = useState("");
  const [isDeleteModalOpen, setDeleteModalOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [pagination, setPagination] = useState({ page: 1, size: 25 });
  const [totalPages, setTotalPages] = useState(1);
  const [sortField, setSortField] = useState(null);
  const [sortDirection, setSortDirection] = useState(null);
  const [searchText, setSearchText] = useState("");
  const { triggerUpdate } = useUpdate();

  const params = useParams();
  const id = params?.sourceId;

  const getFileList = async () => {
    setLoader(true);
    try {
      const response = await getFiles(
        params?.workspaceId || "",
        true,
        {
          page: pagination.page,
          size: pagination.size,
        },
        searchText || "",
      );
      if (response.status === 200) {
        const files = response.data.data.items;
        setFileList(files);
        setLoader(false);
        const totalFiles = response.data.data.total;
        const totalPages = Math.ceil(totalFiles / pagination.size);
        setTotalPages(totalPages);
      }
    } catch (e) {
      setLoader(false);
    }
  };
  const handleSort = (field) => {
    const { nextField, nextDirection } = getNextSortState(
      field,
      sortField,
      sortDirection,
    );
    setSortField(nextField);
    setSortDirection(nextDirection);
  };

  // Get sorted data based on current sort state
  const getSortedFileList = () => {
    if (!sortField || !sortDirection) {
      // Return original data if no sorting is applied
      return id == 0 ? fileList : sourceList;
    }

    if (id == 0) {
      return getSortedData(fileList, sortField, sortDirection);
    } else {
      return getSortedData(sourceList, sortField, sortDirection);
    }
  };

  const getSourceDetailsList = async () => {
    setLoader(true);
    const data = {
      workspaceId: params?.workspaceId,
      sourceId: id,
      page: 1,
      size: 50,
      searchText: searchText || "",
    };
    try {
      const source_name = await getSourceDetails(data);
      const response = await getSourceConnectorDetails(data, pagination);

      let source_type = source_name?.data?.data?.source_type;
      if (source_type === "azure_storage") {
        source_type = "Azure Blob";
      } else if (source_type === "pg_db") {
        source_type = "PostgresSQL";
      } else {
        source_type = "AWS";
      }
      setSourceType(source_type);
      if (response.status === 200) {
        setLoader(false);
        setSourceList(response.data.data.items);
        setTotalPages(response.data.data.total || 1);
      }
    } catch (e) {
      setLoader(false);
      showError(`${e?.response?.data?.detail}`);
    }
  };

  // Fetch data when id or pagination changes
  useEffect(() => {
    const fetchData = () => {
      if (id == 0) {
        getFileList();
      } else {
        getSourceDetailsList();
      }
    };

    fetchData();
  }, [id, pagination, searchText]);

  useEffect(() => {}, [sortField, sortDirection]);

  const data = {
    1: {
      source: "Mortgage Promotion text header one two lines",
      tableData: [
        {
          fileName: "Dec-2023-Mortgage-promotion.pdf",
          type: "PDF",
          size: "120MB",
          status: "Uploaded",
        },
        {
          fileName: "Dec-2023-Mortgage-promotion.pdf",
          type: "PDF",
          size: "120MB",
          status: "Uploaded",
        },
        {
          fileName: "Jan-2024-Mortgage-promotion.pdf",
          type: "PDF",
          size: "20.3MB",
          status: "Failed",
        },
        {
          fileName: "Dec-2023-Mortgage-promotion.pdf",
          type: "PDF",
          size: "12.9MB",
          status: "Stopped",
        },
        {
          fileName: "Dec-2023-Mortgage-promotion.pdf",
          type: "PDF",
          size: "12.39MB",
          status: "Uploading...",
        },
      ],
    },
    2: {
      source: "Mortgage Promotion text header one",
      tableData: [
        {
          fileName: "Dec-2023-Mortgage-promotion.pdf",
          type: "PDF",
          size: "120MB",
          status: "Uploaded",
        },
        {
          fileName: "Dec-2023-Mortgage-promotion.pdf",
          type: "PDF",
          size: "120MB",
          status: "Uploaded",
        },
        {
          fileName: "Jan-2024-Mortgage-promotion.pdf",
          type: "PDF",
          size: "20.3MB",
          status: "Failed",
        },
        {
          fileName: "Dec-2023-Mortgage-promotion.pdf",
          type: "PDF",
          size: "12.9MB",
          status: "Stopped",
        },
        {
          fileName: "Dec-2023-Mortgage-promotion.pdf",
          type: "PDF",
          size: "12.39MB",
          status: "Uploading...",
        },
      ],
    },
    3: {
      source: "Credit card promotions text header",
      tableData: [
        {
          fileName: "Dec-2023-Mortgage-promotion.pdf",
          type: "PDF",
          size: "120MB",
          status: "Uploaded",
        },
        {
          fileName: "Dec-2023-Mortgage-promotion.pdf",
          type: "PDF",
          size: "120MB",
          status: "Uploaded",
        },
        {
          fileName: "Jan-2024-Mortgage-promotion.pdf",
          type: "PDF",
          size: "20.3MB",
          status: "Failed",
        },
        {
          fileName: "Dec-2023-Mortgage-promotion.pdf",
          type: "PDF",
          size: "12.9MB",
          status: "Stopped",
        },
        {
          fileName: "Dec-2023-Mortgage-promotion.pdf",
          type: "PDF",
          size: "12.39MB",
          status: "Uploading...",
        },
      ],
    },
    4: {
      source: "Mortgage Promotion text header one two lines",
      tableData: [
        {
          fileName: "Dec-2023-Mortgage-promotion.pdf",
          type: "PDF",
          size: "120MB",
          status: "Uploaded",
        },
        {
          fileName: "Dec-2023-Mortgage-promotion.pdf",
          type: "PDF",
          size: "120MB",
          status: "Uploaded",
        },
        {
          fileName: "Jan-2024-Mortgage-promotion.pdf",
          type: "PDF",
          size: "20.3MB",
          status: "Failed",
        },
        {
          fileName: "Dec-2023-Mortgage-promotion.pdf",
          type: "PDF",
          size: "12.9MB",
          status: "Stopped",
        },
        {
          fileName: "Dec-2023-Mortgage-promotion.pdf",
          type: "PDF",
          size: "12.39MB",
          status: "Uploading...",
        },
      ],
    },
  };

  const statusTextColor = {
    Uploaded: "bg-green-100 text-green-800",
    Failed: "bg-red-100 text-red-800",
    Processing: "bg-blue-100 text-blue-800",
    Stopped: "bg-yellow-100 text-yellow-800",
  };

  const getIconColor = (status) => {
    switch (status) {
      case "Uploaded":
        return (
          <Image
            src={tickCircle}
            alt="upload icon"
            className="inline"
            width={16}
            height={16}
          />
        );
      case "Failed":
        return (
          <Image
            src={alert}
            alt="alert icon"
            className="inline"
            width={16}
            height={16}
          />
        );
      case "Processing":
        return (
          <Image
            src={processing}
            alt="processing icon"
            className="inline"
            width={16}
            height={16}
          />
        );
      case "Stopped":
        return (
          <Image
            src={alertUploading}
            alt="stopped icon"
            className="inline"
            width={16}
            height={16}
          />
        );
      default:
        return null;
    }
  };

  const handleDownload = async (item) => {
    const file = {
      file_id: item.id,
      file_name: item.filename,
    };
    try {
      await downloadFileApi(file);
    } catch (error) {
      showError("Failed to download file");
      console.error("Download error:", error);
    }
  };

  const handleFileDelete = async () => {
    if (!selectedFile) return;

    const file = {
      file_id: selectedFile.id,
      filename: selectedFile.filename,
      workspaceId: params?.workspaceId,
    };
    try {
      const response = await deleteFileApi(file);
      if (response.status === 204) {
        showSuccess("File deleted successfully");

        if (id === "0") {
          setFileList((prev) => prev.filter((f) => f.id !== selectedFile.id));
          await getFileList();
          triggerUpdate();
        }
      } else {
        showError("Failed to delete file");
      }
    } catch (error) {
      showError("Failed to delete file");
      console.error("Delete error:", error);
    }
  };

  return (
    <div className="p-8">
      <div className="text-base flex justify-between mb-4">
        <div className="text-sm">
          <span className="font-normal">Connected to</span>{" "}
          <span className="font-semibold">
            {id == 0 ? "Media" : sourceType}
          </span>{" "}
        </div>
        <div className="text-sm">
          <div className="flex gap-3">
            <SearchBox
              value={searchText || ""}
              onDebouncedChange={(e) => {
                setSearchText(e || "");
              }}
              placeholder="Search Files"
            ></SearchBox>

            <Button
              variant="outline"
              className="flex items-center space-x-2 border-gray-300 text-gray-700 hover:bg-gray-50"
              onClick={() => {
                id == 0 ? getFileList() : getSourceDetailsList();
              }}
            >
              <RefreshCw className="w-4 h-4" />
              <span>Refresh Status</span>
            </Button>
          </div>
        </div>
      </div>
      <Table className="border border-gray-300 rounded-2xl mh-[70vh]">
        <TableHeader>
          <TableRow className="border-b-2 bg-custom-tableHeader">
            <TableHead className="text-xs font-semibold bg-custom-tableHeader text-black-10 ps-4"></TableHead>
            <TableHead className="text-xs font-semibold bg-custom-tableHeader text-black-10 ps-4">
              <span>File Name</span>
              <Button
                variant="dataset"
                onClick={() => handleSort("filename")}
                className={`ml-1 p-1 rounded`}
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
            </TableHead>
            <TableHead className="text-xs font-semibold bg-custom-tableHeader text-black-10 ps-4">
              <span>Type</span>
              <Button
                variant="dataset"
                onClick={() => handleSort("mimetype")}
                className={`ml-1 p-1 rounded`}
              >
                <Image
                  src={arrowIcon}
                  alt="Arrow"
                  width={14}
                  height={14}
                  className={`transition-transform ${
                    sortField === "mimetype" &&
                    sortDirection === constants.SORTING.ASCENDING
                      ? "rotate-180"
                      : ""
                  }`}
                />
              </Button>
            </TableHead>
            <TableHead className="text-xs font-semibold bg-custom-tableHeader text-black-10 ps-4">
              <span>Size</span>
              <Button
                variant="dataset"
                onClick={() => handleSort("size")}
                className={`ml-1 p-1 rounded`}
              >
                <Image
                  src={arrowIcon}
                  alt="Arrow"
                  width={14}
                  height={14}
                  className={`transition-transform ${
                    sortField === "size" &&
                    sortDirection === constants.SORTING.ASCENDING
                      ? "rotate-180"
                      : ""
                  }`}
                />
              </Button>
            </TableHead>
            <TableHead className="text-xs font-semibold bg-custom-tableHeader text-black-10 ps-4">
              <span>Status</span>
              <Button
                variant="dataset"
                onClick={() => handleSort("status")}
                className={`ml-1 p-1 rounded`}
              >
                <Image
                  src={arrowIcon}
                  alt="Arrow"
                  width={14}
                  height={14}
                  className={`transition-transform ${
                    sortField === "status" &&
                    sortDirection === constants.SORTING.ASCENDING
                      ? "rotate-180"
                      : ""
                  }`}
                />
              </Button>
            </TableHead>
            <TableHead className="text-xs text-center font-semibold bg-custom-tableHeader text-black-10 ps-4">
              Actions
            </TableHead>
          </TableRow>
        </TableHeader>
        {loader ? (
          <>
            {Array.from({ length: 6 }).map((_, index) => (
              <TableRow
                key={index}
                className="border-b-2 bg-custom-tableHeader bg-white animate-pulse"
              >
                {/* Skeleton for Index */}
                <TableCell className="py-3 px-4">
                  <div className="h-4 w-8 bg-gray-300 rounded-md"></div>
                </TableCell>
                {/* Skeleton for Filename */}
                <TableCell className="py-3 px-4">
                  <div className="h-4 w-2/3 bg-gray-300 rounded-md"></div>
                </TableCell>

                {/* Skeleton for Mimetype */}
                <TableCell className="py-3 px-4">
                  <div className="h-4 w-1/2 bg-gray-300 rounded-md"></div>
                </TableCell>

                {/* Skeleton for Size */}
                <TableCell className="py-3 px-4">
                  <div className="h-4 w-1/4 bg-gray-300 rounded-md"></div>
                </TableCell>

                {/* Skeleton for Status */}
                <TableCell className="py-3 px-4 text-xs">
                  <div className="flex items-center gap-2">
                    <div className="h-4 w-4 bg-gray-300 rounded-full"></div>{" "}
                    {/* Icon placeholder */}
                    <div className="h-4 w-1/3 bg-gray-300 rounded-md"></div>{" "}
                    {/* Status text */}
                  </div>
                </TableCell>

                {/* Skeleton for Actions */}
                <TableCell className="py-3 px-4">
                  <div className="h-4 w-1/4 bg-gray-300 rounded-md"></div>
                </TableCell>
              </TableRow>
            ))}
          </>
        ) : id == 0 ? (
          <TableBody>
            {getSortedFileList() && getSortedFileList().length > 0 ? (
              getSortedFileList().map((item, index) => (
                <TableRow
                  className="border-b-2 border-gray-300 bg-white"
                  key={item.id}
                >
                  <TableCell className="py-3 px-4 text-center border-r border-gray-300">
                    {index + 1}
                  </TableCell>
                  <TableCell className="py-3 px-4">
                    <span>{item?.filename}</span>
                  </TableCell>
                  <TableCell className="py-3 px-4">{item?.mimetype}</TableCell>
                  <TableCell className="py-3 px-4">
                    {(item?.size / (1024 * 1024)).toFixed(2)} Mb
                  </TableCell>
                  <TableCell className="py-3 px-4 text-xs">
                    <span
                      className={`${
                        statusTextColor[item?.status] ||
                        "bg-gray-600 text-gray-800"
                      } rounded p-1 flex justify-start items-center gap-1 w-fit`}
                    >
                      {getIconColor(item.status)} {item.status}
                    </span>
                  </TableCell>
                  <TableCell className="py-3 px-4">
                    {item.status === "Uploaded" && (
                      <div className="flex justify-center items-center">
                        <DropdownMenu>
                          <DropdownMenuTrigger className="focus:outline-none">
                            <button className="p-1 rounded">
                              <MoreHorizontal className="w-4 h-4" />
                            </button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end" className="w-40">
                            <DropdownMenuItem
                              onClick={() => handleDownload(item)}
                              className="hover:!bg-gray-100"
                            >
                              <Image
                                src={downloadIcon}
                                alt="Download"
                                width={14}
                                height={14}
                                className="mr-3"
                              />
                              Download
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              onClick={() => {
                                setSelectedFile(item);
                                setDeleteModalOpen(true);
                              }}
                              className="text-red-600 hover:!bg-red-50"
                            >
                              <Image
                                src={trashIcon}
                                alt="Delete"
                                width={14}
                                height={14}
                                className="mr-3"
                              />
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    )}
                  </TableCell>
                </TableRow>
              ))
            ) : (
              <TableRow className="border-b-2 border-gray-300 bg-white">
                <TableCell
                  colSpan={5}
                  className="py-3 px-4 text-center text-gray-500"
                >
                  No files found.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        ) : (
          <TableBody>
            {sourceList && sourceList.length > 0 ? (
              getSortedFileList().map((item, index) => (
                <TableRow
                  className="border-b-2 border-gray-300 bg-white"
                  key={index}
                >
                  <TableCell className="py-3 px-4 text-center border-r border-gray-300">
                    {index + 1}
                  </TableCell>
                  <TableCell className="py-3 px-4">
                    <span className="text-blue-600">{item.filename}</span>
                  </TableCell>
                  <TableCell className="py-3 px-4">{item.mimetype}</TableCell>
                  <TableCell className="py-3 px-4">{item.size}</TableCell>
                  <TableCell className="py-3 px-4 text-xs">
                    <span
                      className={`${
                        statusTextColor[item?.status] ||
                        "bg-gray-600 text-gray-800"
                      } rounded p-1 flex justify-start items-center gap-1 w-fit`}
                    >
                      {getIconColor(item?.status)} {item?.status}
                    </span>
                  </TableCell>
                  <TableCell className="py-3 px-4">
                    {item.status === "Uploaded" && (
                      <div className="flex justify-center items-center">
                        <DropdownMenu>
                          <DropdownMenuTrigger className="focus:outline-none">
                            <button className="p-1 rounded">
                              <MoreHorizontal className="w-4 h-4" />
                            </button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end" className="w-40">
                            <DropdownMenuItem
                              onClick={() => handleDownload(item)}
                              className="hover:!bg-gray-100"
                            >
                              <Image
                                src={downloadIcon}
                                alt="Download"
                                width={14}
                                height={14}
                                className="mr-3"
                              />
                              Download
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              onClick={() => {
                                setSelectedFile(item);
                                setDeleteModalOpen(true);
                              }}
                              className="text-red-600 hover:!bg-red-50"
                            >
                              <Image
                                src={trashIcon}
                                alt="Delete"
                                width={14}
                                height={14}
                                className="mr-3"
                              />
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    )}
                  </TableCell>
                </TableRow>
              ))
            ) : (
              <TableRow className="border-b-2 border-gray-300 bg-white">
                <TableCell
                  colSpan={5}
                  className="py-3 px-4 text-center text-gray-500"
                >
                  No files found.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        )}
      </Table>
      {(fileList.length > 0 || sourceList.length > 0) && (
        <Paginator
          page={pagination}
          size={"full"}
          totalPages={totalPages}
          showPageSize={true}
          onChange={(opts) => setPagination(opts)}
        />
      )}
      <DeleteModal
        isOpen={isDeleteModalOpen}
        onClose={() => setDeleteModalOpen(false)}
        onDelete={handleFileDelete}
        title={`Delete "${selectedFile?.filename}"?`}
      />
    </div>
  );
};

export default Files;
