// /* The above code is a React component named `Layout`. Here is a summary of what the code is doing: */
"use client";
import { useParams, useRouter } from "next/navigation";
import React, { useState, useEffect } from "react";
import tickCircle from "@/assets/icons/tick-circle.svg";
import alert from "@/assets/icons/alert-triangle.svg";
import processing from "@/assets/icons/processing.svg";
import Image from "next/image";
import { UpdateContext } from "@/context_api/updateContext";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";
import dots from "@/assets/icons/dots-vertical.svg";
import DeleteModal from "@/components/forms/deleteModal";
import {
  getSourceConnectorDetails,
  deleteSourceConnector,
  getSourceConnector,
} from "@/api/Workspace/WorkSpaceFiles";
import { showError, showSuccess } from "@/utils/toastUtils";
import { getFiles } from "@/api/common";

const Layout = ({ children }) => {
  const params = useParams();
  const router = useRouter();
  const [isDelete, setIsDelete] = useState(false);
  const [focusedSourceId, setFocusedSourceId] = useState(0);
  const [sourceList, setSourceList] = useState([]);
  const [sourceLoader, setSourceLoader] = useState(false);
  const [currentSourceID, setCurrentSourceID] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);
  const [fileTotals, setFileTotals] = useState({});
  const [uploadFileCount, setUploadFileCount] = useState(0);
  const [uploadFileLoading, setUploadFileLoading] = useState(true);
  const [pagination, setPagination] = useState({ page: 1, size: 25 });

  const handleRefresh = () => {
    setRefreshKey((prev) => prev + 1);
  };
  // get source details
  const getSourceList = async () => {
    setSourceLoader(true);
    setUploadFileLoading(true);
    const workspaceId = params?.workspaceId || null;
    const page = { page: pagination.page, size: pagination.size };
    try {
      const response = await getSourceConnector(workspaceId, page);
      const def_response = await getFiles(workspaceId, true, page);
      let items = response?.data?.data.items;
      const def_items = def_response?.data?.data?.total;
      setUploadFileCount(def_items);
      setUploadFileLoading(false);

      for (const source of items) {
        await fetchAndUpdateTotal(source?.sources?.source_id);
      }

      if (response.status === 200) {
        setSourceLoader(false);
        setSourceList(response?.data?.data?.items);
        handleRefresh();
      }
    } catch (e) {
      setSourceLoader(false);
      setUploadFileLoading(false);
    }
  };

  const fetchAndUpdateTotal = async (sourceId) => {
    try {
      const length_data = {
        workspaceId: params?.workspaceId,
        sourceId: sourceId,
        page: 1,
        size: 50,
      };

      const metadata = await getSourceConnectorDetails(length_data, pagination);
      const total = metadata?.data?.data?.total || 0;
      setFileTotals((prev) => ({
        ...prev,
        [sourceId]: total,
      }));
    } catch (error) {
      console.error(
        "Error fetching or updating total for source:",
        sourceId,
        error,
      );
    }
  };

  useEffect(() => {
    getSourceList();
  }, []);

  const getStatusIcon = (status) => {
    switch (status) {
      case "Processed":
        return <Image src={tickCircle} alt="tick with circle" />;
      case "Failed":
        return <Image src={alert} alt="tick with circle" />;
      case "Processing":
        return <Image src={processing} alt="tick with circle" />;
      default:
        return null;
    }
  };

  const getStatusTextColor = (status) => {
    switch (status) {
      case "Processed":
        return "text-green-20"; // Green for processed
      case "Failed":
        return "text-red-500"; // Red for failed
      case "Processing":
        return "text-blue-500"; // Blue for processing
      default:
        return "text-gray-600"; // Default color for other statuses
    }
  };

  const handleItemClick = async (sourceId) => {
    setFocusedSourceId(sourceId);
    router.push(`/workspace/${params?.workspaceId}/files/${sourceId || 0}`);
  };

  const handleDelete = async () => {
    const data = {
      id: params?.workspaceId,
      sourceId: currentSourceID,
    };

    try {
      const response = await deleteSourceConnector(data);

      if (response.status === 204) {
        showSuccess(`Source deleted successfully.`);
        router.push(`/workspace/${params?.workspaceId}/files/0`);
        await getSourceList();
      }
    } catch (error) {
      showError(`${error?.response?.data?.detail}`);
    } finally {
      setCurrentSourceID("");
      setIsDelete(false);
    }
  };

  return (
    <div className="grid grid-cols-[1fr_3fr] h-[calc(100vh)] bg-gray-50">
      <div className="bg-white max-h-screen overflow-auto">
        <div className="flex justify-between items-center text-base font-semibold pl-6 pt-4 pb-4 pr-6">
          <span className="">Files</span>
          <span
            className="text-xs font-medium text-blue-500 cursor-pointer"
            onClick={() =>
              router.push(`/get-started/?id=${params?.workspaceId}`)
            }
          >
            + New Files
          </span>
        </div>

        <div className="max-h-screen overflow-auto">
          <div>
            {!uploadFileLoading && (
              <div
                onClick={() => handleItemClick(0)}
                className={
                  focusedSourceId === 0
                    ? "bg-blue-20 border-blue-700"
                    : "border-white hover:bg-blue-20 hover:border-blue-700"
                }
              >
                <div className="py-3 px-4 border-s-2 border-white hover:bg-blue-20 hover:border-blue-700">
                  <div className="text-sm font-medium text-gray-800  flex justify-between items-center">
                    <span className="pl-2 w-5/6">
                      Upload Files
                      <div className="flex items-center gap-4 text-sm text-gray-600 mt-2">
                        <span>{uploadFileCount} files</span>
                      </div>
                    </span>
                  </div>
                </div>
                <hr className="m-0" />
              </div>
            )}
          </div>
          {sourceLoader ? (
            <div className="py-3 px-4 border-s-2 border-white hover:bg-blue-20 hover:border-blue-700 animate-pulse">
              <div className="text-sm font-medium text-gray-800 flex justify-between items-center">
                <span className="pl-2 w-5/6 h-4 bg-gray-300 rounded-md"></span>
                <div className="h-6 w-6 bg-gray-300 rounded-full"></div>
              </div>
              <div className="flex items-center gap-4 text-sm text-gray-600 mt-2">
                <span className="h-4 w-16 bg-gray-300 rounded-md"></span>
                <span className="flex items-center space-x-2">
                  <div className="h-4 w-4 bg-gray-300 rounded-full"></div>
                  <span className="h-4 w-20 bg-gray-300 rounded-md"></span>
                </span>
              </div>
            </div>
          ) : (
            sourceList.map((item) => (
              <div
                onClick={() => handleItemClick(item?.sources?.source_id)}
                onFocus={() => setFocusedSourceId(item?.sources?.source_id)}
                tabIndex={0}
                key={item?.sources?.source_id}
                className={
                  focusedSourceId === item?.sources?.source_id
                    ? "bg-blue-20 border-blue-700"
                    : "border-white hover:bg-blue-20 hover:border-blue-700"
                }
              >
                <div className="py-3 px-4 border-s-2 border-white hover:bg-blue-20 hover:border-blue-700">
                  <div className="text-sm font-medium text-gray-800  flex justify-between items-center">
                    <span className="pl-2 w-5/6">
                      {" "}
                      {item?.sources?.container_name ||
                        item?.sources?.database_name ||
                        item?.sources?.source_name}{" "}
                    </span>
                    <DropdownMenu>
                      <DropdownMenuTrigger className="focus:outline-none">
                        <Image
                          src={dots}
                          alt="options"
                          className="self-start cursor-pointer"
                        />
                      </DropdownMenuTrigger>
                      <DropdownMenuContent
                        align="start"
                        className="w-28 absolute"
                      >
                        <DropdownMenuItem
                          onClick={(e) => {
                            e.stopPropagation();
                            router.push(
                              `/get-started/?id=${params?.workspaceId}&sourceId=${item?.sources?.source_id}`,
                            );
                          }}
                          className="hover:!bg-blue-100"
                        >
                          Edit
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          onClick={(e) => {
                            e.stopPropagation();
                            setIsDelete(true);
                            setCurrentSourceID(item?.sources?.source_id);
                          }}
                          className="hover:!bg-blue-100"
                        >
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                  <div className="flex items-center gap-4 text-sm text-gray-600 mt-2 pl-2">
                    <span>
                      {fileTotals[item?.sources?.source_id] || 0} files
                    </span>
                    <span className="flex items-center space-x-2">
                      {getStatusIcon(item?.status)}
                      <span className={getStatusTextColor(item?.status)}>
                        {item?.status}
                      </span>
                    </span>
                  </div>
                </div>
                <hr className="m-0" />
              </div>
            ))
          )}
          {}
        </div>
      </div>
      <UpdateContext.Provider
        value={{ triggerUpdate: () => setUploadFileCount(uploadFileCount - 1) }}
      >
        <div key={refreshKey} className="max-h-screen overflow-auto">
          {children}
        </div>
      </UpdateContext.Provider>
      <DeleteModal
        title="Are you sure you want to delete this file?"
        isOpen={isDelete}
        onClose={() => setIsDelete(false)}
        onDelete={handleDelete}
      />
    </div>
  );
};

export default Layout;
