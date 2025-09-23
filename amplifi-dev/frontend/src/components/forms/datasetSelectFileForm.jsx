import { getFiles } from "@/api/common";
import React, { useState, useEffect } from "react";
import SearchBox from "@/design_components/utility/search-box";
import { filterByKey } from "@/utils/filter";

const DatasetSelectFileForm = ({
  setIsOpen,
  workspaceId,
  setSelectedFile,
  selectedFile,
}) => {
  const handleCancel = () => {
    setIsOpen(false);
  };

  const [selectedFileIDs, setSelectedFileIDs] = useState(selectedFile || []);
  const [searchText, setSearchText] = useState("");
  const [fileList, setFileList] = useState([]);
  const [filteredFileList, setFilteredFileList] = useState([]);
  console.log("file selected --", selectedFile, selectedFileIDs);

  const getFileList = async () => {
    const data = {
      id: workspaceId,
      only_uploaded: true,
    };

    const response = await getFiles(data?.id, true);
    console.log(response);
    if (response.status === 200) {
      setFileList(response.data.data);
      setFilteredFileList(response.data.data);
    }
  };
  useEffect(() => {
    getFileList();
  }, []);

  const handleSelectAll = (checked) => {
    setSelectedFileIDs([]);
    if (checked) {
      const fileIds = fileList.map((item) => item.id);
      setSelectedFileIDs(fileIds);
    }
  };
  const handleCheckboxChange = (id, checked) => {
    setSelectedFileIDs((prevState) => {
      if (checked) {
        // Add user to selected list
        return [...prevState, id];
      } else {
        // Remove user from selected list
        return prevState.filter((file) => file !== id);
      }
    });
  };

  const handleAddFiles = () => {
    setSelectedFile(selectedFileIDs);
    setIsOpen(false);
  };

  return (
    <div className="flex flex-col gap-4 p-4">
      <div className="flex gap-4">
        <SearchBox
          className={"flex-1"}
          value={searchText}
          onChange={(e) => {
            setSearchText(e.target.value);
            setFilteredFileList(
              filterByKey(fileList, "filename", e.target.value),
            );
          }}
          placeholder="Search files"
        />
        {filteredFileList?.length > 1 && (
          <div className="flex gap-2 items-center justify-end">
            <input
              type="checkbox"
              className="w-4 h-4"
              onChange={(e) => handleSelectAll(e.target.checked)}
            />
            <div className=" text-sm font-bold">Select All</div>
          </div>
        )}
      </div>
      <div className="flex flex-1 max-h-[calc(100vh-250px)] overflow-y-auto flex-col gap-4 pb-8">
        {filteredFileList?.map((item, index) => (
          <div className="flex gap-2 items-center" key={item.id}>
            <input
              type="checkbox"
              className="w-4 h-4"
              onChange={(e) => handleCheckboxChange(item.id, e.target.checked)}
              checked={selectedFileIDs.includes(item.id)}
            />
            <div
              title={item?.filename}
              className="font-medium text-md truncate max-w-[430px]"
            >
              {item?.filename}
            </div>
          </div>
        ))}
      </div>
      <div className="flex gap-4 mt-4 fixed bottom-0 left-0 right-0 bg-white p-4 border-t text-sm font-medium justify-end z-10">
        <button className="border py-1 px-3 rounded" onClick={handleCancel}>
          Cancel
        </button>
        <button
          type="button"
          onClick={handleAddFiles}
          className="bg-blue-10 text-white py-1 px-3 rounded "
        >
          Add {selectedFileIDs?.length} files
        </button>
      </div>
    </div>
  );
};

export default DatasetSelectFileForm;
