"use client";
import React, { useState, useEffect } from "react";
import { getDataSet } from "@/api/dataset";
import { useParams } from "next/navigation";
import { showError, showSuccess } from "@/utils/toastUtils";
import { agentic_constants } from "@/lib/AgenticConstants";

const DataSetFormSearch = ({ setIsOpen, selectedValue, setSelectedValue }) => {
  const params = useParams();
  const [selectedUsers, setSelectedUsers] = useState([]);
  const [fileList, setFileList] = useState([]);
  const [pagination, setPagination] = useState({ page: 1, size: 25 });
  const [totalPages, setTotalPages] = useState(1);

  const getDatasetList = async (ingested, type) => {
    try {
      const response = await getDataSet(
        params?.workspaceId,
        {
          page: 1,
          size: 50,
        },
        ingested,
        type,
      );

      if (response.status === 200) {
        setFileList(response?.data?.data?.items);
      }
    } catch (error) {
      showError(`${error?.response?.data?.detail}`);
    }
  };

  useEffect(() => {
    if (fileList && selectedValue) {
      setSelectedUsers(selectedValue || []);
    }
  }, [selectedValue, fileList]);

  useEffect(() => {
    if (params) {
      getDatasetList(
        agentic_constants.TRUTH_VALUES.TRUE,
        agentic_constants.DATASET_TYPE.UNSTRUCTURED,
      );
    }
  }, [params]);

  const handleCancel = () => {
    setIsOpen(false);
  };

  const getFileList = async () => {
    const data = {
      id: params?.workspaceId,
      page: 1,
      size: 50,
      only_uploaded: true,
    };
    const page = { page: pagination.page, size: pagination.size };
    const response = await getFiles(data?.id, true, page);
    if (response.status === 200) {
      setFileList(response.data.data.items);
      setTotalPages(response.data.data.total || 1);
    }
  };
  // useEffect(()=>{
  //   if(params?.workspaceId){
  //     getFileList()
  //   }
  // },[params?.workspaceId])
  const handleCheckboxChange = (name, checked) => {
    setSelectedUsers((prevState) => {
      if (checked) {
        // Add user to selected list
        return [...prevState, name];
      } else {
        // Remove user from selected list
        return prevState.filter((user) => user.id !== name.id);
      }
    });
  };

  const handleAddUsers = () => {
    setSelectedValue(selectedUsers);
    setIsOpen(false);
  };

  return (
    <div className="flex flex-col gap-4 p-4">
      {fileList?.map((item, index) => (
        <div className="flex gap-2 items-center" key={index}>
          <input
            type="checkbox"
            checked={selectedUsers.some((user) => user.id === item.id)}
            className="w-4 h-4"
            onChange={(e) =>
              handleCheckboxChange(
                {
                  id: item.id,
                  name: item.name,
                  knowledge_graph: item.knowledge_graph,
                },
                e.target.checked,
              )
            }
          />
          <div className="font-medium text-sm">{item?.name}</div>
        </div>
      ))}
      <div className="flex gap-4 mt-4 fixed bottom-0 left-0 right-0 bg-white p-4 border-t text-sm font-medium justify-end">
        <button className="border py-1 px-3 rounded" onClick={handleCancel}>
          Cancel
        </button>
        <button
          type="button"
          onClick={handleAddUsers}
          className="bg-blue-10 text-white py-1 px-3 rounded "
        >
          Added {selectedUsers?.length} datasets
        </button>
      </div>
    </div>
  );
};

export default DataSetFormSearch;
