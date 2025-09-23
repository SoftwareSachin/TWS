import { getDataSet, getDatasetsByOrganization } from "@/api/dataset";
import React, { useState, useEffect } from "react";
import Paginator from "@/components/utility/paginator";
import { Page } from "@/types/Paginated";
import { Dataset, DatasetResponse } from "@/types/Dataset";
import { SelectDatasetProps } from "@/types/props/DatasetProps";

const SelectDatasetForm: React.FC<SelectDatasetProps> = ({
  setIsOpen,
  parentId,
  selectedDataset,
  setSelectedDataset,
  type,
  multiple = true,
  datasetFrom = "workspace",
  setSelectedDatasetId,
  selectedDatasetId,
}) => {
  const [pagination, setPagination] = useState<Page>({ page: 1, size: 10 });
  const [totalPages, setTotalPages] = useState(1);
  const [selectedDatasetIds, setSelectedDatasetIds] = useState<string[]>(
    selectedDatasetId || [],
  );
  const [datasetList, setDatasetList] = useState<DatasetResponse[]>([]);

  const handleCancel = () => {
    setIsOpen(false);
  };

  const getDatasetList = async (ingested?: boolean) => {
    const response =
      datasetFrom === "workspace"
        ? await getDataSet(parentId, pagination, ingested, type)
        : await getDatasetsByOrganization(parentId, pagination);
    if (response.status === 200) {
      setDatasetList(response.data.data.items);
      setTotalPages(response.data.data.total || 1); // Assuming backend returns this
    }
  };

  useEffect(() => {
    getDatasetList(true);
  }, [pagination]);

  const handleCheckboxChange = (name: string, checked: boolean) => {
    if (multiple) {
      setSelectedDatasetIds((prevState) => {
        if (checked) {
          return [...prevState, name];
        } else {
          return prevState.filter((dataset) => dataset !== name);
        }
      });
    } else {
      setSelectedDatasetIds([name]);
    }
  };

  const handleAddDatasets = () => {
    if (setSelectedDatasetId) {
      setSelectedDatasetId(selectedDatasetIds);
    }
    if (setSelectedDataset) {
      setSelectedDataset(
        datasetList.filter((x) => selectedDatasetIds.includes(x.id)),
      );
    }
    setIsOpen(false);
  };

  return (
    <>
      <div className="flex flex-1 max-h-[85vh] overflow-y-auto flex-col gap-4 p-4">
        {datasetList?.map((item, index) => (
          <div className="flex gap-2 items-center" key={index}>
            <input
              type="checkbox"
              className="w-4 h-4"
              onChange={(e) => handleCheckboxChange(item.id, e.target.checked)}
              checked={selectedDatasetIds.includes(item.id)}
            />
            <div className="font-medium text-sm">{item?.name}</div>
          </div>
        ))}
      </div>

      <Paginator
        page={pagination}
        totalPages={totalPages}
        onChange={(opts) => setPagination(opts)}
        size="small"
      />

      <div className="flex gap-4 mt-2 bg-white p-4 border-t text-sm font-medium justify-end">
        <button className="border py-1 px-3 rounded" onClick={handleCancel}>
          Cancel
        </button>
        <button
          type="button"
          onClick={handleAddDatasets}
          className="bg-blue-10 text-white py-1 px-3 rounded"
        >
          Add {selectedDatasetIds?.length} Datasets
        </button>
      </div>
    </>
  );
};

export default SelectDatasetForm;
