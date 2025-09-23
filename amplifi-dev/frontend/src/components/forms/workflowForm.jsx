import Image from "next/image";
import React, { useState, useEffect } from "react";
import arrow from "@/assets/icons/arrow-down-large.svg";
import Select from "react-select";
import { ChevronRight } from "lucide-react";
import DrawerVertical from "./drawervertical";
import DestinationFormSearch from "./destinationFormSearch";
import DatasetListForm from "./datasetListForm";
import { getDatasetsByOrganization } from "@/api/dataset";
import { generateCronExpression } from "@/utils/convertToCrown";
import { reverseCronExpression } from "@/utils/convertCrown";
import { useUser } from "@/context_api/userContext";
import SelectDatasetForm from "@/components/forms/selectDatasetForm";

const WorkflowForm = ({
  setCronExpression,
  selectedValue,
  setSelectedValue,
  selectedDestinationValue,
  setSelectedDestinationValue,
  cronExpression,
}) => {
  // Options for Trigger Time and Frequency dropdowns
  const triggerTimeOptions = [
    { value: "morning", label: "Morning" },
    { value: "afternoon", label: "Afternoon" },
    { value: "evening", label: "Evening" },
  ];

  const frequencyOptions = [
    { value: "daily", label: "Daily" },
    { value: "weekly", label: "Weekly" },
    { value: "monthly", label: "Monthly" },
  ];

  // State for storing selected values
  const [triggerTime, setTriggerTime] = useState(null);
  const [frequency, setFrequency] = useState(null);
  const [isDrawer, setIsDrawer] = useState(false);
  const [isDestinationDrawer, setIsDestinationDrawer] = useState(false);
  const [dataSetList, setDataSetList] = useState([]);
  const { user } = useUser();
  // Other hooks
  useEffect(() => {
    getWorkSpaceList();
  }, []);

  useEffect(() => {
    if (cronExpression) {
      console.log("testing --- =", cronExpression);

      setTriggerTime(reverseCronExpression(cronExpression)?.triggerTime);
      setFrequency(reverseCronExpression(cronExpression)?.frequency);
    }
  }, [cronExpression]);
  // get workspace list
  const getWorkSpaceList = async () => {
    const data = {
      id: user?.clientId,
      page: 1,
      size: 50,
    };

    try {
      const response = await getDatasetsByOrganization(data.id);
      setDataSetList(response?.data?.data?.items);
      console.log(response?.data?.data?.items);
    } catch (e) {}
  };

  useEffect(() => {
    if (triggerTime && frequency) {
      const cron = generateCronExpression(triggerTime, frequency);
      setCronExpression(cron);
    }
  }, [triggerTime, frequency]);

  return (
    <div className="w-full h-[82vh] grid grid-cols-2 items-stretch bg-gray-100">
      <div className="flex flex-col items-center justify-center">
        <div className="bg-white flex flex-col gap-2 w-64 p-4">
          <div className="bg-gray-700 text-white rounded-md text-xs w-fit px-3">
            Dataset
          </div>
          <div
            className="text-sm border rounded flex items-center justify-between py-1 px-3 text-gray-400"
            onClick={() => setIsDrawer(true)}
          >
            {selectedValue ? selectedValue.name : "Select Dataset"}
            <ChevronRight className="h-4 w-4" />
          </div>
        </div>
        <Image src={arrow} alt="arrow icon" />
        <div className="bg-white flex flex-col gap-2 w-64 p-4 rounded-lg">
          <div className="bg-gray-700 text-white rounded-md text-xs w-fit px-3">
            Destination
          </div>
          <div
            className="text-sm border rounded flex items-center justify-between py-1 px-3 text-gray-400"
            onClick={() => setIsDestinationDrawer(true)}
          >
            {selectedDestinationValue
              ? selectedDestinationValue.name
              : "Select Destination"}{" "}
            <ChevronRight className="h-4 w-4" />
          </div>
        </div>
      </div>
      <div className="bg-white flex-1 p-4">
        <div className="text-base font-semibold">Schedule</div>
        <div className="flex gap-4 w-full">
          <div className="w-1/2">
            <label className="text-sm font-medium">Trigger Time</label>
            <Select
              options={triggerTimeOptions}
              placeholder="Select Trigger Time"
              className="mt-1 focus:outline-none text-xs"
              value={triggerTime}
              onChange={(selectedOption) => setTriggerTime(selectedOption)}
              components={{ IndicatorSeparator: () => null }}
            />
          </div>
          <div className="w-1/2">
            <label className="text-sm font-medium">Frequency</label>
            <Select
              options={frequencyOptions}
              placeholder="Select Frequency"
              className="mt-1 text-xs focus:outline-none "
              value={frequency}
              onChange={(selectedOption) => setFrequency(selectedOption)}
              components={{ IndicatorSeparator: () => null }}
            />
          </div>
        </div>
      </div>

      <DrawerVertical
        isOpen={isDrawer}
        onClose={() => setIsDrawer(false)}
        title="Select Dataset"
      >
        <SelectDatasetForm
          setIsOpen={setIsDrawer}
          parentId={user?.clientId}
          setSelectedDataset={(dataset) => setSelectedValue(dataset?.[0])}
          selectedDatasetId={selectedValue ? [selectedValue.id] : []}
          multiple={false}
          datasetFrom={"organization"}
        />
      </DrawerVertical>
      <DrawerVertical
        isOpen={isDestinationDrawer}
        onClose={() => setIsDestinationDrawer(false)}
        title="Select Destination"
      >
        <DestinationFormSearch
          setIsOpen={setIsDestinationDrawer}
          selectedValue={selectedDestinationValue}
          setSelectedValue={setSelectedDestinationValue}
        />
      </DrawerVertical>
    </div>
  );
};

export default WorkflowForm;
