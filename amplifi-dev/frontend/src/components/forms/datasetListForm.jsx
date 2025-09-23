"use client";
import React, { useState } from "react";
import {
  RadioGroup,
  RadioGroupItem,
} from "@/design_components/radio/radio-group"; // Ensure correct import path

const DatasetListForm = ({
  setIsOpen,
  selectedValue,
  setSelectedValue,
  dataSetList,
}) => {
  const handleRadioChange = (id) => {
    // Find the selected option from the dataSetList array
    const selectedOption = dataSetList.find((option) => option.id === id);
    if (selectedOption) {
      // Set the selected value as an object with `id` and `name`
      setSelectedValue({
        id: selectedOption.id,
        name: selectedOption.name,
      });
    }
    setIsOpen(false);
  };

  return (
    <div className="flex flex-col gap-4 overflow-scroll">
      <RadioGroup value={selectedValue?.id} onValueChange={handleRadioChange}>
        {dataSetList?.map((option) => (
          <div
            key={option?.id}
            className="flex space-x-2 border-b break-words flex-col font-medium text-sm py-3 px-4"
          >
            <div className="flex gap-2 items-center">
              <RadioGroupItem
                value={option?.id}
                id={option?.id}
                className={`w-4 h-4 rounded-full border-2 
                  ${selectedValue?.id === option?.id ? "bg-blue-500 border-blue-500 text-white" : "border-gray-300"}
                  ${selectedValue?.id === option?.id ? "checked:bg-white" : ""}`}
              />
              <label htmlFor={option?.id}>{option?.name}</label>
            </div>
            <div className="text-gray-500 !ms-6">{option?.description}</div>
          </div>
        ))}
      </RadioGroup>
    </div>
  );
};

export default DatasetListForm;
