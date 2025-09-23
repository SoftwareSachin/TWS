"use client";
import React, { useState, useEffect } from "react";
import {
  RadioGroup,
  RadioGroupItem,
} from "@/design_components/radio/radio-group"; // Ensure correct import path
import { getDestination } from "@/api/common";
import { useUser } from "@/context_api/userContext";

const DestinationFormSearch = ({
  setIsOpen,
  selectedValue,
  setSelectedValue,
}) => {
  const [destinationList, setDestinationList] = useState();

  const { user } = useUser();
  const getDestinationList = async () => {
    try {
      const response = await getDestination(user?.clientId);

      if (response?.status === 200) {
        setDestinationList(response?.data?.data?.items);
      }
    } catch (e) {
      console.error("Error fetching destinations:", e);
    }
  };

  useEffect(() => {
    getDestinationList();
  }, []);

  const handleRadioChange = (value) => {
    setSelectedValue(value);
    setIsOpen?.(false); // Optional chaining for function calls
  };

  return (
    <div className="flex flex-col gap-4 overflow-scroll">
      <RadioGroup value={selectedValue} onValueChange={handleRadioChange}>
        {destinationList?.map((option) => (
          <div
            key={option?.id}
            className="flex space-x-2 border-b break-words flex-col font-medium text-sm py-3 px-4"
          >
            <div className="flex gap-2 items-center">
              <RadioGroupItem
                value={{ id: option?.id, name: option?.name }}
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

export default DestinationFormSearch;
