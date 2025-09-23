import { agentic_constants } from "@/lib/AgenticConstants";
import { OptionType } from "@/types/Agentic";
import { Checkbox, CheckboxIndicator } from "@radix-ui/react-checkbox";
import { CheckIcon } from "@radix-ui/react-icons";
import {
  MultiValueGenericProps,
  OptionProps,
  StylesConfig,
} from "react-select";
import MCP from "@/assets/icons/MCP.svg";
import System from "@/assets/icons/tools.svg";
import Image from "next/image";

export const userSelectCustomStyles = {
  control: (provided: any) => ({
    ...provided,
    borderColor: "#2929297A",
    boxShadow: "1px solid grey",
    "&:hover": {
      borderColor: "#2929297A",
    },
    padding: "0 0.5rem",
    height: "40px",
    minHeight: "41px",
    fontSize: "14px",
    borderRadius: "6px",
    color: "#000",
  }),
  option: (provided: any, state: { isFocused: any }) => ({
    ...provided,
    backgroundColor: state.isFocused ? "#E5E7EB" : "white",
    color: "#374151",
    cursor: "pointer",
    "&:active": {
      backgroundColor: "#E5E7EB",
    },
  }),
  singleValue: (provided: any) => ({
    ...provided,
    color: "#374151",
  }),
  dropdownIndicator: (provided: any) => ({
    ...provided,
    padding: "0 8px",
    fontSize: "14px",
    color: "#374151",
    "&:hover": {
      color: "#374151",
    },
  }),
};

export const customStylesChatApp: StylesConfig<OptionType, false> = {
  control: (provided, state) => ({
    ...provided,
    border: "1px solid #d1d5db",
    borderRadius: "4px",
    backgroundColor: "#ffffff",
    boxShadow: state.isFocused ? "0 0 0 1px #007bff" : "none",
    "&:hover": {
      borderColor: "#9ca3af",
    },
    padding: "2px",
    fontSize: "0.875rem",
  }),
  menu: (provided) => ({
    ...provided,
    backgroundColor: "#ffffff",
    border: "1px solid #d1d5db",
    borderRadius: "4px",
    boxShadow: "0 2px 8px rgba(0, 0, 0, 0.1)",
    marginTop: "4px",
    zIndex: 10,
  }),
  option: (provided, state) => ({
    ...provided,
    backgroundColor: state.isFocused ? "#f1f5f9" : "#ffffff",
    color: "#1f2937",
    padding: "8px",
    fontSize: "0.875rem",
    "&:active": {
      backgroundColor: "#e2e8f0",
    },
  }),
};

export const customStylesWorkspaceTool: StylesConfig<OptionType, true> = {
  control: (provided, state) => ({
    ...provided,
    border: "1px solid #ababab",
    borderRadius: "4px",
    backgroundColor: "#ffffff",
    boxShadow: state.isFocused ? "0 0 0 1px #1E2BA3" : "none",
    "&:hover": {
      //borderColor: "#9ca3af",
    },
    padding: "2px",
    fontSize: "0.875rem",
  }),
  menu: (provided) => ({
    ...provided,
    backgroundColor: "#ffffff",
    border: "1px solid #d1d5db",
    borderRadius: "1px",
    boxShadow: "0 2px 8px rgba(0, 0, 0, 0.1)",
    marginTop: "4px",
    zIndex: 10,
  }),
  option: (provided, state) => ({
    ...provided,
    backgroundColor: state.isFocused ? "#f1f5f9" : "#ffffff",
    color: "#1f2937",
    padding: "8px",
    fontSize: "0.875rem",
    "&:active": {
      backgroundColor: "#e2e8f0",
    },
  }),
  multiValue: (provided) => ({
    ...provided,
    backgroundColor: "#007bff",
    borderRadius: "22px",
  }),
  multiValueLabel: (provided) => ({
    ...provided,
    color: "#ffffff",
    marginLeft: "8px",
    padding: "4px",
  }),
  multiValueRemove: (provided) => ({
    ...provided,
    color: "#ffffff",
    borderRadius: "0 10px 10px 0",
    ":hover": {
      backgroundColor: "#0056b3",
      color: "#ffffff",
    },
    marginRight: "4px",
  }),
};

export const customStylesAgent: StylesConfig<OptionType, true> = {
  control: (provided, state) => ({
    ...provided,
    border: "1px solid #d1d5db",
    borderRadius: "4px",
    backgroundColor: "#ffffff",
    boxShadow: state.isFocused ? "0 0 0 1px #007bff" : "none",
    "&:hover": {
      borderColor: "#9ca3af",
    },
    padding: "2px",
    fontSize: "0.875rem",
  }),
  menu: (provided) => ({
    ...provided,
    backgroundColor: "#ffffff",
    border: "1px solid #d1d5db",
    borderRadius: "4px",
    boxShadow: "0 2px 8px rgba(0, 0, 0, 0.1)",
    marginTop: "4px",
    zIndex: 10,
  }),
  menuList: (provided) => ({
    ...provided,
    maxHeight: "120px",
  }),
  option: (provided, state) => ({
    ...provided,
    backgroundColor: state.isFocused ? "#f1f5f9" : "#ffffff",
    color: "#1f2937",
    padding: "8px",
    fontSize: "0.875rem",
    "&:active": {
      backgroundColor: "#e2e8f0",
    },
  }),
  multiValue: (provided, { data }) => ({
    ...provided,
    backgroundColor: data.tool_type === "mcp" ? "#3B65B2" : "#8B5CF6",
    borderRadius: "10px",
  }),
  multiValueLabel: (provided, { data }) => ({
    ...provided,
    color: "#ffffff",
    padding: "4px 8px",
  }),
  multiValueRemove: (provided, { data }) => ({
    ...provided,
    color: "#ffffff",
    borderRadius: "0 10px 10px 0",
    ":hover": {
      backgroundColor: data.tool_type === "mcp" ? "#3B65B2" : "#8B5CF6",
      color: "#ffffff",
    },
  }),
};

export const CustomMultiValueLabel = (
  props: MultiValueGenericProps<OptionType, true>,
) => {
  const { data } = props;
  const icon =
    data.tool_type === agentic_constants.TOOL_KIND.MCP ? MCP : System;

  return (
    <div className="flex pl-2 pr-2 items-center space-x-1 text-white">
      <Image
        src={icon}
        alt="tool-icon"
        width={12}
        height={16}
        className="invert brightness-0"
      />
      <span>{data.label}</span>
    </div>
  );
};

export const CustomOptionToolPage = ({
  children,
  isSelected,
  innerProps,
  innerRef,
  data,
  setValue,
  getValue,
}: OptionProps<OptionType, true>) => {
  const handleCheckboxChange = (checked: boolean) => {
    const currentValues = getValue();
    const newValues = checked
      ? [...currentValues, data]
      : currentValues.filter((opt) => opt.value !== data.value);
    setValue(newValues, "select-option");
  };

  return (
    <div
      ref={innerRef}
      {...innerProps}
      className="flex items-center space-x-2 px-2 py-1"
      onClick={(e) => e.stopPropagation()}
    >
      <Checkbox
        checked={isSelected}
        onCheckedChange={handleCheckboxChange}
        className="w-5 h-5 border-2 rounded-sm flex items-center justify-center data-[state=checked]:bg-blue-600 data-[state=checked]:text-white"
      >
        <CheckboxIndicator className="flex items-center justify-center">
          <CheckIcon className="w-4 h-4" />
        </CheckboxIndicator>
      </Checkbox>
      <span>{children}</span>
    </div>
  );
};
