import React from "react";
import { CheckboxGroupProps } from "@/types/props/CheckboxGroupProps";
import { Checkbox } from "@/design_components/check-box/Checkbox";

export const CheckboxGroup = ({
  value,
  onChange,
  options,
  errorMap = {},
  className = "",
}: CheckboxGroupProps) => {
  const handleToggle = (optionValue: string) => {
    const newValue = value.includes(optionValue)
      ? value.filter((val) => val !== optionValue)
      : [...value, optionValue];

    onChange(newValue);
  };

  return (
    <div className={`flex flex-wrap gap-4 ${className}`}>
      {options.map((option) => {
        const isChecked = value.includes(option.value);

        return (
          <div key={option.value} className="space-y-1">
            <label
              htmlFor={option.value}
              className={`flex items-center gap-2 px-2 py-1 rounded-lg cursor-pointer ${
                option.disabled ? "cursor-not-allowed opacity-50" : ""
              }`}
            >
              <Checkbox
                id={option.value}
                checked={isChecked}
                onCheckedChange={() =>
                  !option.disabled && handleToggle(option.value)
                }
                disabled={option.disabled}
              />
              <span className="text-sm">{option.label}</span>
            </label>

            {isChecked && option.renderContent}

            {errorMap?.[option.value] && (
              <div className="text-red-600 text-sm">
                {errorMap[option.value]}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};
