import {
  RadioGroup,
  RadioGroupItem,
} from "@/design_components/radio/radio-group";
import { RadioButtonProps } from "@/types/props/RadioButtonProps"; // adjust path as per your setup

export const RadioButton = ({
  value,
  onChange,
  options,
  errorMap = {},
  className = "",
  isDisabled,
}: RadioButtonProps) => {
  return (
    <RadioGroup
      value={value}
      onValueChange={onChange}
      className={`flex gap-4 ${className}`}
    >
      {options.map((option) => {
        const isSelected = value === option.value;
        return (
          <div key={option.value} className={`px-3 py-1 space-y-2 rounded-lg`}>
            {/* ✅ Wrap item and label inside a native label tag */}
            <label
              htmlFor={option.value}
              className="flex items-center space-x-2 cursor-pointer"
            >
              <RadioGroupItem
                value={option.value}
                id={option.value}
                selected={isSelected}
                className={`text-white`}
                disabled={isDisabled}
              />
              <span className="text-sm ml-[10px]">{option.label}</span>
            </label>

            {isSelected && option.renderContent}

            {errorMap?.[option.value] && (
              <div className="text-red-600 text-sm">
                {errorMap[option.value]}
              </div>
            )}
          </div>
        );
      })}
    </RadioGroup>
  );
};
