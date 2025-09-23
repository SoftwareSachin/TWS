import { OptionType } from "@/types/RadioButton";

export type RadioButtonProps = {
  value?: string;
  onChange: (value: string) => void;
  options: OptionType[];
  errorMap?: Record<string, string>;
  className?: string;
  isDisabled?: boolean;
};
