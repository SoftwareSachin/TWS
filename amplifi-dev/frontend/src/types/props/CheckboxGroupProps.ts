export type OptionType = {
  value: string;
  label: string;
  disabled?: boolean;
  renderContent?: React.ReactNode;
};

export type CheckboxGroupProps = {
  value: string[]; // array of selected values
  onChange: (value: string[]) => void;
  options: OptionType[];
  errorMap?: Record<string, string>;
  className?: string;
};
