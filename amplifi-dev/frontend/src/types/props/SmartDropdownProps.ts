export type OptionType = {
  label: string;
  value: string;
  tool_type?: string;
  dataset_required?: boolean;
  disabled?: boolean;
  icon?: React.ReactNode;
};

export type SmartDropdownProps = {
  options: OptionType[];
  value: string[] | string | null;
  onChange: (value: OptionType[] | OptionType | null) => void;
  variant?: "single" | "multi";
  placeholder?: string;
  searchable?: boolean;
  showTags?: boolean;
  className?: string;
  state?: "default" | "error";
  onOpenChange: (isOpen: boolean) => void;
  showIconsInTags?: boolean;
  isDisabled?: boolean;
  isOpen?: boolean;
};
