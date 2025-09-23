import { ReactNode } from "react";

export type OptionType = {
  label: string;
  value: string;
  renderContent?: ReactNode;
};
