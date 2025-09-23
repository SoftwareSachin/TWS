import { cva } from "class-variance-authority";
import * as React from "react";
import * as LabelPrimitive from "@radix-ui/react-label";

export const labelVariants = cva(
  "text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70",
);

// Define the props for the Label component
export type LabelProps = React.ComponentPropsWithoutRef<
  typeof LabelPrimitive.Root
>;
