"use client";

import React from "react";
import * as LabelPrimitive from "@radix-ui/react-label";

import { cn } from "@/lib/utils";
import { labelVariants } from "@/types/props/LabelProps";
import { LabelProps } from "@radix-ui/react-label";

const Label = React.forwardRef<HTMLLabelElement, LabelProps>(
  ({ className, ...props }, ref) => (
    <LabelPrimitive.Root
      ref={ref}
      className={cn(labelVariants(), className)}
      {...props}
    />
  ),
);

Label.displayName = LabelPrimitive.Root.displayName;

export { Label };
