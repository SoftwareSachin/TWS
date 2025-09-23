import { ControllerProps, FieldValues } from "react-hook-form";
import * as React from "react";
import { Label } from "@/components/ui/label";
import { Slot } from "@radix-ui/react-slot";

export type FormFieldContextValue = {
  name: string;
};

export type FormFieldProps<T extends FieldValues> = ControllerProps<T>;

export const FormItemContext = React.createContext<{ id: string } | undefined>(
  undefined,
);

export type FormItemProps = React.HTMLAttributes<HTMLDivElement>;

export type FormLabelProps = React.ComponentPropsWithoutRef<typeof Label>;

export type FormControlProps = React.ComponentPropsWithoutRef<typeof Slot>;

export type FormDescriptionProps = React.HTMLAttributes<HTMLParagraphElement>;

export type FormMessageProps = React.HTMLAttributes<HTMLParagraphElement> & {
  children?: React.ReactNode;
};
