import * as React from "react";

export interface DrawerVerticalProps {
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
  title: string;
  width?: string; // Optional width prop
}

export interface DialogOverlayProps
  extends React.ComponentPropsWithoutRef<"div"> {
  className?: string;
}

export interface DialogContentProps
  extends React.ComponentPropsWithoutRef<"div"> {
  className?: string;
  children: React.ReactNode;
}

export interface DialogHeaderProps extends React.HTMLProps<HTMLDivElement> {
  className?: string;
}

export interface DialogFooterProps extends React.HTMLProps<HTMLDivElement> {
  className?: string;
}

export interface DialogTitleProps extends React.HTMLProps<HTMLDivElement> {
  className?: string;
}

export interface DialogDescriptionProps
  extends React.HTMLProps<HTMLDivElement> {
  className?: string;
}
