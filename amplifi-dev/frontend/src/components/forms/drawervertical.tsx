"use client";
import React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogOverlay,
  DialogTitle,
} from "@/components/ui/dialog";
import { DrawerVerticalProps } from "@/types/props/UiUtilProps";

const DrawerVertical: React.FC<DrawerVerticalProps> = ({
  isOpen,
  onClose,
  children,
  title,
  width,
}) => {
  return (
    <Dialog open={isOpen} onOpenChange={(isOpen) => !isOpen && onClose()}>
      {/* Overlay */}
      <DialogOverlay
        className="fixed -top-80 right-0 bottom-0 left-0 z-50 bg-black-20 bg-opacity-70"
        onClick={onClose}
      />

      {/* Right-aligned Dialog Content */}
      <DialogContent
        className={`fixed right-0 top-0 left-auto h-full ${width ? `w-[${width}]` : "w-72"} bg-white z-50 shadow-lg transition-transform transform ease-out duration-300 p-0 flex flex-col gap-0`}
        style={{
          transform: isOpen ? "translateX(0)" : "translateX(100%)",
        }}
      >
        {/* Header with only the necessary height */}
        <DialogHeader className="flex-none">
          <DialogTitle className="font-semibold border-b text-sm bg-gray-100 py-3 px-4 rounded-tl-md">
            {title}
          </DialogTitle>
        </DialogHeader>

        {/* Content area, which can scroll if needed */}
        <div className="flex-1 flex flex-col overflow-y-auto max-h-[calc(100vh - 40px)]">
          {children}
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default DrawerVertical;
