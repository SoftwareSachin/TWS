import React, { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogOverlay,
} from "@/components/ui/dialog";
import { Pencil } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";

const WorkflowModal = ({
  isOpen,
  onClose,
  children,
  title,
  type,
  onSubmit,
  editableTitle,
  setEditableTitle,
  titleText,
  setTitleText,
  dataFlowId,
}) => {
  const searchParams = useSearchParams();
  const router = useRouter();
  const dataFlowIdFromParams = searchParams.get("dataFlow");

  // Handle title change
  const handleTitleChange = (e) => {
    setTitleText(e.target.value);
  };

  const handleCancel = () => {
    if (dataFlowIdFromParams) {
      router.push(`workflows/${dataFlowIdFromParams}`);
    }
    onClose();
  };

  return (
    <Dialog
      open={isOpen}
      onOpenChange={(isOpen) => {
        if (!isOpen) {
          onClose();
          if (dataFlowIdFromParams) {
            router.push(`workflows/${dataFlowIdFromParams}`);
          }
        }
      }}
    >
      <DialogOverlay className="z-50 flex items-end bg-black-20 opacity-80" />
      <DialogContent className="sm:max-w-full min-h-full p-0 flex flex-col gap-0">
        <DialogHeader>
          <div className="flex items-center gap-2">
            {editableTitle ? (
              <input
                type="text"
                value={titleText}
                onChange={handleTitleChange}
                onBlur={() => setEditableTitle(false)} // Save on blur
                className="font-semibold py-4 px-4 border text-sm bg-gray-10 rounded-lg focus:outline-none"
              />
            ) : (
              <DialogTitle className="font-semibold py-3 px-4 border text-sm bg-gray-10 rounded-lg flex-1">
                {titleText}
                <button
                  onClick={() => setEditableTitle(true)}
                  className="text-gray-500 p-2 hover:text-gray-700"
                >
                  <Pencil className="h-4 w-4" />
                </button>
              </DialogTitle>
            )}
          </div>
        </DialogHeader>

        <div className="flex-1">{children}</div>

        <DialogFooter>
          <DialogTitle className="flex justify-end border-t p-4 w-full gap-4 text-sm">
            <button
              onClick={handleCancel}
              className="bg-white border font-medium py-2 px-4 rounded-md"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md"
              onClick={onSubmit}
            >
              Next
            </button>
          </DialogTitle>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default WorkflowModal;
