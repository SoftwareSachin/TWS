import React, { useState } from "react";
import clsx from "clsx";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogOverlay,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

interface LargeModalProps {
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
  title?: string;
  type?: string;
  onSubmit?: () => void;
  actionButton?: string;
  fullWidth?: boolean;
  fullHeight?: boolean;
  disabled?: boolean;
  onRegenerate?: () => void;
  showRegenerate?: boolean;
  showDelete?: boolean;
  regenerateDisabled?: boolean;
  deleteDisabled?: boolean;
  hideCancelButton?: boolean;
  hideSubmitButton?: boolean;
  regenerateText?: string;
  deleteText?: string;
  cancelText?: string;
  onDelete?: () => void;
  regenerateButtonClassName?: string;
}

const LargeModal: React.FC<LargeModalProps> = ({
  isOpen,
  onClose,
  children,
  title,
  type,
  onSubmit,
  actionButton,
  cancelText,
  fullWidth = true,
  fullHeight = true,
  disabled = false,
  onRegenerate,
  showDelete = false,
  showRegenerate = false,
  regenerateDisabled = false,
  deleteDisabled = false,
  hideCancelButton = false,
  hideSubmitButton = false,
  regenerateText = "Regenerate",
  deleteText = "Delete",
  onDelete,
  regenerateButtonClassName,
}) => {
  const [configureModal, setConfigureModal] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleAction = () => {
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
    }, 1000);
    onSubmit?.();
  };
  return (
    <Dialog open={isOpen} onOpenChange={(isOpen) => !isOpen && onClose()}>
      <DialogOverlay className="DialogOverlay z-50 fixed inset-0 bg-black-20 opacity-80 flex items-end pl-[60%]" />
      <DialogContent
        className={clsx(
          "p-0 flex flex-col bg-white rounded-lg shadow-lg",
          fullWidth ? "sm:max-w-full w-full" : "max-w-lg",
          fullHeight ? "h-[100vh]" : "max-h-[85vh]",
        )}
      >
        <DialogHeader>
          <DialogTitle className="font-semibold py-3 px-4 border text-sm rounded-t-lg">
            {title}
          </DialogTitle>
        </DialogHeader>
        <div className="flex-1 self-center w-full max-h-[calc(100vh-95px)] overflow-auto">
          {children}
        </div>

        <DialogFooter>
          <DialogTitle
            className={"flex justify-end border-t p-2 w-full gap-4 text-sm"}
          >
            {!hideCancelButton && (
              <Button
                onClick={onClose}
                className="bg-white text-black border font-medium py-2 px-4 rounded-md"
              >
                {cancelText ?? "Cancel"}
              </Button>
            )}
            {showDelete && (
              <Button
                type="button"
                className="bg-white text-black border font-medium py-2 px-4 rounded-md"
                onClick={() => onDelete?.()} // ✅ function that calls onDelete when clicked
                disabled={deleteDisabled}
              >
                {deleteText}
              </Button>
            )}
            {showRegenerate && (
              <Button
                type="button"
                className={
                  regenerateButtonClassName ??
                  "bg-white text-black border font-medium py-2 px-4 rounded-md"
                }
                onClick={() => onRegenerate?.()}
                disabled={regenerateDisabled}
              >
                {regenerateText}
              </Button>
            )}

            {!hideSubmitButton && (
              <Button
                type="submit"
                isLoading={loading}
                className="bg-blue-600 text-white font-medium py-2 px-4 rounded-md"
                onClick={handleAction}
                disabled={disabled}
              >
                {actionButton}
              </Button>
            )}
          </DialogTitle>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default LargeModal;
