import React, { useState } from "react";
import { Dialog, DialogContent, DialogOverlay } from "@/components/ui/dialog";
import warningIcon from "@/assets/icons/warning-icon.svg";
import Image from "next/image";
import { Button } from "@/components/ui/button";

const WarningModal = ({
  isOpen,
  onClose,
  onRegenerate,
  title,
  subTitle = "This action cannot be undone",
  submitText = "Delete",
  cancelText = "Cancel",
}) => {
  const [disabled, setDisabled] = useState(false);
  const handleOnClick = async () => {
    setDisabled(true);
    try {
      await onRegenerate();
    } finally {
      setDisabled(false);
      onClose();
    }
  };
  return (
    <Dialog open={isOpen} onOpenChange={(isOpen) => !isOpen && onClose()}>
      <DialogOverlay className="DialogOverlay z-50 flex items-end pl-[60%] bg-black-20 opacity-80"></DialogOverlay>
      <DialogContent className="sm:max-w-sm p-0 text-sm">
        <div className="flex gap-2 items-center justify-center flex-col font-semibold mt-14">
          <Image src={warningIcon} alt="delete icon" width={44} height={64} />
          <div>{title}</div>
          <div className="font-normal text-black-20">{subTitle}</div>
        </div>
        <div className="flex flex-end justify-center gap-4 p-4">
          <button
            type="submit"
            className=" px-4 py-2 border rounded text-sm outline-none"
            onClick={onClose}
          >
            {cancelText}
          </button>
          <Button
            isLoading={disabled}
            type="submit"
            className="bg-custom-warning text-white border px-4 py-2 rounded text-sm hover:bg-custom-warning/80"
            onClick={handleOnClick}
          >
            {submitText}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default WarningModal;
