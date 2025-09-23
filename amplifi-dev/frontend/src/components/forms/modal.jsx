import React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogOverlay,
  DialogTitle,
} from "@/components/ui/dialog";

const Modal = ({ isOpen, onClose, children, title, size }) => {
  return (
    <Dialog
      open={isOpen}
      onOpenChange={(isOpen) => {
        if (!isOpen) {
          onClose();
        }
      }}
    >
      <DialogOverlay
        className="DialogOverlay z-50 flex items-end pl-[60%] bg-black-20 opacity-80"
        onClick={(e) => e.preventDefault()}
      ></DialogOverlay>
      <DialogContent
        className={`${size} sm:max-w-xl p-0`}
        onPointerDownOutside={(e) => e.preventDefault()}
        onInteractOutside={(e) => e.preventDefault()}
        onEscapeKeyDown={(e) => e.preventDefault()}
      >
        <DialogHeader>
          <DialogTitle className="font-semibold py-3 px-4 border text-sm bg-gray-10 rounded-t-lg">
            {title}
          </DialogTitle>
        </DialogHeader>
        {children}
      </DialogContent>
    </Dialog>
  );
};

export default Modal;
