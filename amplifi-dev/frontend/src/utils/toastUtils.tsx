import { toast, ToastOptions } from "react-toastify";
import CustomToast from "@/components/ui/toast";

const TOAST_ID = "GLOBAL_TOAST";

const showToast = (
  message: string,
  options: ToastOptions & { subMessage?: string } = {},
) => {
  const { subMessage, ...rest } = options;

  const toastContent = (
    <CustomToast
      message={message}
      subMessage={subMessage}
      type={rest.type as any}
    />
  );

  const config: ToastOptions = {
    ...rest,
    toastId: TOAST_ID,
    closeButton: false,
    hideProgressBar: true,
    autoClose: 3000,
    draggable: false,
    icon: false,
  };

  if (toast.isActive(TOAST_ID)) {
    toast.update(TOAST_ID, { ...config, render: toastContent });
  } else {
    toast(toastContent, config);
  }
};

export const showSuccess = (message: string, options: ToastOptions = {}) =>
  showToast(message, { ...options, type: "success" });

export const showError = (message: string, options: ToastOptions = {}) =>
  showToast(message, { ...options, type: "error" });

export const showInfo = (message: string, options: ToastOptions = {}) =>
  showToast(message, { ...options, type: "info" });

export const showWarning = (message: string, options: ToastOptions = {}) =>
  showToast(message, { ...options, type: "warning" });

export const dismissToast = () => toast.dismiss();
