import { Check, X, Info, AlertTriangle } from "lucide-react";

import React, { JSX } from "react";

interface CustomToastProps {
  message: string;
  subMessage?: string;
  type?: "success" | "error" | "info" | "warning";
}

const typeStyles: Record<string, string> = {
  success: "bg-green-50 border-green-600 text-green-800",
  error: "bg-red-50 border-red-600 text-red-800",
  info: "bg-blue-50 border-blue-600 text-blue-800",
  warning: "bg-yellow-50 border-yellow-600 text-yellow-800",
};

const iconMap: Record<string, JSX.Element> = {
  success: (
    <div className="w-4 h-4 mt-1 bg-[#48C1B5] rounded-[4px] flex items-center justify-center">
      <Check size={10} className="text-white" />
    </div>
  ),
  error: (
    <div className="w-4 h-4 bg-[#F4B0A1] mt-1 rounded-[4px] flex items-center justify-center">
      <X size={10} className="text-white" />
    </div>
  ),
  info: (
    <div className="w-4 h-4 mt-1 bg-blue-600 rounded-[4px] flex items-center justify-center">
      <Info className="text-white" />
    </div>
  ),
  warning: (
    <div className="w-4 h-4 mt-1 bg-yellow-600 rounded-[4px] flex items-center justify-center">
      <AlertTriangle className="text-white" />
    </div>
  ),
};

const CustomToast: React.FC<CustomToastProps> = ({
  message,
  subMessage,
  type = "info",
}) => {
  return (
    <div
      className={`relative flex w-[400px] max-w-full items-start rounded-md border px-4 py-3 ${typeStyles[type]}`}
    >
      <div className="w-5 flex-shrink-0">{iconMap[type]}</div>

      <div className="flex-1 text-md font-medium">
        {message}
        {subMessage && <p className="text-xs mt-1">{subMessage}</p>}
      </div>

      <div className="absolute top-2 right-2">
        <X size={14} className="text-gray-400" />
      </div>
    </div>
  );
};

export default CustomToast;
