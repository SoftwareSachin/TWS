import React from "react";
import Image from "next/image";

import Success from "@/assets/icons/success-state.svg";
import Caution from "@/assets/icons/caution-state.svg";
import Processing from "@/assets/icons/processing-state.svg";
import Alert from "@/assets/icons/alert-octagon.svg";

export type StatusType =
  | "default"
  | "loading"
  | "success"
  | "error"
  | "warning";
export type SizeType = "sm" | "md" | "lg";

export interface StatusBadgeProps {
  name: string;
  status?: StatusType;
  size?: SizeType;
  className?: string;
}

interface StatusConfig {
  bgColor: string;
  textColor: string;
  icon: string | null;
  iconAlt: string;
}

interface SizeConfig {
  padding: string;
  text: string;
  iconSize: number;
  gap: string;
}

const StatusBadge: React.FC<StatusBadgeProps> = ({
  name,
  status = "default",
  size = "md",
  className = "",
}) => {
  const statusConfig: Record<StatusType, StatusConfig> = {
    default: {
      bgColor: "bg-gray-200",
      textColor: "text-gray-700",
      icon: null,
      iconAlt: "",
    },
    loading: {
      bgColor: "bg-blue-100",
      textColor: "text-blue-700",
      icon: Processing,
      iconAlt: "Processing icon",
    },
    success: {
      bgColor: "bg-green-100",
      textColor: "text-green-700",
      icon: Success,
      iconAlt: "Success icon",
    },
    error: {
      bgColor: "bg-red-100",
      textColor: "text-red-700",
      icon: Caution,
      iconAlt: "Error icon",
    },
    warning: {
      bgColor: "bg-orange-100",
      textColor: "text-orange-700",
      icon: Alert,
      iconAlt: "Warning icon",
    },
  };

  const sizeConfig: Record<SizeType, SizeConfig> = {
    sm: {
      padding: "px-2 py-1",
      text: "text-xs",
      iconSize: 10,
      gap: "gap-1",
    },
    md: {
      padding: "px-3 py-1.5",
      text: "text-sm",
      iconSize: 12,
      gap: "gap-1.5",
    },
    lg: {
      padding: "px-4 py-2",
      text: "text-base",
      iconSize: 16,
      gap: "gap-2",
    },
  };

  const config = statusConfig[status];
  const sizes = sizeConfig[size];

  const baseClasses = "inline-flex items-center rounded-full font-medium";
  const combinedClasses = `
    ${baseClasses}
    ${config.bgColor}
    ${config.textColor}
    ${sizes.padding}
    ${sizes.text}
    ${sizes.gap}
    ${className}
  `
    .trim()
    .replace(/\s+/g, " ");

  return (
    <span className={combinedClasses}>
      {config.icon && (
        <Image
          src={config.icon}
          alt={config.iconAlt}
          width={sizes.iconSize}
          height={sizes.iconSize}
          className={status === "loading" ? "animate-spin" : ""}
        />
      )}
      {name}
    </span>
  );
};

export default StatusBadge;
