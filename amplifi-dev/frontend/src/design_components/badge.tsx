import React from "react";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";

interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "primary" | "secondary" | "outline";
  removable?: boolean;
  onRemove?: (e: React.MouseEvent) => void;
  icon?: React.ReactNode;
  iconPosition?: "left" | "right";
}

export const Badge: React.FC<BadgeProps> = ({
  className,
  children,
  variant = "default",
  removable = false,
  onRemove,
  icon,
  iconPosition = "left",
  ...props
}) => {
  const variants = {
    default: "bg-blue-100 text-blue-700 hover:bg-blue-200",
    primary: "bg-blue-600 text-white hover:bg-blue-700",
    secondary: "bg-gray-100 text-gray-700 hover:bg-gray-200",
    outline: "border border-gray-300 bg-white text-gray-700 hover:bg-gray-50",
  };

  const renderIcon = () => {
    if (!icon) return null;
    return (
      <div className="flex-shrink-0 w-3 h-3 flex items-center justify-center">
        {icon}
      </div>
    );
  };

  return (
    <div
      className={cn(
        "inline-flex items-center rounded-full text-xs px-3 py-1 gap-1 transition-colors",
        variants[variant],
        className,
      )}
      {...props}
    >
      {icon && iconPosition === "left" && renderIcon()}
      <span className="truncate">{children}</span>
      {icon && iconPosition === "right" && renderIcon()}
      {removable && onRemove && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onRemove(e);
          }}
          className="hover:opacity-70 transition-opacity flex-shrink-0"
          type="button"
        >
          <X className="w-3 h-3" />
        </button>
      )}
    </div>
  );
};
