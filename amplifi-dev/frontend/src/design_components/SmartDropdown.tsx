"use client";

import * as React from "react";
import { Check, ChevronDown, Search } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/design_components/badge";
import {
  OptionType,
  SmartDropdownProps,
} from "@/types/props/SmartDropdownProps";

export const SmartDropdown: React.FC<SmartDropdownProps> = ({
  options,
  value,
  onChange,
  variant = "single",
  placeholder = "Select...",
  searchable = false,
  showTags = false,
  className = "",
  state = "default",
  onOpenChange,
  showIconsInTags = true,
  isDisabled = false,
  isOpen: externalIsOpen,
}) => {
  const [internalOpen, setInternalOpen] = React.useState(false);

  // Use external control if provided, otherwise use internal state
  const open = externalIsOpen !== undefined ? externalIsOpen : internalOpen;
  const setOpen =
    externalIsOpen !== undefined
      ? (newOpen: boolean | ((prev: boolean) => boolean)) => {
          const nextOpen =
            typeof newOpen === "function" ? newOpen(open) : newOpen;
          onOpenChange?.(nextOpen);
        }
      : setInternalOpen;
  const [search, setSearch] = React.useState("");
  const triggerRef = React.useRef<HTMLButtonElement>(null);
  const [contentWidth, setContentWidth] = React.useState(0);

  // Adjust menu width when opening
  React.useEffect(() => {
    if (open && triggerRef.current) {
      setContentWidth(triggerRef.current.offsetWidth);
    }
  }, [open]);

  React.useEffect(() => {
    // Only call onOpenChange if we're using internal state
    if (externalIsOpen === undefined) {
      onOpenChange?.(open);
    }
  }, [open, onOpenChange, externalIsOpen]);

  const selectedValues = React.useMemo<OptionType[]>(() => {
    if (!value) return [];
    const arr = Array.isArray(value) ? value : [value];
    return options.filter((opt) => arr.includes(opt.value));
  }, [value, options]);

  const isSelected = (opt: OptionType) =>
    selectedValues.some((v) => v.value === opt.value);

  const toggleValue = (opt: OptionType) => {
    if (variant === "single") {
      onChange(opt);
      setOpen(false);
    } else {
      if (isSelected(opt)) {
        onChange(selectedValues.filter((v) => v.value !== opt.value));
      } else {
        onChange([...selectedValues, opt]);
      }
    }
  };

  const removeValue = (opt: OptionType) => {
    if (!isDisabled) {
      onChange(selectedValues.filter((v) => v.value !== opt.value));
    }
  };

  const filtered = search
    ? options.filter((o) =>
        o.label.toLowerCase().includes(search.toLowerCase()),
      )
    : options;

  const renderOptionIcon = (option: OptionType) => {
    if (!option.icon) return null;
    return (
      <div className="flex-shrink-0 w-4 h-4 flex items-center justify-center">
        {option.icon}
      </div>
    );
  };

  const renderSelectedContent = () => {
    if (selectedValues.length === 0) {
      return (
        <span className="truncate flex-1 text-left text-gray-500">
          {placeholder}
        </span>
      );
    }

    if (variant === "single") {
      const selected = selectedValues[0];
      return (
        <div className="flex items-center gap-2 flex-1 text-left">
          {selected.icon && renderOptionIcon(selected)}
          <span className="truncate">{selected.label}</span>
        </div>
      );
    }

    if (!showTags) {
      return (
        <span className="truncate flex-1 text-left">
          {selectedValues.length} selected
        </span>
      );
    }

    return null;
  };

  return (
    <div className={cn("relative w-full", className)}>
      <button
        ref={triggerRef}
        type="button"
        onClick={() => !isDisabled && setOpen((o) => !o)}
        disabled={isDisabled}
        className={cn(
          "w-full flex items-start border px-3 py-2 rounded-md text-sm bg-white",
          {
            "border-gray-300 hover:border-gray-400":
              state === "default" && !isDisabled,
            "border-red-500": state === "error",
            "bg-gray-100 border-gray-200 text-gray-500 cursor-not-allowed":
              isDisabled,
          },
        )}
      >
        <div className="w-full flex flex-col">
          {/* Header row with placeholder/label */}
          <div className="w-full flex justify-between items-center">
            {renderSelectedContent()}
          </div>

          {/* Tags section with chevron positioned on the right */}
          {showTags && variant === "multi" && selectedValues.length > 0 && (
            <div className="w-full flex justify-between items-start mt-1">
              <div className="flex flex-wrap gap-1 flex-1">
                {selectedValues.map((v) => (
                  <Badge
                    key={v.value}
                    variant="primary"
                    removable={!isDisabled}
                    onRemove={!isDisabled ? () => removeValue(v) : undefined}
                    icon={showIconsInTags ? v.icon : undefined}
                    iconPosition="left"
                  >
                    {v.label}
                  </Badge>
                ))}
              </div>
              <ChevronDown className="h-4 w-4 ml-2 flex-shrink-0 self-center" />
            </div>
          )}

          {/* Chevron for cases where no tags are shown */}
          {(!showTags ||
            variant === "single" ||
            selectedValues.length === 0) && (
            <div className="absolute right-3 top-1/2 -translate-y-1/2">
              <ChevronDown className="h-4 w-4" />
            </div>
          )}
        </div>
      </button>

      {/* dropdown menu */}
      {open && !isDisabled && (
        <div
          style={{ width: contentWidth, maxHeight: 300 }}
          className="absolute z-50 mt-1 bg-white border border-gray-200 rounded-md shadow-md overflow-y-auto"
        >
          {searchable && (
            <div className="p-2 border-b border-gray-200 sticky top-0 bg-white z-10">
              <div className="relative">
                <Search className="absolute left-2 top-3 h-4 w-4 text-gray-400" />
                <input
                  type="text"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search"
                  className="w-full pl-8 pr-2 py-2 border rounded text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
                  autoFocus
                />
              </div>
            </div>
          )}

          <div className="max-h-[240px] overflow-y-auto scrollbar-thin">
            {filtered.length === 0 ? (
              <div className="p-2 text-sm text-gray-500">No options</div>
            ) : (
              filtered.map((opt) => (
                <div
                  key={opt.value}
                  onClick={() => toggleValue(opt)}
                  className={cn(
                    "flex items-center px-3 py-2 cursor-pointer text-sm hover:bg-gray-100 border-b border-gray-200 last:border-b-0",
                    opt.disabled && "opacity-50 pointer-events-none",
                  )}
                >
                  {variant === "multi" ? (
                    <div
                      className={cn(
                        "flex items-center justify-center w-4 h-4 mr-2 border rounded flex-shrink-0",
                        isSelected(opt)
                          ? "bg-blue-600 border-blue-600"
                          : "border-gray-400",
                      )}
                    >
                      {isSelected(opt) && (
                        <Check className="w-3 h-3 text-white" />
                      )}
                    </div>
                  ) : (
                    <Check
                      className={cn(
                        "w-4 h-4 mr-2 text-blue-600 flex-shrink-0",
                        !isSelected(opt) && "opacity-0",
                      )}
                    />
                  )}

                  {/* Option icon */}
                  {(opt as OptionType).icon && (
                    <div className="mr-2">{renderOptionIcon(opt)}</div>
                  )}

                  <span className="truncate">{opt.label}</span>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
};
