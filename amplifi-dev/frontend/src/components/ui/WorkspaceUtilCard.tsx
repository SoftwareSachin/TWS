import Image from "next/image";
import React from "react";
import rightArrow from "@/assets/icons/right-single-arrow-blue.svg";
import systemTool from "@/assets/icons/settings.svg";
import externalIcon from "@/assets/icons/globe.svg";
import internalIcon from "@/assets/icons/local.svg";
import MCP from "@/assets/icons/MCP.svg";
import AGENT from "@/assets/icons/agents.svg";
import FILES from "@/assets/icons/files.svg";
import azureIcon from "@/assets/icons/azure-icon.svg";
import databaseIcon from "@/assets/icons/database-icon.svg";
import localIcon from "@/assets/icons/local-icon.svg";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "./dropdown-menu";
import NewAvatar from "./newAvatar";
import { WorkspaceUtilCardProps } from "@/types/props/WorkspaceUtilCardProps";
import Link from "next/link";
import { Button } from "./button";
import dots from "@/assets/icons/dotsVerticals.svg";
import { constants } from "@/lib/constants";

const getTagIcon = (tag: string) => {
  switch (tag?.toLowerCase()) {
    case "system tool":
      return systemTool;
    case "mcp":
      return MCP;
    case "external":
      return externalIcon;
    case "internal":
      return internalIcon;
    case "agentic":
      return AGENT;
    case "files":
      return FILES;
    default:
      return systemTool;
  }
};

const getSourceTypeIcon = (sourceType: string | null) => {
  switch (sourceType?.toLowerCase()) {
    case "azure_storage":
      return azureIcon; // External/cloud icon for Azure
    case "pg_db":
      return databaseIcon; // Database icon for PostgreSQL
    case "mysql_db":
      return databaseIcon; // Database icon for MySQL
    default:
      return localIcon; // Local/internal icon for local data
  }
};

const WorkspaceUtilCard: React.FC<WorkspaceUtilCardProps> = ({
  title,
  description,
  tag,
  allToolNames,
  allAgentNames,
  onEdit,
  onDelete,
  onShowDetails,
  actionUrl,
  actionText,
  openInNewTab = false,
  onActionClick,
  sourceType,
  isEditable = true,
}) => {
  const getActionText = () => {
    if (actionText) return actionText;
    return "Show Details";
  };

  const handleActionClick = (e: React.MouseEvent) => {
    if (onActionClick) {
      e.preventDefault();
      onActionClick();
      return;
    }
  };

  const getActionUrl = () => {
    if (actionUrl) return actionUrl;
    return ``;
  };

  const renderActionButton = () => {
    const buttonContent = (
      <>
        <span className="text-custom-customBlue text-small font-medium">
          {getActionText()}
        </span>
        <Image src={rightArrow} alt="Arrow Right Icon" width={16} height={16} />
      </>
    );

    if (onShowDetails) {
      return (
        <button
          type="button"
          className="bg-transparent rounded outline-offset-[-1px] flex items-center gap-0.5 cursor-pointer text-[12px]"
          onClick={(e) => {
            e.preventDefault();
            onShowDetails();
          }}
        >
          {buttonContent}
        </button>
      );
    }

    if (actionUrl) {
      return (
        <Link
          href={actionUrl}
          className="bg-transparent rounded outline-offset-[-1px] flex items-center gap-0.5 cursor-pointer text-[12px]"
          {...(openInNewTab && {
            target: "_blank",
            rel: "noopener noreferrer",
          })}
        >
          {buttonContent}
        </Link>
      );
    }

    return (
      <Button
        type="button"
        className="bg-transparent rounded outline-offset-[-1px] flex items-center gap-0.5 text-[12px] opacity-50 cursor-not-allowed"
        disabled
      >
        {buttonContent}
      </Button>
    );
  };

  return (
    <div className="w-full h-[190px] p-4 relative rounded-[4px] shadow-[0px_1px_3px_0px_rgba(0,0,0,0.16),0px_1px_2px_-1px_rgba(0,0,0,0.16)] inline-flex justify-start items-start gap-20 overflow-hidden bg-[linear-gradient(0deg,_rgba(255,255,255,0.25)_0%,_rgba(255,255,255,0.25)_100%),_linear-gradient(119deg,_#EFF1FE_0%,_#FFF_50.64%,_#F6F7FE_100%)]">
      {" "}
      <div className="flex-1 self-stretch inline-flex flex-col justify-between items-start">
        <div className="flex flex-col gap-3">
          <div className="w-80 flex flex-col gap-3">
            <div className="inline-flex items-start gap-2">
              <div className="flex items-center gap-2">
                <NewAvatar title={title} />
                <div className="flex flex-col justify-center">
                  <div className="text-custom-customBlue text-base font-semibold break-words whitespace-pre-wrap">
                    {title}
                  </div>
                </div>
              </div>
            </div>
            <div className="w-80 text-[12px] font-500 px-1.5 break-words whitespace-pre-wrap">
              {description?.trim() ? (
                <span className="text-gray-600">{description}</span>
              ) : (
                <span className="text-[#ababab]">
                  (No description added yet)
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="w-full flex justify-between items-end pt-3 border-t-[1px] border-t-[#eeeeee]">
          <div className="flex items-center gap-1.5">
            <div
              className="px-2 py-1 bg-[#f6f7ff] border border-[#d7dbfc] rounded-[4px] flex items-center gap-1.5"
              title={allToolNames?.join(", ")}
            >
              <div className="w-3.5 h-3.5 relative overflow-hidden">
                <Image
                  src={getTagIcon(tag)}
                  className="w-full h-full object-contain"
                  alt="Tag Icon"
                  width={14}
                  height={14}
                />
              </div>
              <span className="text-zinc-800/90 text-xs font-medium truncate p-100">
                {(() => {
                  // Special handling for datasets to show file count prominently
                  if (
                    tag === "files" &&
                    allToolNames &&
                    allToolNames.length > 0
                  ) {
                    return allToolNames[0]; // Show the first item (file count) directly
                  }

                  const mainTag =
                    tag.length > 20 ? tag.slice(0, 20) + "..." : tag;
                  const extraCount = (allToolNames?.length || 0) - 1;
                  return extraCount > 0
                    ? `${mainTag} +${extraCount} more`
                    : mainTag;
                })()}
              </span>
            </div>

            {/* Conditional source type tag */}
            {sourceType !== undefined && (
              <div
                className="px-2 py-1 bg-[#f6f7ff] border border-[#d7dbfc] rounded-[4px] flex items-center gap-1.5"
                title={`Source Type: ${sourceType || constants.SOURCE_TYPE.LOCAL}`}
              >
                <div className="w-3.5 h-3.5 relative overflow-hidden">
                  <Image
                    src={getSourceTypeIcon(sourceType)}
                    className="w-full h-full object-contain"
                    alt="Source Type Icon"
                    width={14}
                    height={14}
                  />
                </div>
                <span className="text-gray-800/90 text-xs font-medium truncate">
                  {sourceType || constants.SOURCE_TYPE.LOCAL}
                </span>
              </div>
            )}
          </div>
          {renderActionButton()}
        </div>
      </div>
      {/* 3-dot menu */}
      <div className="absolute top-4 right-4">
        <DropdownMenu>
          <DropdownMenuTrigger
            className={`focus:outline-none ${!isEditable || tag?.toLowerCase() === "internal" ? "opacity-50 cursor-not-allowed" : ""}`}
            disabled={!isEditable || tag?.toLowerCase() === "internal"}
          >
            <Image
              src={dots}
              alt="options"
              className={`self-start ${isEditable && tag?.toLowerCase() !== "internal" ? "cursor-pointer" : "cursor-not-allowed"}`}
              width={20}
              height={20}
            />
          </DropdownMenuTrigger>
          {isEditable && tag?.toLowerCase() !== "internal" && (
            <DropdownMenuContent align="end" className="w-28">
              {onEdit && (
                <DropdownMenuItem
                  onClick={onEdit}
                  className="hover:!bg-blue-100"
                >
                  Edit
                </DropdownMenuItem>
              )}
              {onDelete && (
                <DropdownMenuItem
                  onClick={onDelete}
                  className="hover:!bg-blue-100"
                >
                  Delete
                </DropdownMenuItem>
              )}
            </DropdownMenuContent>
          )}
        </DropdownMenu>
      </div>
    </div>
  );
};

export default WorkspaceUtilCard;
