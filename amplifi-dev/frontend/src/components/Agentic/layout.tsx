"use client";

import React, { ChangeEvent } from "react";
import Image from "next/image";
import searchIcon from "@/assets/icons/search.svg";
import Paginator from "@/components/utility/paginator";
import { WorkspacePageWrapperProps } from "@/types/props/WorkspacePageWrapperProps";
import { Input } from "../ui/input";
import { Button } from "../ui/button";

const customButtonText = (title: string) => {
  switch (title.toLowerCase()) {
    case "mcps":
      return "Add External MCP";
    case "agents":
      return "Create New Agent";
    case "tools":
      return "Add Tool to Workspace";
    case "chats":
      return "Create New Chat";
    default:
      return `Create New ${title}`;
  }
};

const WorkspacePageWrapper: React.FC<WorkspacePageWrapperProps> = ({
  title,
  itemCount,
  searchTerm,
  onSearchChange,
  onCreateClick,
  renderItems,
  loading,
  error,
  CreateModal,
  DeleteModal,
  pagination,
  totalPages,
  onPaginationChange,
}) => {
  return (
    <div className="p-6">
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-semibold">{title}</h1>
            <div className="w-8 h-[26px] bg-[#babfc6] text-slate-900/90 text-sm font-normal flex items-center justify-center border-[2px] border-white rounded-[13px] px-2.5 py-1.5">
              {itemCount}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <div className="h-8 px-4 py-px bg-white rounded outline outline-1 outline-offset-[-1px] outline-zinc-800/40 inline-flex items-center gap-2">
              <div className="w-4 h-4 relative overflow-hidden">
                <Image
                  src={searchIcon}
                  alt="Search icon"
                  fill
                  className="object-contain"
                />
              </div>
              <Input
                type="text"
                placeholder="Search here"
                className="flex-1 bg-transparent text-sm text-zinc-800/40 placeholder:text-zinc-800/40 focus:outline-none border-none shadow-none"
                value={searchTerm}
                onChange={(e: ChangeEvent<HTMLInputElement>) =>
                  onSearchChange(e.target.value)
                }
              />
            </div>

            <Button
              className="text-[14px] h-8 bg-indigo-600 text-white px-3 py-[5px] rounded-[4px]"
              onClick={onCreateClick}
            >
              {customButtonText(title)}
            </Button>
          </div>
        </div>

        <div className="grid gap-4 grid-cols-[repeat(auto-fill,minmax(384px,1fr))]">
          {loading ? (
            <p>Loading...</p>
          ) : error ? (
            <p className="text-red-500">{error}</p>
          ) : itemCount === 0 ? (
            <p>No items found.</p>
          ) : (
            renderItems()
          )}
        </div>
      </div>

      {CreateModal}
      {DeleteModal}

      {itemCount > 0 && pagination && totalPages && onPaginationChange && (
        <Paginator
          page={pagination}
          size={"full"}
          totalPages={totalPages}
          showPageSize={true}
          onChange={onPaginationChange}
        />
      )}
    </div>
  );
};

export default WorkspacePageWrapper;
