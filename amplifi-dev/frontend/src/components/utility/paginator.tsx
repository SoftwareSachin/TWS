import React from "react";
import {
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
} from "lucide-react";
import { Page } from "@/types/Paginated";

interface PaginatorProps {
  totalPages: number;
  page: Page;
  onChange: (options: { page: number; size: number }) => void;
  size?: "full" | "small";
  className?: string;
  showPageSize?: boolean;
}

const pageSizeOptions = [10, 25, 50];

const Paginator: React.FC<PaginatorProps> = ({
  page,
  totalPages,
  onChange,
  size = "full",
  showPageSize = "true",
  className = "",
}) => {
  const changePage = (pageNum: number) => {
    if (pageNum >= 1 && pageNum <= totalPages && pageNum !== page.page) {
      onChange({ page: pageNum, size: page.size });
    }
  };

  const changePageSize = (newSize: number) => {
    onChange({ page: 1, size: newSize });
  };

  const visiblePages = () => {
    const pages: number[] = [];
    const maxButtons = 4;

    let start = Math.max(1, page.page - Math.floor(maxButtons / 2));
    let end = Math.min(totalPages, start + maxButtons - 1);

    if (end - start < maxButtons - 1) {
      start = Math.max(1, end - maxButtons + 1);
    }

    for (let i = start; i <= end; i++) {
      pages.push(i);
    }

    return pages;
  };

  return (
    <div
      className={`flex flex-col md:flex-row md:items-center md:justify-center gap-4 mt-4 ${className}`}
    >
      {/* Page size dropdown */}
      {showPageSize && (
        <div className="flex items-center gap-2">
          <label className="text-[0.8rem] font-medium">Items per page:</label>
          <select
            value={page.size}
            onChange={(e) => changePageSize(Number(e.target.value))}
            className="border border-gray-300 rounded px-2 py-1 text-sm"
          >
            {pageSizeOptions.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Pagination Controls */}
      <div className="flex items-center gap-2">
        {size === "full" && (
          <button
            title={"First Page"}
            onClick={() => changePage(1)}
            disabled={page.page === 1}
            className="p-2 rounded hover:bg-gray-100 disabled:opacity-50"
          >
            <ChevronsLeft size={18} />
          </button>
        )}

        <button
          title={"Previous Page"}
          onClick={() => changePage(page.page - 1)}
          disabled={page.page === 1}
          className="p-2 rounded hover:bg-gray-100 disabled:opacity-50"
        >
          <ChevronLeft size={18} />
        </button>

        {size === "full" &&
          visiblePages().map((pageNum) => (
            <button
              key={pageNum}
              onClick={() => changePage(pageNum)}
              className={`px-3 py-1 text-sm rounded ${
                page.page === pageNum
                  ? "bg-blue-600 text-white"
                  : "bg-gray-100 hover:bg-gray-200"
              }`}
            >
              {pageNum}
            </button>
          ))}

        <button
          onClick={() => changePage(page.page + 1)}
          disabled={page.page === totalPages}
          className="p-2 rounded hover:bg-gray-100 disabled:opacity-50"
          title={"Next Page"}
        >
          <ChevronRight size={18} />
        </button>

        {size === "full" && (
          <button
            onClick={() => changePage(totalPages)}
            disabled={page.page === totalPages}
            className="p-2 rounded hover:bg-gray-100 disabled:opacity-50"
            title={"Last Page"}
          >
            <ChevronsRight size={18} />
          </button>
        )}
      </div>
    </div>
  );
};

export default Paginator;
