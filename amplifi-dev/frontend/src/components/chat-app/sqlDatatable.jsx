import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import React, { useEffect, useState } from "react";
import { downloadFileApi } from "@/api/chatApp";
import { showError, showSuccess } from "@/utils/toastUtils";

const SqlDatatableComponent = ({ tableData, csvFileId, csvFileName }) => {
  const [selectedTableIndex, setSelectedTableIndex] = useState(0);
  const [isDownloading, setIsDownloading] = useState(false);

  useEffect(() => {
    console.log("tableData:", tableData);
  }, [tableData]);

  const downloadCsvFile = async () => {
    if (!csvFileId) return;

    setIsDownloading(true);
    try {
      const fileObj = {
        file_id: csvFileId,
        file_name: csvFileName || "context_file",
      };

      const success = await downloadFileApi(fileObj);

      if (success) {
        showSuccess("CSV file downloaded successfully!");
      } else {
        showError("Failed to download CSV file. Please try again.");
      }
    } catch (error) {
      console.error("Error downloading CSV:", error);
      showError(
        "An error occurred while downloading the file. Please try again.",
      );
    } finally {
      setIsDownloading(false);
    }
  };

  const isMultipleTables = Array.isArray(tableData[0]);
  const tables = isMultipleTables ? tableData : [tableData];

  const selectedTable = tables[selectedTableIndex] || [];
  const headers = selectedTable.length ? Object.keys(selectedTable[0]) : [];

  return (
    <>
      {csvFileId && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-3 mt-2">
          <div className="flex items-center justify-between">
            <div className="flex items-start">
              <svg
                className="w-5 h-5 text-blue-500 mt-0.5 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                ></path>
              </svg>
              <div>
                {csvFileId && (
                  <p className="text-sm text-blue-600">
                    Click to download the complete dataset in CSV format.
                  </p>
                )}
              </div>
            </div>
            {csvFileId && (
              <Button
                onClick={downloadCsvFile}
                disabled={isDownloading}
                className="inline-flex items-center px-3 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isDownloading ? (
                  <>
                    <svg
                      className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      ></circle>
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      ></path>
                    </svg>
                    Downloading...
                  </>
                ) : (
                  <>
                    <svg
                      className="w-4 h-4 mr-2"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                      ></path>
                    </svg>
                    Download CSV
                  </>
                )}
              </Button>
            )}
          </div>
        </div>
      )}

      {tables.length > 1 && (
        <div className="flex gap-2 mb-2 mt-2 ">
          {tables.map((_, idx) => (
            <button
              key={idx}
              onClick={() => setSelectedTableIndex(idx)}
              className={`cursor-pointer inline-flex items-center px-[16px] py-[12px] text-sm font-medium text-gray-700 bg-custom-contextBgColor border border-[#E2E8F0] rounded-[16px] whitespace-nowrap max-w-full overflow-hidden
                  hover:bg-custom-contextHoverButtonColor ${
                    selectedTableIndex === idx
                      ? "!bg-custom-contextHoverButtonColor"
                      : ""
                  }`}
            >
              Table {idx + 1}
            </button>
          ))}
        </div>
      )}

      {selectedTable.length > 0 ? (
        <div className="container mt-2 ">
          <div className="border mr-auto border-gray-200 w-fit max-h-[250px] overflow-auto">
            <Table className="border relative border-gray-300 rounded-2xl mh-[70vh]">
              <TableHeader className="sticky top-0">
                <TableRow className="border-b-2 border-gray-300">
                  {headers.map((header) => (
                    <TableHead
                      key={header}
                      className="text-xs font-semibold bg-gray-200 text-black ps-4"
                    >
                      {header}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {selectedTable.map((row, rowIndex) => (
                  <TableRow
                    key={rowIndex}
                    className="border-b-2 border-gray-300 bg-white"
                  >
                    {headers.map((header) => (
                      <TableCell key={header} className="py-3 px-4">
                        {row[header]}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </div>
      ) : (
        <div>No data available.</div>
      )}
    </>
  );
};

export default SqlDatatableComponent;
