import React from "react";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

const ViewDestinationForm = ({ data }) => {
  return (
    <>
      <div className="mx-4 mb-4 flex gap-4 flex-col text-sm font-medium">
        <div>
          <div>{data?.description}</div>
          <div>Total Records: {}</div>
        </div>
        <Table className="border">
          <TableHeader className="bg-gray-200 text-xs">
            <TableRow>
              <TableHead className="text-black-20 font-semibold">
                Active Workflows
              </TableHead>
              <TableHead className="text-black-20 font-semibold">
                Last updated
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody className="text-sm font-medium">
            <TableRow>
              <TableCell className="font-medium">Loan LLM workflow</TableCell>
              <TableCell> today, 9:00 AM</TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </div>
      <div className="border-t flex flex-end justify-end gap-4 p-4">
        <button
          type="submit"
          className="border px-4 py-2 rounded text-sm text-red-500"
        >
          Delete Connection
        </button>
        <button
          type="submit"
          className="bg-blue-10 text-white px-4 py-2 rounded text-sm"
        >
          Go to DB Client
        </button>
      </div>
    </>
  );
};

export default ViewDestinationForm;
