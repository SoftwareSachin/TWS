import React, { useEffect, useState } from "react";
import Select from "react-select"; // Import react-select
import { CirclePause, Pencil, Play } from "lucide-react";
import tickCircle from "@/assets/icons/tick-circle.svg";
import alert from "@/assets/icons/alert-triangle.svg";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import Image from "next/image";
import { reverseCronExpression } from "@/utils/convertCrown";
import { useRouter } from "next/navigation";
import { useUser } from "@/context_api/userContext";

const ScheduleData = ({ apiData, startFlow, stopFlow, runHistory }) => {
  const { user } = useUser();
  const router = useRouter();

  // Options for Trigger Time and Frequency dropdowns
  const triggerTimeOptions = [
    { value: "morning", label: "Morning" },
    { value: "afternoon", label: "Afternoon" },
    { value: "evening", label: "Evening" },
  ];

  const frequencyOptions = [
    { value: "daily", label: "Daily" },
    { value: "weekly", label: "Weekly" },
    { value: "monthly", label: "Monthly" },
  ];

  // State for storing selected values

  const [triggerTime, setTriggerTime] = useState(null);
  const [frequency, setFrequency] = useState(null);

  useEffect(() => {
    if (apiData?.schedule_config?.cron_expression) {
      setTriggerTime(
        reverseCronExpression(apiData?.schedule_config?.cron_expression)
          ?.triggerTime,
      );
      setFrequency(
        reverseCronExpression(apiData?.schedule_config?.cron_expression)
          ?.frequency,
      );
    }
  }, [apiData]);

  const data = [
    {
      schedule: "25 Aug, 2024  09:00 (Scheduled)",
      status: "Success",
    },
    {
      schedule: "25 Aug, 2024  09:00 (Scheduled)",
      status: "Success",
    },
    {
      schedule: "25 Aug, 2024  09:00 (Scheduled)",
      status: "Success",
    },
    {
      schedule: "25 Aug, 2024  09:00 (Scheduled)",
      status: "Failed",
    },
    {
      schedule: "25 Aug, 2024  09:00 (Scheduled)",
      status: "Failed",
    },
  ];

  // Function to get status badge classes
  const getStatusBadge = (status) => {
    switch (status) {
      case "Success":
        return "bg-green-100 text-green-800";
      default:
        return "bg-red-100 text-red-800"; // Default case for other statuses
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case "Success":
        return <Image src={tickCircle} alt="tick with circle" />;
      case "Failed":
        return <Image src={alert} alt="tick with circle" />;
      default:
        return null;
    }
  };

  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();
  };

  return (
    <div className="p-6">
      <div className="text-base font-semibold">Schedule</div>
      <form onSubmit={handleSubmit}>
        <div className="flex gap-4 w-full">
          <div className="w-1/2">
            <label className="text-sm font-medium">Trigger Time</label>
            <Select
              options={triggerTimeOptions}
              placeholder="Select Trigger Time"
              className="mt-1 focus:outline-none text-xs"
              value={triggerTime}
              onChange={(selectedOption) => setTriggerTime(selectedOption)}
              components={{ IndicatorSeparator: () => null }}
              isDisabled={true}
            />
          </div>
          <div className="w-1/2">
            <label className="text-sm font-medium">Frequency</label>
            <Select
              options={frequencyOptions}
              placeholder="Select Frequency"
              className="mt-1 text-xs focus:outline-none"
              value={frequency}
              onChange={(selectedOption) => setFrequency(selectedOption)}
              components={{ IndicatorSeparator: () => null }}
              isDisabled={true}
            />
          </div>
        </div>

        <div className="flex gap-3 mt-4">
          <button
            type="submit"
            className="bg-blue-10 text-white flex gap-2 items-center rounded px-3 py-2 text-sm font-medium"
            onClick={() => {
              apiData?.is_active ? stopFlow() : startFlow();
            }}
          >
            {apiData?.is_active ? (
              <CirclePause className="w-6 h-6 text-white" />
            ) : (
              <Play color="#ffffff" className="w-6 h-6" />
            )}

            {apiData?.is_active ? "Pause" : "Start"}
          </button>
          <button
            type="button"
            className="text-sm font-medium px-3 py-2 flex gap-2 items-center border rounded"
            onClick={() =>
              router.push(
                `/workflows?id=${user?.clientId}&dataFlow=${apiData?.id}`,
              )
            }
          >
            <Pencil className="w-4 h-4" />
            Edit
          </button>
        </div>
      </form>
      <div className="text-base font-semibold mt-8">Schedule</div>
      <Table className="border border-gray-300 rounded-2xl mt-2">
        <TableHeader>
          <TableRow className="border-b-2 border-gray-300">
            <TableHead className="text-xs font-semibold bg-gray-200 text-black-10 ps-4">
              Run Schedule
            </TableHead>
            <TableHead className="text-xs font-semibold bg-gray-200 text-black-10 ps-4">
              Status
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {runHistory?.length > 0 ? (
            runHistory.map((items) => (
              <TableRow
                className="border-b-2 border-gray-300 bg-white"
                key={items?.run_id}
              >
                <TableCell className="py-3 px-4 overflow-hidden whitespace-nowrap text-ellipsis text-sm">
                  {items?.created_at}
                </TableCell>
                <TableCell className="py-3 px-4 overflow-hidden whitespace-nowrap text-ellipsis text-sm">
                  <span
                    className={`w-fit gap-1 py-1 px-3 rounded-md text-xs flex  ${getStatusBadge(
                      items?.status,
                    )}`}
                  >
                    {getStatusIcon(items?.status)}
                    {items?.status}
                  </span>
                </TableCell>
              </TableRow>
            ))
          ) : (
            <TableRow>
              <TableCell colSpan={2} className="py-4 text-center text-gray-500">
                No history found.
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  );
};

export default ScheduleData;
