/* The provided code is a React component named `WorkflowDetails`. Here is an overview of what the code
is doing: */
"use client";
import { ArrowLeft } from "lucide-react";
import Image from "next/image";
import React, { useState, useEffect } from "react";
import dots from "@/assets/icons/dots-vertical.svg";
import bgimage from "@/assets/images/workspace/workspace-default.svg";
import arrow from "@/assets/icons/arrow-down-large.svg";
import { useParams, useRouter } from "next/navigation";
import ScheduleData from "@/components/workflow/scheduleData";
import {
  getWorkflowDetail,
  startWorkflow,
  stopWorkflow,
  workFlowRunHistory,
} from "@/api/workflows";
import { useUser } from "@/context_api/userContext";
import { showError, showSuccess } from "@/utils/toastUtils";
import Loader from "@/components/loader";

const WorkflowDetails = () => {
  const [dataFlowDetail, setDataFlowDetail] = useState(null);
  const [runHistory, setRunHistory] = useState([]);
  const [loader, setLoader] = useState(false);
  const { user } = useUser();
  const params = useParams();
  const data = {
    oId: user?.clientId,
    workFlowId: params?.workflowId,
  };

  // start and stop workflow
  const startFlow = async () => {
    try {
      const response = await startWorkflow(data);

      if (response.status === 200) {
        getWorkFlowDEtails();
        showSuccess(`${response?.data?.message}`);
      }
    } catch (e) {}
  };

  const stopFlow = async () => {
    try {
      const response = await stopWorkflow(data);

      if (response.status === 200) {
        getWorkFlowDEtails();
        showSuccess(`${response?.data?.message}`);
      }
    } catch (e) {}
  };

  const getWorkFlowDEtails = async () => {
    setLoader(true);

    try {
      const response = await getWorkflowDetail(data);

      if (response.status === 200) {
        setDataFlowDetail(response?.data?.data);
        setLoader(false);
      }
    } catch (e) {
      showError("Something went wrong");
      setLoader(false);
    }
  };

  // get run history

  const getWorkflowRunHistory = async () => {
    try {
      const response = await workFlowRunHistory(data);

      setRunHistory(response?.data?.data?.items);
    } catch (e) {}
  };
  useEffect(() => {
    if (user) {
      getWorkFlowDEtails();
      getWorkflowRunHistory();
    }
  }, [user]);

  const router = useRouter();

  return (
    <>
      {loader ? (
        <Loader />
      ) : (
        <div>
          <div className="flex justify-between px-8 py-4 bg-white relative h-fit w-full">
            <Image
              src={bgimage}
              alt="bg image"
              className="absolute top-0 right-0 h-full "
            />
            <div className="relative ">
              <div
                className="text-sm font-medium"
                onClick={() => router.push(`/workflows/?id=${user?.clientId}`)}
              >
                <ArrowLeft className="w-4 h-4 inline" /> Back to Workflow
              </div>
              <div className="font-medium text-2xl">
                {dataFlowDetail?.name || "N/A"}
              </div>
              <div className="flex items-center gap-1">
                <div
                  className={`h-2 w-2 rounded-full ${
                    dataFlowDetail?.is_active ? "bg-green-10" : "bg-red-700"
                  }`}
                ></div>
                {dataFlowDetail?.is_active ? "Currently running" : "Stopped"}
              </div>
            </div>
            <div className="border rounded-sm p-2 self-start z-20">
              <Image src={dots} alt="dots icon" />
            </div>
          </div>
          <div className="w-full grid grid-cols-2 items-stretch bg-gray-100 h-screen">
            <div className="flex flex-col items-center justify-center">
              <div className="bg-white flex flex-col gap-2 w-64 p-4">
                <div className="bg-gray-700 text-white rounded-md text-xs w-fit px-3">
                  Dataset
                </div>
                <div className="text-base font-semibold">
                  {dataFlowDetail?.dataset_name}
                </div>
                {/* <div className='font-medium text-sm'>2 Datasets</div> */}
              </div>
              <Image src={arrow} alt="arrow icon" />
              <div className="bg-white flex flex-col gap-2 w-64 p-4">
                <div className="bg-gray-700 text-white rounded-md text-xs w-fit px-3">
                  Destination
                </div>
                <div className="text-base font-semibold">
                  {dataFlowDetail?.destination_name}
                </div>
                {/* <div className='font-medium text-sm'>Databricks Vector Search</div> */}
              </div>
            </div>
            <div className="bg-white h-full">
              <ScheduleData
                apiData={dataFlowDetail}
                startFlow={startFlow}
                stopFlow={stopFlow}
                runHistory={runHistory}
              />
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default WorkflowDetails;
