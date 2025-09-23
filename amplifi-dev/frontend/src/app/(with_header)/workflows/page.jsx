/* The above code is a React component that manages workflows. Here is a summary of what the code is
doing: */
"use client";
import React, { useState, useEffect } from "react";
import NoDataScreen from "@/components/empty-screens/noData";
import { Button } from "@/components/ui/button";
import Image from "next/image";
import emptyscreen from "@/assets/images/empty-screens/workflow-empty-image.svg";
import searchIcon from "@/assets/icons/search-icon.svg";
import DetailCardWorkflow from "@/components/workflow/detailCardWorkflow";
import WorkflowForm from "@/components/forms/workflowForm";
import WorkflowModal from "@/components/forms/workflowModal";
import { useRouter, useSearchParams } from "next/navigation";
import {
  createWorkFlow,
  editWorkFlow,
  executeWorkFlow,
  getWorkflow,
  getWorkflowDetail,
} from "@/api/workflows";
import { showError, showSuccess } from "@/utils/toastUtils";
import { decodeToken } from "@/components/utility/decodeJwtToken";
import { getCookie } from "@/utils/cookieHelper";
import { constants } from "@/lib/constants";

const Page = () => {
  const [workflowModal, setWorkflowModal] = useState(false);
  const [editableTitle, setEditableTitle] = useState(false);
  const [titleText, setTitleText] = useState("Untitled workflow 01");
  const [cronExpression, setCronExpression] = useState("");
  const [selectedValue, setSelectedValue] = useState(null);
  const [selectedDestinationValue, setSelectedDestinationValue] =
    useState(null);
  const [workflowList, setWorkflowList] = useState([]);
  const [totalWorkflow, setTotalWorkflow] = useState(0);
  const [dataFlowDetail, setDataFlowDetail] = useState(null);
  const [loader, setLoader] = useState(true);
  const [totalPages, setTotalPages] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const searchParams = useSearchParams();
  let search = searchParams.get("id");
  const dataFlowId = searchParams.get("dataFlow");
  const route = useRouter();
  if (!search) {
    const token = getCookie(constants.JWT_TOKEN);
    const userDetails = decodeToken(token);
    search = userDetails.clientId;
  }

  // other hooks
  useEffect(() => {
    getWorkflowList();
  }, [currentPage]);

  useEffect(() => {
    if (dataFlowId) {
      setWorkflowModal(true);
      getWorkFlowDEtails();
    }
  }, [dataFlowId]);

  useEffect(() => {
    if (dataFlowDetail) {
      setTitleText(dataFlowDetail?.name);
      setSelectedValue({
        id: dataFlowDetail?.dataset_id,
        name: `${dataFlowDetail?.dataset_name}`,
      });
      setSelectedDestinationValue({
        id: dataFlowDetail?.destination_id,
        name: `${dataFlowDetail?.destination_name}`,
      });
      setCronExpression(dataFlowDetail?.schedule_config?.cron_expression);
    }
  }, [dataFlowDetail]);

  // getting old data ..
  const getWorkFlowDEtails = async () => {
    const data = {
      oId: search,
      workFlowId: dataFlowId,
    };
    try {
      const response = await getWorkflowDetail(data);
      if (response.status === 200) {
        setDataFlowDetail(response?.data?.data);
      }
    } catch (e) {}
  };

  // execute the workflow ..
  const executeWorkflow = async (workflowId) => {
    const data = {
      oId: search,
      workFlowId: workflowId,
    };
    try {
      const response = await executeWorkFlow(data);

      if (response.status === 200) {
        showSuccess(`${response?.data?.message}`);
        if (dataFlowId) {
          route.push(`workflows/${dataFlowId}`);
        }
        setWorkflowModal(false);
        getWorkflowList();
      }
    } catch (error) {
      setWorkflowModal(false);
      showError(`${error?.response?.data?.detail}`);
    }
  };
  // Submit handler for creating a workflow

  const handleSubmit = async () => {
    const payload = {
      ...(titleText && { name: titleText }),
      // ...(titleText && {description :titleText}),
      is_active: true,
      ...(selectedDestinationValue && {
        destination_id: selectedDestinationValue.id,
      }),
      ...(selectedValue && { dataset_id: selectedValue.id }),
      ...(titleText && {
        schedule_config: {
          cron_expression: cronExpression,
        },
      }),
    };

    const data = {
      id: search,
      body: payload,
    };

    const editData = {
      body: payload,
      oId: search,
      workFlowId: dataFlowId,
    };
    try {
      const response = dataFlowId
        ? await editWorkFlow(editData)
        : await createWorkFlow(data);

      if (response.status === 200) {
        executeWorkflow(response?.data?.data?.id);
      }
    } catch (e) {
      showError(`${error?.response?.data?.detail}`);
    }
  };

  // get workflow details
  const getWorkflowList = async () => {
    setLoader(true);
    try {
      const data = {
        id: search,
        page: currentPage,
        size: 12,
      };

      const response = await getWorkflow(data);
      if (response.status === 200) {
        setLoader(false);
        setWorkflowList(response?.data?.data?.items);
        setTotalWorkflow(response?.data?.data?.total);
        setTotalPages(response?.data?.data?.pages);
      }
    } catch (e) {
      setLoader(false);
    }
  };

  const handlePageChange = (page) => {
    if (page > 0 && page <= totalPages) {
      setCurrentPage(page);
    }
  };

  return (
    <>
      <div className="m-8">
        <div className="flex justify-between">
          <div className="font-medium text-2xl">
            Workflows
            <span className="bg-gray-300 rounded-3xl font-normal text-sm px-2 py-1 ms-2">
              {totalWorkflow < 10 ? `0${totalWorkflow}` : totalWorkflow}
            </span>
          </div>
          {workflowList?.length > 0 && (
            <div className="rounded-lg flex gap-2">
              <span className="bg-white px-2 flex items-center gap-2 rounded-lg border border-gray-400">
                <Image src={searchIcon} alt="search icon" />
                <input
                  placeholder="Search here"
                  className="bg-white outline-none"
                />
              </span>
              <Button
                className="bg-blue-10"
                onClick={() => setWorkflowModal(true)}
              >
                + Create Workflow
              </Button>
            </div>
          )}
        </div>

        {loader ? (
          <div className="grid grid-cols-4 w-full gap-4 mt-4">
            {Array.from({ length: 8 }).map((_, index) => (
              <DetailCardWorkflow key={index} loading={true} />
            ))}
          </div>
        ) : workflowList?.length > 0 ? (
          <>
            <div className="grid grid-cols-4 w-full gap-4 mt-4">
              {workflowList?.map((item, index) => (
                <DetailCardWorkflow key={index} item={item} index={index} />
              ))}
            </div>
            {totalPages > 1 && (
              <div className="flex justify-center items-right gap-2 mt-6">
                <button
                  onClick={() => handlePageChange(currentPage - 1)}
                  className={`w-10 h-10 flex items-center justify-center rounded-full border border-gray-300 text-gray-500 hover:bg-gray-200 transition-all ${
                    currentPage === 1 ? "opacity-50 cursor-not-allowed" : ""
                  }`}
                  disabled={currentPage === 1}
                >
                  {"<"}
                </button>
                {Array.from({ length: totalPages }).map((_, index) => {
                  const page = index + 1;
                  return (
                    <button
                      key={page}
                      onClick={() => handlePageChange(page)}
                      className={`w-10 h-10 flex items-center justify-center rounded-full text-gray-700 border border-gray-300 hover:bg-blue-100 transition-all ${
                        currentPage === page ? "bg-blue-500 text-white" : ""
                      }`}
                    >
                      {page}
                    </button>
                  );
                })}
                <button
                  onClick={() => handlePageChange(currentPage + 1)}
                  className={`w-10 h-10 flex items-center justify-center rounded-full border border-gray-300 text-gray-500 hover:bg-gray-200 transition-all ${
                    currentPage === totalPages
                      ? "opacity-50 cursor-not-allowed"
                      : ""
                  }`}
                  disabled={currentPage === totalPages}
                >
                  {">"}
                </button>
              </div>
            )}
          </>
        ) : (
          <NoDataScreen
            title="No Workflows Created"
            subtitle="You haven’t set up any workflows yet. Start by building your first workflow to streamline your process!"
            buttonText="Create Workflow"
            image={emptyscreen}
            onClick={() => setWorkflowModal(true)}
          />
        )}
      </div>
      <WorkflowModal
        isOpen={workflowModal}
        onClose={() => setWorkflowModal(false)}
        title="Create Workflow"
        size="true"
        type="workflow"
        editableTitle={editableTitle}
        setEditableTitle={setEditableTitle}
        titleText={titleText}
        setTitleText={setTitleText}
        onSubmit={handleSubmit}
        dataFlowId={dataFlowId}
      >
        <WorkflowForm
          setCronExpression={setCronExpression}
          cronExpression={cronExpression}
          selectedValue={selectedValue}
          setSelectedValue={setSelectedValue}
          selectedDestinationValue={selectedDestinationValue}
          setSelectedDestinationValue={setSelectedDestinationValue}
        />
      </WorkflowModal>
    </>
  );
};

export default Page;
