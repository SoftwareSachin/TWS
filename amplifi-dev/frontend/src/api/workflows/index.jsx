import { http } from "..";

// create workflows
export const createWorkFlow = (data) => {
  return http.post(`/organization/${data.id}/workflow`, data.body);
};

// get workflow list
export const getWorkflow = (data) => {
  return http.get(
    `/organization/${data.id}/workflow?order=ascendent&page=${data.page}&size=${data.size}`,
  );
};

// GET WORKFLOW DETAILS
export const getWorkflowDetail = (data) => {
  return http.get(`organization/${data.oId}/workflow/${data.workFlowId}`);
};

// start workflow
export const startWorkflow = (data) => {
  return http.post(
    `/organization/${data.oId}/workflow/${data.workFlowId}/start`,
  );

  return response;
};

// stop workflow
export const stopWorkflow = (data) => {
  return http.post(
    `/organization/${data.oId}/workflow/${data.workFlowId}/stop`,
  );

  return response;
};

// Edit workflow
export const editWorkFlow = (data) => {
  return http.put(
    `/organization/${data.oId}/workflow/${data.workFlowId}`,
    data.body,
  );
};

// execute workflow
export const executeWorkFlow = (data) => {
  return http.post(
    `/organization/${data.oId}/workflow/${data.workFlowId}/execute`,
    {
      run_config: {},
    },
  );
};

// get run history

export const workFlowRunHistory = (data) => {
  return http.get(`/organization/${data.oId}/workflow/${data.workFlowId}/runs`);
};
