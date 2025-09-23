import { http } from "..";

export const addUsersInWorkspace = (data) => {
  return http.post(
    `/organization/${data?.orgId}/workspace/${data?.id}/add_users`,
    data.body,
  );
};

export const removeUsersFromWorkspace = (data) => {
  return http.post(
    `/organization/${data?.orgId}/workspace/${data?.id}/remove_users`,
    data.body,
  );
};

export const getUsersFromWorkspace = (data, pagination) => {
  return http.get(
    `/organization/${data?.orgId}/workspace/${data?.id}/get_users?page=${pagination.page}&size=${pagination.size}`,
  );
};

export const getWorkSpaceByID = (data) => {
  return http.get(`/organization/${data?.id}/workspace/${data?.workspaceId}`);
};
