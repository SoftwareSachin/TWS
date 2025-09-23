import { http } from "..";

export const UserListData = () => {
  return http.get(`/user/list`);
};

export const organisationUserListData = (orgID, pagination) => {
  return http.get(
    `/user/list?organization_id=${orgID}&page=${pagination.page}&size=${pagination.size}`,
  );
};

export const addUser = (payload) => {
  return http.post(`/user/invite-user`, payload);
};

export const deleteUser = (user_id) => {
  try {
    return http.delete(`/user/${user_id}`);
  } catch (error) {
    console.error("Error deleting user:", error);
    throw error;
  }
};

// Send link to create user
export const sendCreateUserLink = (payload) => {
  return http.post(`/user/invite-user`, payload);
};
