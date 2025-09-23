import { http } from "..";

export const createOrganisation = (data) => {
  return http.post(`/organization`, data);
};
