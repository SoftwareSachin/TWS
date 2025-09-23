import { httpV2 } from "..";

export const search = (data) => {
  return httpV2.post(`/workspace/${data?.id}/search`, data.body);
};
