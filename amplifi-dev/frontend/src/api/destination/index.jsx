import { http } from "..";

export const destinationCardData = async (organization_id) => {
  return await http.get(`/organization/${organization_id}/destination`);
};

export const createDestination = (organization_id, destinationData) => {
  return http.post(
    `/organization/${organization_id}/destination`,
    destinationData,
  );
};

export const getDestinationStatus = (organization_id, destination_id) => {
  return http.get(
    `/organization/${organization_id}/destination/${destination_id}/connection_status`,
  );
};

export const deleteDestination = async (organization_id, destination_id) => {
  const response = await http.delete(
    `/organization/${organization_id}/destination/${destination_id}`,
  );
  return response.data;
};
