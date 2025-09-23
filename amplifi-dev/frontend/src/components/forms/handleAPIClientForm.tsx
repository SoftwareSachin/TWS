import React, { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { Input } from "../ui/input";
import { AddApiClientFormSchema } from "@/types/ApiClient";
import { Button } from "@/components/ui/button";
import { showError, showSuccess } from "@/utils/toastUtils";
import { AddApiClientFormValues, ApiClient } from "@/types/ApiClient";
import { addApiClient, updateApiClient } from "@/api/api-client";
import { AddApiClientFormProps } from "@/types/props/ApiClientProps";
import { useUser } from "@/context_api/userContext";

const HandleAPIClientForm: React.FC<AddApiClientFormProps> = ({
  onSave,
  setIsOpen,
  apiClientData,
}) => {
  const [loading, setLoading] = useState<boolean>(false);
  const { user } = useUser();

  const isEdit = !!apiClientData;

  const earliestExpiryDate = new Date();
  earliestExpiryDate.setDate(earliestExpiryDate.getDate() + 1);
  const minExpiryDate = earliestExpiryDate.toISOString().split("T")[0];

  const handleApiClient = async (apiClient: ApiClient) => {
    setLoading(true);
    try {
      const request = {
        name: apiClient.name,
        expires_at: apiClient.expires_at,
      };
      let response;
      const userClientId = user?.clientId || "";
      if (isEdit && apiClientData) {
        response = await updateApiClient(
          userClientId,
          apiClientData.id,
          request,
        );
      } else {
        response = await addApiClient(userClientId, request);
      }
      if (response.status === 200) {
        showSuccess(`${response?.data?.message}`);
        onSave(response?.data?.data?.client_secret);
      }
    } catch (error: any) {
      showError(`${error?.response?.data?.message}`);
      console.error("response-- in add/edit api client", error);
    } finally {
      setLoading(false);
    }
  };

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<AddApiClientFormValues>({
    resolver: zodResolver(AddApiClientFormSchema),
    defaultValues: {
      apiClient: {
        name: apiClientData?.name || "",
        expires_at: apiClientData?.expires_at
          ? apiClientData.expires_at.split("T")[0]
          : minExpiryDate,
      },
    },
  });

  const onSubmit = (data: AddApiClientFormValues) => {
    handleApiClient(data.apiClient);
  };

  return (
    <form className="space-y-4 pt-4" onSubmit={handleSubmit(onSubmit)}>
      <div className="grid grid-cols-2 gap-4 items-center px-4">
        {!isEdit && (
          <div>
            <label className="block text-sm font-medium">Name</label>
            <Input {...register(`apiClient.name`)} />
            <p className="text-xs text-red-600">
              {errors.apiClient?.name?.message}
            </p>
          </div>
        )}

        <div>
          <label className="block text-sm font-medium">Expires At</label>
          <Input
            type="date"
            {...register(`apiClient.expires_at`)}
            min={minExpiryDate}
          />
          <p className="text-xs text-red-600">
            {errors.apiClient?.expires_at?.message}
          </p>
        </div>
      </div>

      <div className="flex justify-end space-x-2 p-4 border-t">
        <button
          type="button"
          onClick={() => setIsOpen(false)}
          className="px-4 py-2 border rounded-md"
        >
          Cancel
        </button>
        <Button
          isLoading={loading}
          type="submit"
          className="px-4 py-2 bg-blue-600 text-white rounded-md"
        >
          {isEdit ? "Update API Client" : "Add API Client"}
        </Button>
      </div>
    </form>
  );
};

export default HandleAPIClientForm;
