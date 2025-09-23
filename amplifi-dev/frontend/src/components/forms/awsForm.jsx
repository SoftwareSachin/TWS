import { Input } from "@/components/ui/input";
import { useRouter } from "next/navigation";
import React, { useState, useEffect } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { useSearchParams } from "next/navigation";

import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Button } from "@/components/ui/button";
import { sourceConnector, testConnection } from "@/api/Workspace/workspace";
import { showError, showSuccess } from "@/utils/toastUtils";
import { useUser } from "@/context_api/userContext";
import { editSource, getSourceDetails } from "@/api/Workspace/WorkSpaceFiles";
import { constants } from "@/lib/constants";

const AwsForm = ({ workSpaceId, type, sId }) => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const search = searchParams.get("id");
  const [sourceId, setSourceId] = useState(null);
  const [loading, setLoading] = useState(false);
  const { user } = useUser();

  const getAzureDetail = async () => {
    const data = {
      workspaceId: workSpaceId,
      sourceId: sId,
    };
    try {
      const response = await getSourceDetails(data);

      if (response.status === 200) {
        // Removed sensitive AWS response logging for security
      }
    } catch (e) {}
  };

  useEffect(() => {
    if (sId) {
      // getAzureDetail()
    }
  }, [sId]);

  // Define validation schema using Zod
  const formSchema = z.object({
    sourceType: z.string().min(1, "Source type is required"),
    containerName: z.string().min(1, "Container name is required"),
    sasUrl: z.string().min(1, "SAS URL is required"),
  });

  // Initialize react-hook-form with the schema
  const form = useForm({
    resolver: zodResolver(formSchema),
    defaultValues: {
      sourceType: type === constants.SOURCE.AZURE ? "azure_storage" : "",
      containerName: "",
      sasUrl: "",
    },
  });

  const getConnectionStatus = async () => {
    setLoading(true);
    const data = {
      workSpaceId,
      sourceId: sourceId,
    };
    if (sourceId) {
      try {
        const response = await testConnection(data);

        if ((response.status = 200)) {
          showSuccess(`${response?.data?.data?.message}`);
          router.push(`/workspace/${workSpaceId}/files/0`);
        }
      } catch (error) {
        setSourceId(null);
        showError(`${error.response.data.detail}`);
      } finally {
        setLoading(false);
      }
    }
  };
  const onSubmit = async (values) => {
    const payload = {
      source_type: values.sourceType,
      container_name: values.containerName,
      sas_url: btoa(values.sasUrl),
    };
    const data = {
      id: workSpaceId,
      body: payload,
      ...(sId && { sourceId: sId }),
    };
    try {
      const response = sId
        ? await editSource(workSpaceId, sId, payload)
        : await sourceConnector(data);

      if (response.status === 200) {
        showSuccess(`${response?.data?.message}`);
        setSourceId(response?.data?.data?.id);
      }
    } catch (error) {
      showError(`${error.response?.data?.detail}`);
    }
  };

  // Handler for form submission
  const handleSubmit = (e) => {
    e.preventDefault(); // Prevent the default form submission
    // You can add validation or additional logic here before routing
    router.push(`/workspace/?id=${user?.clientId}`); // Navigate to the workspace page
  };

  return (
    <>
      {type === constants.SOURCE.AWS ? (
        <>
          <form className="space-y-4" onSubmit={handleSubmit}>
            <div className="flex gap-4 px-4">
              <div className="w-1/2">
                <label
                  htmlFor="label"
                  className="block mb-2 text-sm font-medium text-gray-900"
                >
                  Label
                </label>
                <Input
                  type="text"
                  id="label"
                  placeholder="Input text"
                  required
                />
              </div>
              <div className="w-1/2">
                <label
                  htmlFor="region"
                  className="block mb-2 text-sm font-medium text-gray-900"
                >
                  Region
                </label>
                <Input
                  type="text"
                  id="region"
                  placeholder="US East (Ohio)"
                  required
                />
              </div>
            </div>

            <div className="px-4">
              <label
                htmlFor="s3-access-key"
                className="block mb-2 text-sm font-medium text-gray-900"
              >
                S3 Access Key
              </label>
              <Input
                type="text"
                id="s3-access-key"
                placeholder="Key here"
                required
              />
            </div>
            <div className="px-4">
              <label
                htmlFor="s3-secret-key"
                className="block mb-2 text-sm font-medium text-gray-900"
              >
                S3 Secret Key
              </label>
              <Input
                type="text"
                id="s3-secret-key"
                placeholder="Secret key here"
                required
              />
            </div>

            <div className="border-t flex justify-end gap-4 p-4">
              <button
                type="submit"
                className="border px-4 py-2 rounded text-sm"
              >
                Test Connection
              </button>
              <button
                type="submit"
                className="bg-blue-500 text-white px-4 py-2 rounded text-sm"
              >
                Next
              </button>
            </div>
          </form>
        </>
      ) : (
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <div className="w-1/2 px-4">
              <FormField
                control={form.control}
                name="sourceType"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Source Type</FormLabel>
                    <FormControl>
                      <Input
                        placeholder="Enter source type"
                        {...field}
                        disabled={true}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>
            <div className="w-1/2 px-4">
              <FormField
                control={form.control}
                name="containerName"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Container Name</FormLabel>
                    <FormControl>
                      <Input
                        placeholder="Enter container name"
                        {...field}
                        disabled={!!sourceId}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>
            <div className="w-1/2 px-4">
              <FormField
                control={form.control}
                name="sasUrl"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>SAS URL</FormLabel>
                    <FormControl>
                      <Input
                        placeholder="Enter SAS URL"
                        {...field}
                        disabled={!!sourceId}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>
            <div className="border-t flex justify-end gap-4 p-4">
              {sourceId ? (
                <Button
                  type="button"
                  onClick={getConnectionStatus}
                  isLoading={loading}
                >
                  Test Connection
                </Button>
              ) : (
                <Button type="submit">Next</Button>
              )}
            </div>
          </form>
        </Form>
      )}
    </>
  );
};

export default AwsForm;
