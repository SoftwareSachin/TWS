import { Input } from "@/components/ui/input";
import { useRouter } from "next/navigation";
import React, { useState, useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";

import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
  FormDescription,
} from "@/components/ui/form";
import { Button } from "@/components/ui/button";
import { sourceConnector, testConnection } from "@/api/Workspace/workspace";
import { showError, showSuccess } from "@/utils/toastUtils";
import { editSource, getSourceDetails } from "@/api/Workspace/WorkSpaceFiles";
import { constants } from "@/lib/constants";
import {
  GrooveFormSchema,
  GrooveFormData,
} from "@/types/GrooveSourceConnector";
import { GrooveSourceConnectorProps } from "@/types/props/GrooveSourceConnectorProps";

const GrooveForm: React.FC<GrooveSourceConnectorProps> = ({
  workSpaceId,
  type,
  sId,
}) => {
  const router = useRouter();
  const [sourceId, setSourceId] = useState(null);
  const [loading, setLoading] = useState(false);

  const getGrooveDetails = async () => {
    const data = {
      workspaceId: workSpaceId,
      sourceId: sId,
    };
    try {
      const response = await getSourceDetails(data);
      if (response.status === 200) {
        // Handle existing source details if needed
        const sourceData = response.data.data;
        if (sourceData) {
          form.reset({
            sourceType: sourceData.source_type || "groove_source",
            sourceName: sourceData.source_name || "",
            grooveApiKey: "", // Don't populate API key for security
            autoDetectionEnabled: sourceData.auto_detection_enabled || false,
            monitoringFrequencyMinutes:
              sourceData.monitoring_frequency_minutes || 30,
            ticketBatchSize: sourceData.ticket_batch_size || 10,
            reIngestUpdatedTickets:
              sourceData.re_ingest_updated_tickets || false,
          });
        }
      }
    } catch (e) {
      console.error("Error fetching Groove details:", e);
    }
  };

  useEffect(() => {
    if (sId) {
      getGrooveDetails();
    }
  }, [sId]);

  // Initialize react-hook-form with the schema
  const form = useForm<GrooveFormData>({
    resolver: zodResolver(GrooveFormSchema),
    defaultValues: {
      sourceType: type === constants.SOURCE.GROOVE ? "groove_source" : "",
      sourceName: "",
      grooveApiKey: "",
      autoDetectionEnabled: false,
      monitoringFrequencyMinutes: 30,
      ticketBatchSize: 10,
      reIngestUpdatedTickets: false,
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
        if (response.status === 200) {
          showSuccess(`${response?.data?.data?.message}`);
          router.push(`/workspace/${workSpaceId}/files/0`);
        }
      } catch (error: any) {
        setSourceId(null);
        showError(`${error.response?.data?.detail}`);
      } finally {
        setLoading(false);
      }
    }
  };

  const onSubmit = async (values: GrooveFormData) => {
    setLoading(true);
    const payload = {
      source_type: values.sourceType,
      source_name: values.sourceName,
      groove_api_key: btoa(values.grooveApiKey), // Base64 encode the API key
      // Auto-detection configuration
      auto_detection_enabled: values.autoDetectionEnabled,
      monitoring_frequency_minutes: values.monitoringFrequencyMinutes,
      ticket_batch_size: values.ticketBatchSize,
      re_ingest_updated_tickets: values.reIngestUpdatedTickets,
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
    } catch (error: any) {
      showError(`${error.response?.data?.detail}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Form {...form}>
      <form
        onSubmit={form.handleSubmit(onSubmit)}
        className="space-y-4 overflow-auto h-full max-h-[80vh]  flex flex-wrap"
        style={{ minHeight: "400px" }}
      >
        <div className="w-full px-4">
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
                    value={
                      typeof field.value === "boolean"
                        ? String(field.value)
                        : (field.value ?? "")
                    }
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
            name="sourceName"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Source Name</FormLabel>
                <FormControl>
                  <Input
                    placeholder="Enter source name (e.g., My Groove Support)"
                    {...field}
                    value={
                      typeof field.value === "boolean"
                        ? String(field.value)
                        : (field.value ?? "")
                    }
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
            name="grooveApiKey"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Groove API Key</FormLabel>
                <FormControl>
                  <Input
                    type="password"
                    placeholder="Enter your Groove API key"
                    {...field}
                    value={
                      typeof field.value === "boolean"
                        ? String(field.value)
                        : (field.value ?? "")
                    }
                    disabled={!!sourceId}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
        </div>

        {/* Auto-Detection Configuration Section */}
        <div className="border-t pt-6 w-full flex flex-wrap">
          <h3 className="text-lg font-semibold mb-4 px-4">
            Auto-Detection Configuration
          </h3>

          <div className="w-full px-4 mb-4">
            <FormField
              control={form.control}
              name="autoDetectionEnabled"
              render={({ field }) => (
                <FormItem className="flex flex-row items-start space-x-3 space-y-0">
                  <FormControl>
                    <input
                      type="checkbox"
                      id="auto_detection"
                      checked={field.value ? true : false}
                      onChange={field.onChange}
                      className="h-4 w-4 rounded border border-gray-300 bg-white checked:bg-blue-500 checked:border-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
                    />
                  </FormControl>
                  <div className="space-y-1 leading-none">
                    <FormLabel>Enable Auto-Detection</FormLabel>
                    <FormDescription>
                      Automatically monitor for new tickets and process them
                    </FormDescription>
                  </div>
                </FormItem>
              )}
            />
          </div>

          {form.watch("autoDetectionEnabled") && (
            <>
              <div className="w-1/2 px-4 mb-4">
                <FormField
                  control={form.control}
                  name="monitoringFrequencyMinutes"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Monitoring Frequency (minutes)</FormLabel>
                      <FormControl>
                        <Input
                          type="number"
                          placeholder="30"
                          {...field}
                          value={
                            typeof field.value === "boolean"
                              ? String(field.value)
                              : (field.value ?? "")
                          }
                          onChange={(e) => {
                            const val = e.target.value;
                            field.onChange(val === "" ? "" : Number(val));
                          }}
                        />
                      </FormControl>
                      <FormDescription>
                        How often to check for new tickets (1-1440 minutes)
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>

              <div className="w-1/2 px-4 mb-4">
                <FormField
                  control={form.control}
                  name="ticketBatchSize"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Ticket Batch Size</FormLabel>
                      <FormControl>
                        <Input
                          type="number"
                          placeholder="10"
                          {...field}
                          value={
                            typeof field.value === "boolean"
                              ? ""
                              : (field.value?.toString() ?? "")
                          }
                          onChange={(e) =>
                            field.onChange(parseInt(e.target.value) || 10)
                          }
                        />
                      </FormControl>
                      <FormDescription>
                        Number of tickets to process in each batch (1-100)
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>

              <div className="w-1/2 px-4 mb-4">
                <FormField
                  control={form.control}
                  name="reIngestUpdatedTickets"
                  render={({ field }) => (
                    <FormItem className="flex flex-row items-start space-x-3 space-y-0">
                      <FormControl>
                        <input
                          type="checkbox"
                          id="re_ingest_updated_tickets"
                          checked={field.value ? true : false}
                          onChange={field.onChange}
                          className="h-4 w-4 rounded border border-gray-300 bg-white checked:bg-blue-500 checked:border-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
                        />
                      </FormControl>
                      <div className="space-y-1 leading-none">
                        <FormLabel>Re-ingest Updated Tickets</FormLabel>
                        <FormDescription>
                          Automatically re-process tickets that have been
                          updated
                        </FormDescription>
                      </div>
                    </FormItem>
                  )}
                />
              </div>
            </>
          )}
        </div>

        <div className="border-t flex justify-end gap-4 p-4 w-full">
          {sourceId ? (
            <Button
              type="button"
              onClick={getConnectionStatus}
              isLoading={loading}
            >
              Test Connection
            </Button>
          ) : (
            <Button type="submit" isLoading={loading}>
              Next
            </Button>
          )}
        </div>
      </form>
    </Form>
  );
};

export default GrooveForm;
