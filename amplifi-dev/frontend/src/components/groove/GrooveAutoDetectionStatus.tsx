import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { showError, showSuccess } from "@/utils/toastUtils";
import {
  getSourceDetails,
  configureGrooveAutoDetection,
} from "@/api/Workspace/WorkSpaceFiles";
import { constants } from "@/lib/constants";
import { GrooveAutoDetectionStatusProps } from "@/types/props/GrooveAutoDetectionStatusProps";

const GrooveAutoDetectionStatus = ({
  workspaceId,
  sourceId,
}: GrooveAutoDetectionStatusProps) => {
  const [sourceData, setSourceData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [formData, setFormData] = useState({
    autoDetectionEnabled: false,
    monitoringFrequencyMinutes: 30,
    ticketBatchSize: 10,
    reIngestUpdatedTickets: false,
  });

  const fetchSourceDetails = async () => {
    setLoading(true);
    try {
      const response = await getSourceDetails({
        workspaceId,
        sourceId,
      });

      if (response.status === 200) {
        const data = response.data.data;
        setSourceData(data);
        setFormData({
          autoDetectionEnabled: data.auto_detection_enabled || false,
          monitoringFrequencyMinutes: data.monitoring_frequency_minutes || 30,
          ticketBatchSize: data.ticket_batch_size || 10,
          reIngestUpdatedTickets: data.re_ingest_updated_tickets || false,
        });
      }
    } catch (error) {
      showError("Failed to fetch source details");
      console.error("Error fetching source details:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (workspaceId && sourceId) {
      fetchSourceDetails();
    }
  }, [workspaceId, sourceId]);

  const handleSave = async () => {
    setLoading(true);
    try {
      const response = await configureGrooveAutoDetection(
        workspaceId,
        sourceId,
        formData,
      );

      if (response.status === 200) {
        showSuccess("Auto-detection settings updated successfully");
        setIsEditing(false);
        fetchSourceDetails(); // Refresh data
      }
    } catch (error) {
      showError("Failed to update auto-detection settings");
      console.error("Error updating settings:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = () => {
    setIsEditing(false);
    // Reset form data to original values
    setFormData({
      autoDetectionEnabled: sourceData?.auto_detection_enabled || false,
      monitoringFrequencyMinutes:
        sourceData?.monitoring_frequency_minutes || 30,
      ticketBatchSize: sourceData?.ticket_batch_size || 10,
      reIngestUpdatedTickets: sourceData?.re_ingest_updated_tickets || false,
    });
  };

  if (loading && !sourceData) {
    return (
      <Card className="w-full">
        <CardContent className="p-6">
          <div className="flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <span className="ml-2">Loading...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!sourceData) {
    return null;
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Auto-Detection Configuration</span>
          <span
            className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
              sourceData.auto_detection_enabled
                ? "bg-green-100 text-green-800"
                : "bg-gray-100 text-gray-800"
            }`}
          >
            {sourceData.auto_detection_enabled ? "Enabled" : "Disabled"}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {isEditing ? (
          // Edit Mode
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="autoDetection"
                checked={formData.autoDetectionEnabled}
                onCheckedChange={(checked) =>
                  setFormData((prev) => ({
                    ...prev,
                    autoDetectionEnabled: !!checked,
                  }))
                }
              />
              <label htmlFor="autoDetection" className="text-sm font-medium">
                Enable Auto-Detection
              </label>
            </div>

            {formData.autoDetectionEnabled && (
              <>
                <div className="space-y-2">
                  <label htmlFor="frequency" className="text-sm font-medium">
                    Monitoring Frequency (minutes)
                  </label>
                  <Input
                    id="frequency"
                    type="number"
                    min="1"
                    max="1440"
                    value={formData.monitoringFrequencyMinutes}
                    onChange={(e) =>
                      setFormData((prev) => ({
                        ...prev,
                        monitoringFrequencyMinutes:
                          parseInt(e.target.value) || 30,
                      }))
                    }
                  />
                  <p className="text-xs text-gray-500">
                    How often to check for new tickets (1-1440 minutes)
                  </p>
                </div>

                <div className="space-y-2">
                  <label htmlFor="batchSize" className="text-sm font-medium">
                    Ticket Batch Size
                  </label>
                  <Input
                    id="batchSize"
                    type="number"
                    min="1"
                    max="100"
                    value={formData.ticketBatchSize}
                    onChange={(e) =>
                      setFormData((prev) => ({
                        ...prev,
                        ticketBatchSize: parseInt(e.target.value) || 10,
                      }))
                    }
                  />
                  <p className="text-xs text-gray-500">
                    Number of tickets to process in each batch (1-100)
                  </p>
                </div>

                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="reIngest"
                    checked={formData.reIngestUpdatedTickets}
                    onCheckedChange={(checked) =>
                      setFormData((prev) => ({
                        ...prev,
                        reIngestUpdatedTickets: !!checked,
                      }))
                    }
                  />
                  <label htmlFor="reIngest" className="text-sm font-medium">
                    Re-ingest Updated Tickets
                  </label>
                </div>
              </>
            )}

            <div className="flex space-x-2 pt-4">
              <Button onClick={handleSave} disabled={loading}>
                {loading ? "Saving..." : "Save Changes"}
              </Button>
              <Button variant="outline" onClick={handleCancel}>
                Cancel
              </Button>
            </div>
          </div>
        ) : (
          // View Mode
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <span className="text-sm font-medium text-gray-500">
                  Status
                </span>
                <p className="text-sm">
                  {sourceData.auto_detection_enabled ? "Active" : "Inactive"}
                </p>
              </div>

              {sourceData.auto_detection_enabled && (
                <>
                  <div>
                    <span className="text-sm font-medium text-gray-500">
                      Monitoring Frequency
                    </span>
                    <p className="text-sm">
                      {sourceData.monitoring_frequency_minutes} minutes
                    </p>
                  </div>

                  <div>
                    <span className="text-sm font-medium text-gray-500">
                      Batch Size
                    </span>
                    <p className="text-sm">
                      {sourceData.ticket_batch_size} tickets
                    </p>
                  </div>

                  <div>
                    <span className="text-sm font-medium text-gray-500">
                      Re-ingest Updates
                    </span>
                    <p className="text-sm">
                      {sourceData.re_ingest_updated_tickets
                        ? "Enabled"
                        : "Disabled"}
                    </p>
                  </div>
                </>
              )}
            </div>

            <div className="pt-4">
              <Button onClick={() => setIsEditing(true)}>
                Edit Configuration
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default GrooveAutoDetectionStatus;
