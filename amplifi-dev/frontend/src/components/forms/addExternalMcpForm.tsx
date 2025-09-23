"use client";

import React, { useState, useEffect, ChangeEvent } from "react";
import Image from "next/image";
import LargeModal from "@/components/forms/largeModal";
import { Input } from "@/components/ui/input";
import {
  AGENTIC_NAME_MAX_LENGTH,
  AGENTIC_DESCRIPTION_MAX_LENGTH,
} from "@/lib/file-constants";
import { AddExternalMcpFormProps, ExternalMcpFormData } from "@/types/Agentic";
import { ExternalMcpFormSchema } from "@/form_schemas/ExternalMcpForm";
import { FormLabel } from "@/components/ui/FormLabel";
import CodeEditor from "@uiw/react-textarea-code-editor";
import { Textarea } from "../ui/textarea";
import { Button } from "../ui/button";
import { validateConfigCode } from "@/api/agents";
import Success from "@/assets/icons/success-state.svg";
import caution from "@/assets/icons/caution-state.svg";
import Processing from "@/assets/icons/processing-state.svg";
import {
  identifyUserFromObject,
  captureEvent,
  hashString,
} from "@/utils/posthogUtils";
import { useUser } from "@/context_api/userContext";

const defaultConfigCode = `{
  "example_mcp": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/example-mcp", "run-task"],
    "env": {}
  }
}`;

const AddExternalMcpForm: React.FC<AddExternalMcpFormProps> = ({
  isOpen,
  onClose,
  onSubmit,
  initialData,
  isEditMode = false,
  isReadOnly,
}) => {
  const { user } = useUser();
  const [formData, setFormData] = useState<ExternalMcpFormData>(
    initialData || {
      mcpName: "",
      description: "",
      configCode: defaultConfigCode,
      tool_kind: "mcp",
      mcp_tool: {
        mcp_subtype: "external",
        mcp_server_config: "",
      },
    },
  );
  const [errors, setErrors] = useState<any>({});
  const [testSuccess, setTestSuccess] = useState<boolean | null>(null);
  const [testErrorMsg, setTestErrorMsg] = useState<string | null>(null);
  const [availableTools, setAvailableTools] = useState<any[]>([]);
  const [showTools, setShowTools] = useState(false);
  const [isTesting, setIsTesting] = useState(false);

  const handleChange = (
    e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    setErrors((prev: any) => ({ ...prev, [name]: undefined }));
  };

  const handleConfigCodeChange = (value: string) => {
    setFormData((prev) => ({ ...prev, configCode: value }));
    setErrors((prev: any) => ({ ...prev, configCode: undefined }));
  };

  const validateForm = (data: ExternalMcpFormData) => {
    try {
      ExternalMcpFormSchema.parse(data);
      setErrors({});
      return true;
    } catch (err: any) {
      const fieldErrors: any = {};
      if (err.errors) {
        err.errors.forEach((error: any) => {
          fieldErrors[error.path[0]] = { message: error.message };
        });
      }
      setErrors(fieldErrors);
      return false;
    }
  };

  const handleTest = async () => {
    setIsTesting(true);
    setTestSuccess(null);
    setTestErrorMsg(null);

    let parsedConfig;
    try {
      identifyUserFromObject(user);
      captureEvent("tool_auth_test_run", {
        tool_id: hashString(formData.mcpName || "unnamed_mcp"),
        auth_status: isTesting
          ? "testing"
          : testSuccess === true
            ? "success"
            : testSuccess === false
              ? "failed"
              : "not_started",
        retry_count: testSuccess === false ? 1 : 0,
        description: "User runs auth test on a tool",
      });
      parsedConfig = JSON.parse(formData.configCode);
    } catch (parseError) {
      console.error("mcp configurrration error", parseError);
      setTestSuccess(false);
      setTestErrorMsg("Invalid JSON format.");
      setIsTesting(false);
      return;
    }

    try {
      const response = await validateConfigCode(
        JSON.stringify({ mcp_server_config: parsedConfig }),
      );
      setTestSuccess(true);
      setTestErrorMsg(null);
      setAvailableTools(response.data?.data?.available_tools || []);
      setShowTools(false);
    } catch (error: any) {
      setTestSuccess(false);
      setAvailableTools([]);
      let message = "Invalid MCP schema structure.";
      if (error?.response?.data) {
        if (typeof error.response.data === "string") {
          message = error.response.data;
        } else {
          message =
            error.response.data?.detail?.message ||
            "Validation failed please check the configuration again.";
        }
      }
      setTestErrorMsg(message);
    } finally {
      setIsTesting(false);
    }
  };

  const handleSubmit = () => {
    identifyUserFromObject(user);
    captureEvent("mcp_tool_created", {
      tool_type: "external",
      org_id: hashString(user?.clientId || ""),
      validation_status:
        testSuccess === true
          ? "success"
          : testSuccess === false
            ? "failed"
            : "not_tested",
      description: "User submits new MCP tool form",
    });
    if (!validateForm(formData)) return;
    const submissionData = {
      ...formData,
      configCode: JSON.stringify({
        mcp_server_config: JSON.parse(formData.configCode),
      }),
    };
    onSubmit(submissionData);
  };

  useEffect(() => {
    if (!isOpen) {
      setFormData({
        mcpName: "",
        description: "",
        configCode: defaultConfigCode,
        tool_kind: "mcp",
      });
      setErrors({});
      setTestSuccess(null);
      setTestErrorMsg(null);
      setIsTesting(false);
      setAvailableTools([]);
      setShowTools(false);
    }
  }, [isOpen]);

  useEffect(() => {
    if (isOpen && initialData) {
      setFormData({
        ...initialData,
        configCode: initialData.configCode || "",
      });
      setErrors({});
    }
  }, [isOpen, initialData]);

  return (
    <LargeModal
      isOpen={isOpen}
      onClose={onClose}
      onSubmit={handleSubmit}
      title={
        isReadOnly ? "MCP Details" : isEditMode ? "Edit MCP" : "Create New MCP"
      }
      actionButton="Submit"
      hideSubmitButton={isReadOnly}
      type="externalMcp"
      fullWidth={false}
      fullHeight={false}
      disabled={testSuccess !== true}
    >
      <form
        className="p-4 space-y-4 text-sm"
        onSubmit={(e) => {
          e.preventDefault();
          handleSubmit();
        }}
      >
        {/* MCP Name */}
        <div>
          <FormLabel htmlFor="mcpName">
            MCP Name <span className="text-red-500">*</span>
          </FormLabel>
          <Input
            id="mcpName"
            name="mcpName"
            value={formData.mcpName}
            onChange={handleChange}
            placeholder="Enter MCP Name"
            maxLength={AGENTIC_NAME_MAX_LENGTH}
            disabled={isReadOnly}
          />
          <p className="text-xs text-muted-foreground text-right pt-2 h-0">
            {formData.mcpName.length}/{AGENTIC_NAME_MAX_LENGTH} characters
          </p>
          {errors.mcpName && (
            <p className="text-red-500 text-sm">{errors.mcpName.message}</p>
          )}
        </div>

        {/* Description */}
        <div>
          <FormLabel htmlFor="description">
            MCP Description <span className="text-red-500">*</span>
          </FormLabel>
          <Textarea
            id="description"
            name="description"
            value={formData.description}
            onChange={handleChange}
            placeholder="Write what this MCP is for"
            className="w-full border rounded-sm p-2"
            maxLength={AGENTIC_DESCRIPTION_MAX_LENGTH}
            disabled={isReadOnly}
          />
          <p className="text-xs text-muted-foreground text-right h-0">
            {formData.description.length}/{AGENTIC_DESCRIPTION_MAX_LENGTH}{" "}
            characters
          </p>
          {errors.description && (
            <p className="text-red-500 text-sm">{errors.description.message}</p>
          )}
        </div>

        {/* Config Code */}
        <div className="flex flex-col mb-1">
          <FormLabel htmlFor="configCode">
            Paste Config Code Here <span className="text-red-500">*</span>
          </FormLabel>
          <div className="border rounded-sm overflow-hidden">
            <CodeEditor
              value={formData.configCode}
              language="json"
              placeholder="Paste configuration JSON here"
              onChange={(evn: any) => handleConfigCodeChange(evn.target.value)}
              padding={15}
              data-color-mode="light"
              style={{
                fontSize: 12,
                backgroundColor: "white",
                color: "black",
                fontFamily:
                  'ui-monospace,SFMono-Regular,"SF Mono",Consolas,"Liberation Mono",Menlo,monospace',
                minHeight: "200px",
              }}
              className="no-syntax"
              disabled={isReadOnly}
            />
          </div>

          {/* Status Message */}
          <div className="flex items-center justify-between mt-2">
            <div>
              {isTesting ? (
                <div className="flex items-center gap-2 text-custom-Processing text-sm font-medium">
                  <Image
                    src={Processing}
                    alt="Processing icon"
                    width={12}
                    height={12}
                    className="animate-spin"
                  />
                  Processing JSON and detecting tools…
                </div>
              ) : testSuccess === true ? (
                availableTools.length > 0 ? (
                  <div className="text-sm flex items-center gap-2 text-custom-Success font-medium">
                    <Image src={Success} alt="Success" width={12} height={12} />
                    Configured Successfully:
                    <span
                      onClick={() => setShowTools((prev) => !prev)}
                      className="ml-1 underline cursor-pointer"
                    >
                      {availableTools.length} tool(s) detected{" "}
                      {showTools ? "▴" : "▾"}
                    </span>
                  </div>
                ) : (
                  <div className="flex items-center gap-2 text-xs text-custom-Danger font-medium">
                    <Image src={caution} alt="Caution" width={12} height={12} />
                    MCP is connected successfully and has 0 tools.
                  </div>
                )
              ) : testSuccess === false ? (
                <div className="flex items-center gap-2 text-xs text-custom-Danger font-medium">
                  <Image src={caution} alt="Caution" width={12} height={12} />
                  <span>{testErrorMsg}</span>
                </div>
              ) : null}
            </div>

            {!isTesting && (
              <Button
                type="button"
                onClick={handleTest}
                className="mb-3 text-xs bg-white text-black rounded border"
                disabled={isReadOnly}
              >
                {testSuccess === true
                  ? "Reconfigure"
                  : testSuccess === false
                    ? "Retry"
                    : "Configure MCP"}
              </Button>
            )}
          </div>

          {showTools && availableTools.length > 0 && (
            <div className="mt-1.25 border rounded-md p-2 flex flex-wrap gap-1">
              {availableTools.map((tool, index) => (
                <span
                  key={index}
                  className="px-3 py-1 rounded-full bg-[#D7E2CD] text-gray-800 text-xs font-medium"
                >
                  {tool.name}
                </span>
              ))}
            </div>
          )}

          {errors.configCode && (
            <p className="text-red-500 text-sm mt-1">
              {errors.configCode.message}
            </p>
          )}
        </div>
      </form>
    </LargeModal>
  );
};

export default AddExternalMcpForm;
