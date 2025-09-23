"use client";
import React, { useState, useEffect, useRef, ChangeEvent } from "react";
import LargeModal from "@/components/forms/largeModal";
import { Input } from "@/components/ui/input";
import Select from "react-select";
import { showError } from "@/utils/toastUtils";
import {
  AGENTIC_NAME_MAX_LENGTH,
  AGENTIC_DESCRIPTION_MAX_LENGTH,
} from "@/lib/file-constants";
import {
  AddWorkspaceToolFormData,
  AddWorkspaceToolPageProps,
  OptionType,
} from "@/types/Agentic";
import { AddWorkspaceToolFormSchema } from "@/form_schemas/AddWorkspaceToolForm";
import { FormLabel } from "@/components/ui/FormLabel";
import { agentic_constants } from "@/lib/AgenticConstants";
import { RadioButton } from "@/design_components/radio/radio-button";

import { Textarea } from "../ui/textarea";
import { SmartDropdown } from "../../design_components/SmartDropdown";
import { LLMOptions } from "@/agent_schemas/agent_schema";

const llmOptions = LLMOptions;

const AddWorkspaceToolPage: React.FC<AddWorkspaceToolPageProps> = ({
  isOpen,
  onClose,
  onSubmit,
  fieldLabels = {},
  mcpOptions,
  systemToolOptions,
  loadingOptions = false,
  datasetsOptions,
  isEditMode = false,
  initialData,
  showDataset = true,
  showLLMProvider,
  showPromptInstructions,
  agentToEdit,
  fetchSystemTools,
  fetchMCPTools,
  onSystemToolSelectChange,
  isReadOnly,
}) => {
  const [formData, setFormData] = useState<AddWorkspaceToolFormData>(
    initialData ?? {
      agentName: "",
      description: "",
      workspaceTool: "",
      llmProvider: "",
      selectedMCPs: [],
      selectedSystemTools: [],
      selectedDatasets: [],
      prompt_instructions: "",
    },
  );
  const [errors, setErrors] = useState<any>({});
  const [hasInitialized, setHasInitialized] = useState(false);
  const isAgentMode = !!fieldLabels.agentName;
  const [filteredDatasets, setFilteredDatasets] = useState<OptionType[]>(
    datasetsOptions ?? [],
  );

  const [openDropdown, setOpenDropdown] = useState<string | null>(null);

  // Refs for auto-scrolling
  const mcpDropdownRef = useRef<HTMLDivElement>(null);
  const systemToolDropdownRef = useRef<HTMLDivElement>(null);
  const datasetDropdownRef = useRef<HTMLDivElement>(null);

  const toolTypeOptions = [
    {
      value: agentic_constants.TOOL_TYPE.LABEL.MCP,
      label: agentic_constants.TOOL_TYPE.VALUE.MCP,
    },
    {
      value: agentic_constants.TOOL_TYPE.VALUE.SYSTEM_TOOL,
      label: agentic_constants.TOOL_TYPE.LABEL.SYSTEM_TOOL,
    },
  ];

  // Auto-scroll function
  const scrollToElement = (ref: React.RefObject<HTMLDivElement | null>) => {
    if (ref.current) {
      const element = ref.current;
      setTimeout(() => {
        if (element && element.scrollIntoView) {
          element.scrollIntoView({
            behavior: "smooth",
            block: "center",
            inline: "nearest",
          });
        }
      }, 100);
    }
  };

  // Function to handle dropdown open/close
  const handleDropdownToggle = (dropdownId: string, shouldOpen: boolean) => {
    if (shouldOpen) {
      setOpenDropdown(dropdownId);
      // Auto-scroll to the opened dropdown
      const refMap: { [key: string]: React.RefObject<HTMLDivElement | null> } =
        {
          mcp: mcpDropdownRef,
          systemTools: systemToolDropdownRef,
          datasets: datasetDropdownRef,
        };
      const ref = refMap[dropdownId];
      if (ref) {
        scrollToElement(ref);
      }
    } else {
      setOpenDropdown(null);
    }
  };

  useEffect(() => {
    if (isOpen && initialData && !hasInitialized) {
      console.log("initialData", initialData);
      setFormData(initialData);
      setHasInitialized(true);
    }
  }, [isOpen, initialData, hasInitialized]);

  // Reset everything when modal closes
  useEffect(() => {
    if (!isOpen) {
      setFormData({
        agentName: "",
        description: "",
        workspaceTool: "",
        selectedMCPs: [],
        selectedSystemTools: [],
        selectedDatasets: [],
        prompt_instructions: "",
      });
      setErrors({});
      setHasInitialized(false); // allow next open to initialize again
      setOpenDropdown(null); // reset dropdown state
    }
  }, [isOpen]);

  useEffect(() => {
    const selectedSystemTools = formData.selectedSystemTools || [];
    const systemTools = systemToolOptions || [];
    if (
      formData.workspaceTool === agentic_constants.TOOL_TYPE.SYSTEM_TOOL &&
      selectedSystemTools.length === 1
    ) {
      const selectedTool = systemTools.find(
        (opt) => opt.value === selectedSystemTools[0],
      )?.label;

      if (selectedTool === agentic_constants.TOOL_NAMES.TEXT_TO_SQL) {
        setFilteredDatasets(
          (datasetsOptions ?? []).filter(
            (ds: any) => ds.type === agentic_constants.DATASET_TYPE.SQL,
          ),
        );
      } else if (selectedTool === agentic_constants.TOOL_NAMES.VECTOR_SEARCH) {
        setFilteredDatasets(
          (datasetsOptions ?? []).filter(
            (ds: any) =>
              ds.type === agentic_constants.DATASET_TYPE.UNSTRUCTURED,
          ),
        );
      } else if (selectedTool === agentic_constants.TOOL_NAMES.FILE_SYSTEM) {
        setFilteredDatasets(
          (datasetsOptions ?? []).filter(
            (ds: any) =>
              ds.type === agentic_constants.DATASET_TYPE.UNSTRUCTURED,
          ),
        );
      } else if (selectedTool === "Graph Search Tool") {
        setFilteredDatasets(
          (datasetsOptions ?? []).filter(
            (ds: any) =>
              ds.type === agentic_constants.DATASET_TYPE.UNSTRUCTURED &&
              ds.graph_status === agentic_constants.TRUTH_VALUES.TRUE,
          ),
        );
      } else {
        setFilteredDatasets(datasetsOptions ?? []);
      }
    } else {
      setFilteredDatasets(datasetsOptions ?? []);
    }
  }, [
    formData.selectedSystemTools,
    datasetsOptions,
    formData.workspaceTool,
    systemToolOptions,
  ]);

  // Auto-scroll when MCP dropdown appears
  useEffect(() => {
    if (
      formData.workspaceTool === agentic_constants.TOOL_TYPE.MCP_TOOL &&
      mcpOptions?.length
    ) {
      scrollToElement(mcpDropdownRef);
    }
  }, [formData.workspaceTool, mcpOptions]);

  // Auto-scroll when System Tool dropdown appears
  useEffect(() => {
    if (
      formData.workspaceTool === agentic_constants.TOOL_TYPE.SYSTEM_TOOL &&
      systemToolOptions?.length
    ) {
      scrollToElement(systemToolDropdownRef);
    }
  }, [formData.workspaceTool, systemToolOptions]);

  // Auto-scroll when Dataset dropdown appears
  useEffect(() => {
    if (
      formData.workspaceTool === agentic_constants.TOOL_TYPE.SYSTEM_TOOL &&
      showDataset &&
      formData.selectedSystemTools &&
      formData.selectedSystemTools.length > 0 &&
      filteredDatasets?.length
    ) {
      scrollToElement(datasetDropdownRef);
    }
  }, [
    formData.workspaceTool,
    showDataset,
    formData.selectedSystemTools,
    filteredDatasets,
  ]);

  const handleChange = (
    e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    setErrors((prev: any) => ({
      ...prev,
      [name]: undefined, // Clear error for the field being edited
    }));
  };

  const handleLlmProviderChange = (selected: OptionType | null) => {
    setFormData((prev) => ({
      ...prev,
      llmProvider: selected?.value ?? "",
    }));
    setErrors((prev: any) => ({
      ...prev,
      llmProvider: undefined, // Clear error for llmProvider1
    }));
  };

  const toolKind = initialData?.workspaceTool;

  const handleRadioChange = (value: string) => {
    // Block switching to invalid value in edit mode
    console.log({ toolKind, value });
    if (
      (toolKind === agentic_constants.TOOL_KIND.MCP &&
        value === agentic_constants.TOOL_TYPE.SYSTEM_TOOL) ||
      (toolKind === agentic_constants.TOOL_KIND.SYSTEM &&
        value === agentic_constants.TOOL_TYPE.MCP_TOOL)
    ) {
      return;
    }

    setFormData((prev) => ({
      ...prev,
      workspaceTool: value,
      selectedMCPs: [],
      selectedSystemTools: [],
      selectedDatasets: [],
    }));
    setErrors((prev: any) => ({
      ...prev,
      workspaceTool: undefined,
      selectedMCPs: undefined,
      selectedSystemTools: undefined,
      selectedDatasets: undefined,
    }));
    setOpenDropdown(null); // close any open dropdowns when tool type changes

    if (value === agentic_constants.TOOL_TYPE.MCP_TOOL && fetchMCPTools) {
      fetchMCPTools(agentic_constants.TOOL_KIND.MCP);
    } else if (
      value === agentic_constants.TOOL_TYPE.SYSTEM_TOOL &&
      fetchSystemTools
    ) {
      fetchSystemTools(agentic_constants.TOOL_KIND.SYSTEM);
    }
  };

  const validateForm = (
    data: AddWorkspaceToolFormData,
    isAgent: boolean,
    showDataset: boolean,
  ) => {
    try {
      AddWorkspaceToolFormSchema(isAgent, showDataset).parse(data);
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

  const handleSubmit = async () => {
    let flag = !!fieldLabels.agentName;
    if (!validateForm(formData, flag, showDataset)) {
      return;
    }
    try {
      onSubmit(formData);
    } catch (err: any) {
      showError(err.message ?? "Failed to create tool.");
    }
  };

  console.log("toolKind", toolKind);

  return (
    <LargeModal
      isOpen={isOpen}
      onClose={onClose}
      onSubmit={handleSubmit}
      title={
        isReadOnly
          ? "Tool Details"
          : isEditMode
            ? "Edit Tool to Workspace"
            : " Add Tool to Workspace"
      }
      actionButton="Submit"
      hideSubmitButton={isReadOnly}
      type="agent"
      fullWidth={false}
      fullHeight={false}
    >
      <form
        className="p-4 space-y-4 text-sm"
        onSubmit={(e) => {
          e.preventDefault();
          handleSubmit();
        }}
      >
        <div>
          <FormLabel htmlFor="agentName">
            {fieldLabels.agentName ?? "Tool Name"}{" "}
            <span className="text-red-500">*</span>
          </FormLabel>
          <Input
            id="agentName"
            name="agentName"
            type="text"
            className="w-full border rounded-sm p-2"
            maxLength={AGENTIC_NAME_MAX_LENGTH}
            value={formData.agentName}
            onChange={handleChange}
            placeholder={
              fieldLabels.agentName ? "Enter Agent Name" : "Enter Tool Name"
            }
            disabled={isReadOnly}
          />
          <p className="text-xs text-muted-foreground text-right pt-2 h-0">
            {formData.agentName.length}/{AGENTIC_NAME_MAX_LENGTH} characters
          </p>
          {errors.agentName && (
            <p className="text-red-500 text-sm">{errors.agentName.message}</p>
          )}
        </div>

        <div>
          <FormLabel htmlFor="description">
            {fieldLabels.description ?? "Tool Description"}
          </FormLabel>
          <Textarea
            id="description"
            name="description"
            className="w-full border rounded-sm p-2"
            maxLength={AGENTIC_DESCRIPTION_MAX_LENGTH}
            value={formData.description}
            onChange={handleChange}
            placeholder="Describe what this tool is for"
            rows={3}
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

        {showLLMProvider && (
          <div>
            <FormLabel htmlFor="llmProvider">
              {fieldLabels?.llmProvider}
            </FormLabel>
            <Select
              options={llmOptions}
              value={llmOptions.find(
                (option) => option.value === formData.llmProvider,
              )}
              onChange={handleLlmProviderChange}
              isDisabled={isReadOnly}
            />
            {errors.llmProvider && (
              <p className="text-red-500 text-sm">
                {errors.llmProvider.message}
              </p>
            )}
          </div>
        )}
        <div>
          <FormLabel>
            {fieldLabels.workspaceTool ?? "Associate Tool With"}{" "}
            <span className="text-red-500">*</span>
          </FormLabel>
          <RadioButton
            value={formData.workspaceTool}
            onChange={handleRadioChange}
            options={toolTypeOptions}
            isDisabled={isReadOnly}
          ></RadioButton>
          {errors.workspaceTool && (
            <p className="text-red-500 text-sm">
              {errors.workspaceTool.message}
            </p>
          )}
        </div>

        {formData.workspaceTool === agentic_constants.TOOL_TYPE.MCP_TOOL && (
          <div ref={mcpDropdownRef}>
            <FormLabel htmlFor="selectedMCPs">
              Select MCP <span className="text-red-500">*</span>
            </FormLabel>
            <SmartDropdown
              options={mcpOptions || []}
              value={formData.selectedMCPs ?? []}
              onChange={(selected) => {
                const values = Array.isArray(selected)
                  ? selected.map((s) => s.value)
                  : selected
                    ? [selected.value]
                    : [];
                setFormData((prev) => ({
                  ...prev,
                  selectedMCPs: values,
                }));
              }}
              isOpen={openDropdown === "mcp"}
              onOpenChange={(isOpen) => handleDropdownToggle("mcp", isOpen)}
              variant="multi"
              searchable={true}
              showTags
              placeholder="Select MCPs"
              state={errors.selectedMCPs ? "error" : "default"}
              className="text-sm"
              isDisabled={isReadOnly}
            />
            {errors.selectedMCPs && (
              <p className="text-red-500 text-sm">
                {errors.selectedMCPs.message}
              </p>
            )}
          </div>
        )}

        {formData.workspaceTool === agentic_constants.TOOL_TYPE.SYSTEM_TOOL && (
          <>
            <div ref={systemToolDropdownRef}>
              <FormLabel htmlFor="selectedSystemTools">
                Select System Tool <span className="text-red-500">*</span>
              </FormLabel>
              {isAgentMode ? (
                <SmartDropdown
                  options={systemToolOptions || []}
                  value={formData.selectedSystemTools ?? []}
                  onChange={(selected) => {
                    const values = Array.isArray(selected)
                      ? selected.map((s) => s.value)
                      : selected
                        ? [selected.value]
                        : [];
                    setFormData((prev) => ({
                      ...prev,
                      selectedSystemTools: values,
                    }));
                  }}
                  isOpen={openDropdown === "systemTools"}
                  onOpenChange={(isOpen) =>
                    handleDropdownToggle("systemTools", isOpen)
                  }
                  variant="multi"
                  searchable={true}
                  showTags
                  placeholder="Select system tools"
                  state={errors.selectedSystemTools ? "error" : "default"}
                  className="text-sm"
                  isDisabled={isReadOnly}
                />
              ) : (
                <SmartDropdown
                  options={systemToolOptions || []}
                  value={formData.selectedSystemTools?.[0] ?? null}
                  onChange={(selected) => {
                    const selectedOption = selected as OptionType;
                    const selectedValue = selectedOption?.value;

                    setFormData((prev) => ({
                      ...prev,
                      selectedSystemTools: selectedValue ? [selectedValue] : [],
                    }));

                    if (onSystemToolSelectChange && selectedValue) {
                      onSystemToolSelectChange([selectedValue]);
                    }
                  }}
                  isOpen={openDropdown === "systemTools"}
                  onOpenChange={(isOpen) =>
                    handleDropdownToggle("systemTools", isOpen)
                  }
                  className="text-sm"
                  isDisabled={isReadOnly}
                />
              )}
              {errors.selectedSystemTools && (
                <p className="text-red-500 text-sm">
                  {errors.selectedSystemTools.message}
                </p>
              )}
            </div>

            {formData.workspaceTool ===
              agentic_constants.TOOL_TYPE.SYSTEM_TOOL &&
              showDataset &&
              formData.selectedSystemTools &&
              formData.selectedSystemTools.length > 0 && (
                <div ref={datasetDropdownRef}>
                  <FormLabel htmlFor="selectedDatasets">
                    Add Datasets<span className="text-red-500">*</span>
                  </FormLabel>
                  <SmartDropdown
                    options={filteredDatasets ?? []}
                    value={formData.selectedDatasets ?? []}
                    onChange={(selected) => {
                      const values = Array.isArray(selected)
                        ? selected.map((s) => s.value)
                        : selected
                          ? [selected.value]
                          : [];
                      setFormData((prev) => ({
                        ...prev,
                        selectedDatasets: values,
                      }));
                    }}
                    isOpen={openDropdown === "datasets"}
                    onOpenChange={(isOpen) =>
                      handleDropdownToggle("datasets", isOpen)
                    }
                    variant="multi"
                    searchable={true}
                    showTags
                    placeholder="Select datasets"
                    state={errors.selectedDatasets ? "error" : "default"}
                    className="text-sm"
                    isDisabled={isReadOnly}
                  />

                  {errors.selectedDatasets && (
                    <p className="text-red-500 text-sm">
                      {errors.selectedDatasets.message}
                    </p>
                  )}
                </div>
              )}
          </>
        )}

        {showPromptInstructions && (
          <div>
            <FormLabel htmlFor="instructions">
              {fieldLabels?.prompt_instructions}
            </FormLabel>
            <Textarea
              id="instructions"
              name="prompt_instructions"
              className="w-full border rounded-sm p-2"
              value={formData.prompt_instructions}
              onChange={handleChange}
              placeholder="E.g., 'Always be polite and concise in responses.'"
              required
              rows={4}
              disabled={isReadOnly}
            />
            {errors.prompt_instructions && (
              <p className="text-red-600 text-sm">
                {errors.prompt_instructions.message}
              </p>
            )}
          </div>
        )}
      </form>
    </LargeModal>
  );
};

export default AddWorkspaceToolPage;
