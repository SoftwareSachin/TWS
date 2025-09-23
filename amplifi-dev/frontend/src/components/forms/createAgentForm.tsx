import React, { ChangeEvent, useEffect, useState } from "react";
import { useParams } from "next/navigation";
import LargeModal from "@/components/forms/largeModal";
import Select, {
  ActionMeta,
  MultiValue,
  OptionProps,
  SingleValue,
} from "react-select";
import { Input } from "@/components/ui/input";
import {
  AGENTIC_DESCRIPTION_MAX_LENGTH,
  AGENTIC_NAME_MAX_LENGTH,
} from "@/lib/file-constants";
import {
  AgentFormData,
  CreateAgentFormProps,
  OptionType,
} from "@/types/Agentic";
import { AgentFormSchema } from "@/form_schemas/AgentForm";
import { FormLabel } from "@/components/ui/FormLabel";
import { Textarea } from "../ui/textarea";
import { Checkbox, CheckboxIndicator } from "@radix-ui/react-checkbox";
import { CheckIcon } from "lucide-react";
import { agentic_constants } from "@/lib/AgenticConstants";
import {
  CustomMultiValueLabel,
  customStylesAgent,
} from "../styles/selectStyles";
import { LLMOptions } from "@/agent_schemas/agent_schema";
import {
  identifyUserFromObject,
  hashString,
  captureEvent,
} from "@/utils/posthogUtils";
import { useUser } from "@/context_api/userContext";
const CustomOption = ({
  children,
  isSelected,
  innerProps,
  innerRef,
  data,
  setValue,
  getValue,
}: OptionProps<OptionType, true>) => {
  const handleCheckboxChange = (checked: boolean) => {
    const currentValues = getValue();
    const newValues = checked
      ? [...currentValues, data]
      : currentValues.filter((opt) => opt.value !== data.value);
    setValue(newValues, "select-option");
  };
  return (
    <div
      ref={innerRef}
      {...innerProps}
      className="flex items-center space-x-2 px-2 py-1 cursor-pointer hover:bg-gray-100"
      onClick={(e) => e.stopPropagation()}
    >
      <Checkbox
        checked={isSelected}
        onCheckedChange={handleCheckboxChange}
        className="w-5 h-5 border-2 rounded-sm flex items-center justify-center data-[state=checked]:bg-blue-600 data-[state=checked]:text-white"
      >
        <CheckboxIndicator className="flex items-center justify-center">
          <CheckIcon className="w-4 h-4" />
        </CheckboxIndicator>
      </Checkbox>
      <div className="text-sm">
        <span className="font-medium">{data.label}</span>{" "}
        {data.tool_type && (
          <span className="text-custom-toolColor text-xs ml-1">
            (
            {data.tool_type === agentic_constants.TOOL_KIND.MCP
              ? agentic_constants.TOOL_TYPE.MCP_TOOL
              : agentic_constants.TOOL_TYPE.SYSTEM_TOOL}
            )
          </span>
        )}
      </div>
    </div>
  );
};

const CreateAgentForm: React.FC<CreateAgentFormProps> = ({
  isOpen,
  onClose,
  onSubmit,
  fieldLabels,
  workspaceToolOptions,
  onToolsDropdownOpen,
  loadingTools,
  isEditMode = false,
  agentToEdit,
  initialData,
  isReadOnly,
}) => {
  const params = useParams();
  const workspaceId = Array.isArray(params.workspaceId)
    ? params.workspaceId[0]
    : params.workspaceId;

  const { user } = useUser();
  const [formData, setFormData] = useState<AgentFormData>({
    agentName: "",
    description: "",
    workspaceTool: [],
    llmProvider: "OpenAI GPT-3.5",
    prompt_instructions: "",
  });
  const [errors, setErrors] = useState<any>({});
  const llmOptions = LLMOptions;

  const handleChange = (
    e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
    setErrors((prev: any) => ({
      ...prev,
      [name]: undefined,
    }));
  };

  const handleWorkspaceToolsChange = (
    selected: MultiValue<OptionType>,
    _actionMeta: ActionMeta<OptionType>,
  ) => {
    setFormData((prev) => ({
      ...prev,
      workspaceTool: selected.map((option) => option.value),
    }));
    setErrors((prev: any) => ({
      ...prev,
      workspaceTool: undefined,
    }));
  };

  const handleLlmProviderChange = (selected?: SingleValue<OptionType>) => {
    setFormData((prev) => ({
      ...prev,
      llmProvider: selected?.provider || "",
      llmModel: selected?.value || "",
    }));
    setErrors((prev: any) => ({
      ...prev,
      llmProvider: undefined,
    }));
  };

  const validateForm = (data: AgentFormData) => {
    try {
      AgentFormSchema.parse(data);
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

  const handleSubmit = () => {
    if (!validateForm(formData)) {
      return;
    }

    identifyUserFromObject(user);

    if (isEditMode && agentToEdit) {
      // Track tool changes in edit mode
      const oldTools = agentToEdit.workspace_tool_ids || [];
      const newTools = formData.workspaceTool || [];
      const toolsChanged =
        JSON.stringify(oldTools.sort()) !== JSON.stringify(newTools.sort());

      if (toolsChanged) {
        captureEvent("tool_added_to_agent", {
          agent_id_hash: hashString(agentToEdit.id || ""),
          tool_type: "workspace_tool",
          workspace_id_hash: hashString(workspaceId || ""),
          description: "User assigns tool to an agent",
        });
      }

      captureEvent("agent_edited", {
        agent_id_hash: hashString(agentToEdit.id || ""),
        field_changed: "config",
        has_config_change: true,
        workspace_id_hash: hashString(workspaceId || ""),
        description: "User updates config of existing agent",
      });
    } else {
      captureEvent("agent_created", {
        agent_type: "dynamic",
        workspace_id: hashString(workspaceId || ""),
        created_by_id: hashString(user?.clientId || ""),
        description: "User adds a new Agent",
      });
    }

    onSubmit(formData);
  };

  useEffect(() => {
    if (!isOpen && !agentToEdit) {
      setFormData({
        agentName: "",
        description: "",
        workspaceTool: [],
        llmProvider: "OpenAI GPT-3.5",
        prompt_instructions: "",
      });
      setErrors({});
    }
  }, [isOpen, agentToEdit]);

  useEffect(() => {
    console.log("AgentTOedit", initialData);
    if (isOpen && initialData) {
      setFormData(initialData);
    }
  }, [isOpen, agentToEdit]);

  return (
    <LargeModal
      isOpen={isOpen}
      onClose={onClose}
      onSubmit={handleSubmit}
      title={
        isReadOnly
          ? "Agent Details"
          : isEditMode
            ? "Edit Agent"
            : "Create New Agent"
      }
      actionButton="Submit"
      hideSubmitButton={isReadOnly}
      type="agent"
      fullWidth={false}
      fullHeight={false}
    >
      <form
        className="p-6 space-y-4 text-sm"
        onSubmit={(e) => {
          e.preventDefault();
          handleSubmit();
        }}
      >
        <div className="relative">
          <FormLabel htmlFor="agentName">
            {fieldLabels?.agentName} <span className="text-red-500">*</span>
          </FormLabel>
          <Input
            id="agentName"
            name="agentName"
            type="text"
            className="w-full border rounded-sm p-2"
            maxLength={AGENTIC_NAME_MAX_LENGTH}
            value={formData.agentName}
            onChange={handleChange}
            placeholder="Enter Agent Name"
            required
            disabled={isReadOnly}
          />
          <p className="text-sm text-muted-foreground text-right p-0.5">
            {formData.agentName.length}/{AGENTIC_NAME_MAX_LENGTH} characters
          </p>
          {errors.agentName && (
            <p className="text-red-500 text-sm">{errors.agentName.message}</p>
          )}
        </div>

        <div>
          <FormLabel htmlFor="description">
            {fieldLabels?.description}
          </FormLabel>
          <textarea
            id="description"
            name="description"
            className="w-full border rounded-sm p-2"
            maxLength={AGENTIC_DESCRIPTION_MAX_LENGTH}
            value={formData.description}
            onChange={handleChange}
            placeholder="Write what this tool does"
            disabled={isReadOnly}
          />
          <p className="text-sm text-muted-foreground text-right p-0.5">
            {formData.description.length}/{AGENTIC_DESCRIPTION_MAX_LENGTH}{" "}
            characters
          </p>
          {errors.description && (
            <p className="text-red-500 text-sm">{errors.description.message}</p>
          )}
        </div>

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
            <p className="text-red-500 text-sm">{errors.llmProvider.message}</p>
          )}
        </div>

        <div>
          <FormLabel htmlFor="workspaceTool">
            {fieldLabels?.workspaceTool} <span className="text-red-500">*</span>
          </FormLabel>
          <Select
            isMulti
            isSearchable={false}
            options={workspaceToolOptions}
            value={workspaceToolOptions.filter((option) =>
              formData.workspaceTool.includes(option.value),
            )}
            onChange={handleWorkspaceToolsChange}
            className="text-sm"
            hideSelectedOptions={false}
            closeMenuOnSelect={false}
            styles={customStylesAgent}
            onMenuOpen={onToolsDropdownOpen}
            isLoading={loadingTools}
            components={{
              Option: CustomOption,
              MultiValueLabel: CustomMultiValueLabel,
            }}
            isDisabled={isReadOnly}
          />
          {errors.workspaceTool && (
            <p className="text-red-500 text-sm">
              {errors.workspaceTool.message}
            </p>
          )}
        </div>

        <div>
          <FormLabel htmlFor="instructions">
            {fieldLabels?.prompt_instructions}
          </FormLabel>
          <Textarea
            id="instructions"
            name="prompt_instructions"
            className="w-full border rounded-sm p-1"
            value={formData.prompt_instructions}
            onChange={handleChange}
            placeholder="E.g., ‘Always be polite and concise in responses.’"
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
      </form>
    </LargeModal>
  );
};

export default CreateAgentForm;
