import React, { ChangeEvent, useEffect, useState } from "react";
import LargeModal from "@/components/forms/largeModal";
import Select, { ActionMeta, SingleValue, StylesConfig } from "react-select";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  AGENTIC_DESCRIPTION_MAX_LENGTH,
  AGENTIC_NAME_MAX_LENGTH,
  ALLOWED_NAME_REGEX,
  RESERVED_WORDS,
} from "@/lib/file-constants";
import { ChatAppFormData, CreateChatAppFormProps } from "@/types/ChatApp";
import { ChatAppFormSchema } from "@/form_schemas/ChatAppFormSchema";
import { FormLabel } from "@/components/ui/FormLabel";
import { OptionType } from "@/types/Agentic";
import { customStylesChatApp } from "../styles/selectStyles";

const CreateChatAppForm: React.FC<CreateChatAppFormProps> = ({
  isOpen,
  onClose,
  onSubmit,
  fieldLabels,
  agentOptions,
  onAgentsDropdownOpen,
  loadingAgents,
  chatAppToEdit,
}) => {
  const [formData, setFormData] = useState<ChatAppFormData>({
    name: "",
    description: "",
    selectedAgent: "",
    enableVoice: false,
  });
  const [errors, setErrors] = useState<any>({});

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
      [name]: undefined, // Clear error for the field being edited
    }));
  };

  const handleAgentChange = (selected: SingleValue<OptionType>) => {
    setFormData((prev) => ({
      ...prev,
      selectedAgent: selected?.value ?? "",
    }));
    setErrors((prev: any) => ({
      ...prev,
      selectedAgent: undefined, // Clear error for selectedAgent
    }));
  };

  const handleVoiceToggle = () => {
    setFormData((prev) => ({
      ...prev,
      enableVoice: !prev.enableVoice,
    }));
  };

  const validateForm = (data: ChatAppFormData) => {
    try {
      ChatAppFormSchema.parse(data);
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
      // showError("Please validate form details.");
      return false;
    }
  };

  const handleSubmit = () => {
    if (!validateForm(formData)) {
      return;
    }
    onSubmit(formData);
  };

  useEffect(() => {
    if (!isOpen && !chatAppToEdit) {
      setFormData({
        name: "",
        description: "",
        selectedAgent: "",
        enableVoice: false,
      });
      setErrors({});
    }
  }, [isOpen, chatAppToEdit]);

  // Populate form data when editing
  useEffect(() => {
    if (chatAppToEdit) {
      setFormData({
        name: chatAppToEdit.name,
        description: chatAppToEdit.description,
        selectedAgent: chatAppToEdit.agent_id ?? "",
        enableVoice: chatAppToEdit.voice_enabled ?? false,
      });
    }
  }, [chatAppToEdit]);

  return (
    <LargeModal
      isOpen={isOpen}
      onClose={onClose}
      onSubmit={handleSubmit}
      title={chatAppToEdit ? "Update Chat App" : "Create New Chat App"}
      actionButton="Submit"
      type="chatapp"
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
          <FormLabel htmlFor="name">
            {fieldLabels?.name} <span className="text-red-500">*</span>
          </FormLabel>
          <Input
            id="name"
            name="name"
            type="text"
            className="w-full border rounded-sm p-2"
            maxLength={AGENTIC_NAME_MAX_LENGTH}
            value={formData.name}
            onChange={handleChange}
            placeholder="Enter Chat Name"
            required
          />
          <p className="text-xs text-muted-foreground text-right pt-1 h-0">
            {formData.name.length}/{AGENTIC_NAME_MAX_LENGTH} characters
          </p>
          {errors.name && (
            <p className="text-red-500 text-sm">{errors.name.message}</p>
          )}
        </div>

        <div>
          <FormLabel htmlFor="description">
            {fieldLabels?.description}
          </FormLabel>
          <Textarea
            id="description"
            name="description"
            className="w-full border rounded-sm p-2"
            maxLength={AGENTIC_DESCRIPTION_MAX_LENGTH}
            value={formData.description}
            onChange={handleChange}
            placeholder="Write description"
            rows={3}
          />
          <p className="text-xs text-muted-foreground text-right p-0.5 h-0">
            {formData.description.length}/{AGENTIC_DESCRIPTION_MAX_LENGTH}{" "}
            characters
          </p>
          {errors.description && (
            <p className="text-red-500 text-sm">{errors.description.message}</p>
          )}
        </div>

        <div>
          <FormLabel htmlFor="selectedAgent">
            {fieldLabels?.selectedAgent} <span className="text-red-500">*</span>
          </FormLabel>
          <Select
            options={agentOptions}
            value={agentOptions.find(
              (option) => option.value === formData.selectedAgent,
            )}
            onChange={handleAgentChange}
            className="text-sm"
            styles={customStylesChatApp}
            onMenuOpen={onAgentsDropdownOpen}
            isLoading={loadingAgents}
            placeholder="Select an agent"
            isSearchable
          />
          {errors.selectedAgent && (
            <p className="text-red-500 text-sm">
              {errors.selectedAgent.message}
            </p>
          )}
        </div>

        <div>
          <div className="flex items-center justify-between">
            <FormLabel htmlFor="enableVoice">
              {fieldLabels?.enableVoice}
            </FormLabel>
            <button
              type="button"
              onClick={handleVoiceToggle}
              className={`relative inline-flex items-center h-6 rounded-full w-11 transition-colors duration-200 ${
                formData.enableVoice ? "bg-blue-600" : "bg-gray-300"
              }`}
            >
              <span
                className={`inline-block w-4 h-4 transform bg-white rounded-full transition-transform duration-200 ${
                  formData.enableVoice ? "translate-x-6" : "translate-x-1"
                }`}
              />
            </button>
          </div>
        </div>
      </form>
    </LargeModal>
  );
};

export default CreateChatAppForm;
