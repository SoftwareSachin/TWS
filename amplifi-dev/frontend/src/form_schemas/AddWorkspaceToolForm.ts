import { z } from "zod";
import {
  AGENTIC_NAME_MAX_LENGTH,
  AGENTIC_DESCRIPTION_MAX_LENGTH,
  ALLOWED_NAME_REGEX,
  RESERVED_WORDS,
} from "@/lib/file-constants";

export const AddWorkspaceToolFormSchema = (
  isAgent: boolean,
  showDataset: boolean,
) => {
  return z
    .object({
      agentName: z
        .string()
        .min(1, isAgent ? "Agent Name is required" : "Tool Name is required")
        .max(
          AGENTIC_NAME_MAX_LENGTH,
          isAgent
            ? `Agent Name must not exceed ${AGENTIC_NAME_MAX_LENGTH} characters`
            : `Tool Name must not exceed ${AGENTIC_NAME_MAX_LENGTH} characters`,
        )
        .refine((val) => ALLOWED_NAME_REGEX.test(val), {
          message:
            "Name must start with a letter and use only letters, numbers, _ or -",
        })
        .refine((val) => !RESERVED_WORDS.includes(val.toLowerCase()), {
          message: "Name cannot be a reserved word (e.g., select, workspace)",
        }),
      description: z
        .string()
        .max(
          AGENTIC_DESCRIPTION_MAX_LENGTH,
          `Description must not exceed ${AGENTIC_DESCRIPTION_MAX_LENGTH} characters`,
        )
        .optional(),
      workspaceTool: z.enum(["MCP", "System Tool"], {
        errorMap: () => ({
          message: "Please select a workspace tool (MCP or System Tool)",
        }),
      }),
      selectedMCPs: z.array(z.string()).optional(),
      selectedSystemTools: z.array(z.string()).optional(),
      selectedDatasets: z.array(z.string()).optional(),
      prompt_instructions: isAgent
        ? z.string().optional()
        : z.string().optional(),
    })
    .refine(
      (data) =>
        data.workspaceTool !== "MCP" ||
        (data.selectedMCPs && data.selectedMCPs.length > 0),
      {
        message: "At least one MCP must be selected",
        path: ["selectedMCPs"],
      },
    )
    .refine(
      (data) =>
        data.workspaceTool !== "System Tool" ||
        (data.selectedSystemTools && data.selectedSystemTools.length > 0),
      {
        message: "At least one System Tool must be selected",
        path: ["selectedSystemTools"],
      },
    )
    .refine(
      (data) =>
        !showDataset ||
        data.workspaceTool !== "System Tool" ||
        !data.selectedSystemTools ||
        data.selectedSystemTools.length === 0 ||
        (data.selectedDatasets && data.selectedDatasets.length > 0),
      {
        message:
          "At least one dataset must be selected when system tools are selected",
        path: ["selectedDatasets"],
      },
    );
};
