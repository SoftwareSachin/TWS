import { ALLOWED_NAME_REGEX, RESERVED_WORDS } from "@/lib/file-constants";
import { z } from "zod";

export const AgentFormSchema = z.object({
  agentName: z
    .string()
    .min(1, "Agent Name is required")
    .refine((val) => ALLOWED_NAME_REGEX.test(val), {
      message:
        "Name must start with a letter and use only letters, numbers, _ or -",
    })
    .refine((val) => !RESERVED_WORDS.includes(val.toLowerCase()), {
      message: "Name cannot be a reserved word (e.g., select, workspace)",
    }),
  description: z.string().optional(),
  workspaceTool: z
    .array(z.string().min(1))
    .min(1, "At least one tool is required"),
  llmProvider: z.string().min(1, "LLM Provider is required"),
  prompt_instructions: z.string().optional(),
});
