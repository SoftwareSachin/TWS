import { z } from "zod";
import {
  ALLOWED_NAME_REGEX,
  AGENTIC_DESCRIPTION_MAX_LENGTH,
  AGENTIC_NAME_MAX_LENGTH,
  RESERVED_WORDS,
} from "@/lib/file-constants";

export const ChatAppFormSchema = z.object({
  name: z
    .string()
    .min(1, { message: "Chat Name is required" })
    .max(AGENTIC_NAME_MAX_LENGTH, {
      message: `Name must be less than ${AGENTIC_NAME_MAX_LENGTH} characters`,
    })
    .refine((val) => ALLOWED_NAME_REGEX.test(val), {
      message:
        "Chat Name must start with a letter and use only letters, numbers, _ or -",
    })
    .refine((val) => !RESERVED_WORDS.includes(val.toLowerCase()), {
      message: "Chat Name cannot be a reserved word (e.g., select, workspace)",
    }),
  description: z.string().max(AGENTIC_DESCRIPTION_MAX_LENGTH, {
    message: `Description must be less than ${AGENTIC_DESCRIPTION_MAX_LENGTH} characters`,
  }),
  selectedAgent: z.string().min(1, { message: "Please select an agent" }),
  enableVoice: z.boolean().optional(),
});
