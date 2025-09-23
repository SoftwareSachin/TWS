import { z } from "zod";
import {
  AGENTIC_NAME_MAX_LENGTH,
  AGENTIC_DESCRIPTION_MAX_LENGTH,
  ALLOWED_NAME_REGEX,
  RESERVED_WORDS,
} from "@/lib/file-constants";

export const ExternalMcpFormSchema = z.object({
  mcpName: z
    .string()
    .min(1, "MCP Name is required")
    .max(AGENTIC_NAME_MAX_LENGTH)
    .refine((val) => ALLOWED_NAME_REGEX.test(val), {
      message:
        "Name must start with a letter and use only letters, numbers, _ or -",
    })
    .refine((val) => !RESERVED_WORDS.includes(val.toLowerCase()), {
      message: "Name cannot be a reserved word (e.g., select, workspace)",
    }),
  description: z
    .string()
    .max(AGENTIC_DESCRIPTION_MAX_LENGTH)
    .min(1, "MCP Description is required"),
  configCode: z
    .string()
    .min(1, "Configuration JSON is required")
    .refine(
      (val) => {
        try {
          const parsed = JSON.parse(val);
          return (
            typeof parsed === "object" &&
            parsed !== null &&
            !Array.isArray(parsed)
          );
        } catch {
          return false;
        }
      },
      {
        message: "Must be valid JSON and a valid object",
      },
    ),
});
