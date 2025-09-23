import { z } from "zod";

// Define validation schema using Zod
export const GrooveFormSchema = z.object({
  sourceType: z.string().min(1, "Source type is required"),
  sourceName: z.string().min(1, "Source name is required"),
  grooveApiKey: z.string().min(1, "Groove API key is required"),
  autoDetectionEnabled: z.boolean().default(false),
  monitoringFrequencyMinutes: z
    .number()
    .min(1, "Must be at least 1 minute")
    .max(1440, "Must be at most 1440 minutes (24 hours)")
    .default(30),
  ticketBatchSize: z
    .number()
    .min(1, "Must be at least 1")
    .max(100, "Must be at most 100")
    .default(10),
  reIngestUpdatedTickets: z.boolean().default(false),
});

export type GrooveFormData = z.infer<typeof GrooveFormSchema>;
