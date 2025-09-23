import { z } from "zod";

export interface PGSourceConnector {
  source_type: string;
  host: string;
  port: number | string;
  database_name: string;
  username: string;
  password: string;
  id?: string;
  ssl_mode: "disabled" | "required";
}

// PG Source Schema
export const PostgresFormSchema = z.object({
  database_name: z.string().min(1, "Database is required"),
  host: z.string().min(1, "Hostname is required"),
  port: z
    .string() // Start with string to handle input type="number"
    .min(1, { message: "Port is required" })
    .transform((val) => {
      const parsed = parseInt(val, 10); // Transform to number
      return isNaN(parsed) ? 0 : parsed; // If the parsed value is invalid, set it to 0
    })
    .refine((val) => val >= 1 && val <= 65535, {
      // Ensure the value is within valid port range
      message: "Invalid port number",
    }),
  username: z.string().min(1, "Username is required"),
  password: z.string().min(1, "Password is required"),
  ssl_mode: z.enum(["disabled", "required"]).optional(),
});
