import { z } from "zod";

enum DatasetFileOptions {
  selectFile = "selectFile",
  ingestAll = "ingestAll",
}
const NAME_MAX_LENGTH = 25;
const RESERVED_WORDS = ["select", "from", "workspace", "dataset"];
const ALLOWED_NAME_REGEX = /^[a-zA-Z][a-zA-Z0-9 _-]*$/;

export const DatasetSchema = z
  .object({
    name: z
      .string()
      .min(1, { message: "Name is required" })
      .max(NAME_MAX_LENGTH, {
        message: `Name must be less than ${NAME_MAX_LENGTH} characters`,
      })
      .refine((val) => ALLOWED_NAME_REGEX.test(val), {
        message:
          "Name must start with a letter and use only letters, numbers, _ or -",
      })
      .refine((val) => !RESERVED_WORDS.includes(val.toLowerCase()), {
        message: "Name cannot be a reserved word (e.g., select, workspace)",
      }),
    description: z.string().nullable(),
    fileOption: z.nativeEnum(DatasetFileOptions),
    selectedFile: z.array(z.string()).optional(),
    sourceId: z.string().nullable().optional(),
  })
  .superRefine((data, ctx) => {
    if (
      data.fileOption === DatasetFileOptions.selectFile &&
      !data.selectedFile?.length
    ) {
      ctx.addIssue({
        path: ["selectedFile"],
        code: z.ZodIssueCode.custom,
        message: "Please select at least 1 file",
      });
    }
    if (data.fileOption === DatasetFileOptions.ingestAll && !data.sourceId) {
      ctx.addIssue({
        path: ["sourceId"],
        code: z.ZodIssueCode.custom,
        message: "Please select a datasource",
      });
    }
  });
