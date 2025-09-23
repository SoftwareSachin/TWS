import { FormErrors, FormValidation } from "@/types/FormValidation";
import { ZodEffects, ZodError } from "zod";

export const validateForm = (
  formSchema: ZodEffects<any>,
  formData: any,
): FormValidation => {
  const formValidate = formSchema.safeParse(formData);
  const formValidationRes: FormValidation = {
    success: true,
    errors: {},
  };
  if (formValidate.success) return formValidationRes;
  formValidationRes.success = false;
  formValidationRes.errors = fetchFormErrors(formValidate.error);
  return formValidationRes;
};

const fetchFormErrors = (validationRes: ZodError<any>): FormErrors => {
  return Object.fromEntries(
    validationRes.issues
      .filter((issue) => typeof issue.path[0] === "string") // filter out any non-string paths
      .map((issue) => [issue.path[0] as string, issue.message]),
  );
};
