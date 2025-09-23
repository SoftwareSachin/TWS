export interface FormValidation {
  success: boolean;
  errors: FormErrors;
}

export interface FormErrors {
  [fieldName: string | number]: string;
}
