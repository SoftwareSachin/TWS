import { z } from "zod";
import { ERROR_MSG } from "@/lib/error-msg";

export const UserSchema = z.object({
  first_name: z.string().min(1, "First name is required"),
  last_name: z.string().min(1, "Last name is required"),
  email: z.string().email("Invalid email"),
  role: z.string().min(1, "Role is required"),
  organization_id: z.string().min(1, "Organization is required"),
});

export const AddUsersFormSchema = z.object({
  users: z.array(UserSchema).min(1),
});

export interface Users {
  first_name: string;
  last_name: string;
  email: string;
  role: string;
  organization_id: string;
}

export type AddUsersFormValues = z.infer<typeof AddUsersFormSchema>;

export const UserLoginSchema = z.object({
  username: z
    .string()
    .min(1, { message: ERROR_MSG.REQUIRED_EMAIL })
    .email({ message: ERROR_MSG.INVALID_EMAIL }),
  password: z
    .string()
    .min(1, { message: ERROR_MSG.VALIDATION_MSG_PASSWORD_REQUIRED }),
  // .min(10, { message: ERROR_MSG.PASSWORD_LENGTH }),
});

export interface LoginResponse {
  access_token: string;
  refresh_token: string;
  jwt_token?: string;
  FirstLogin?: boolean;
  isMfaEnabled?: boolean;
  QrCode?: string;
  secondFactorAuthenticationToken?: string;
  data?: any;
  status?: number;
  SecondFactorAuthentication?: {
    SecondFactorAuthenticationToken: string;
  } | null;
}

export type UserLoginType = z.infer<typeof UserLoginSchema>;
