/**
 * The `ResetPassword` function in this JavaScript React code handles the form submission for resetting
 * a user's password with validation using Zod schema.
 */
"use client";
import React, { useState } from "react";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import Image from "next/image";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import bgimage from "@/assets/images/signup-bg.png";
import icon from "@/assets/icons/BrandLogo.svg";
import {
  forgotPassword,
  forgotPasswordInvite,
  resetPassword,
} from "@/api/login";
import { showError, showSuccess } from "@/utils/toastUtils";
import { useRouter, useSearchParams } from "next/navigation";
import { useUser } from "@/context_api/userContext";

// Define your form schema using Zod
const formSchema = z
  .object({
    newPassword: z
      .string()
      .min(10, { message: "New password must be at least 10 characters." }),
    confirmPassword: z
      .string()
      .min(10, { message: "Confirm password must match the new password." }),
  })
  .refine((data) => data.newPassword === data.confirmPassword, {
    message: "Passwords must match.",
    path: ["confirmPassword"],
  });
const emailSchema = z.object({
  email: z
    .string()
    .nonempty("Email is required")
    .email("Invalid email address")
    .transform((val) => val.toLowerCase()),
});

const ForgotPassword = () => {
  const [loader, setLoader] = useState(false);
  const { user } = useUser();
  const [sentInvite, setSentInvite] = useState(false);
  const params = useSearchParams();
  const vtoken = params.get("vtoken") || null;
  const form = useForm({
    resolver: zodResolver(formSchema),
    defaultValues: {
      newPassword: "",
      confirmPassword: "",
    },
  });

  const emailForm = useForm({
    resolver: zodResolver(emailSchema),
    defaultValues: {
      email: "",
    },
  });

  const router = useRouter();
  async function sendResetLink(values) {
    setLoader(true);
    console.log(values);
    try {
      const response = await forgotPasswordInvite({ email: values.email });

      if (response.status === 200) {
        setLoader(false);
        showSuccess(`${response.data?.message}`);
        setSentInvite(true);
      }
    } catch (error) {
      setLoader(false);
      showError(`${error?.response?.data?.detail}`);
      router.push("login");
      console.error("Failed to reset password:", error);
    }
  }
  // Handle form submission
  async function onSubmit(values) {
    setLoader(true);
    // Make API call here
    const payload = {
      vtoken,
      password: values.newPassword,
    };
    try {
      const response = await forgotPassword(payload);

      if (response.status === 200) {
        setLoader(false);
        showSuccess(`${response.data?.message}`);
        router.push(`login`);
      }
    } catch (error) {
      setLoader(false);
      showError(`${error?.response?.data?.detail}`);
      console.error("Failed to reset password:", error);
    }
  }

  return (
    <div className="flex flex-row w-full h-full">
      <div className="bg-white flex flex-col justify-evenly loginScreen w-1/2">
        <div className="flex justify-between items-center w-full absolute top-10">
          <Image src={icon} alt="amplify-icon" className="ms-24" />
        </div>
        {vtoken && (
          <>
            <div className="font-medium text-3.2xl text-center">
              Reset Password
            </div>
            <Form {...form}>
              <form
                onSubmit={form.handleSubmit(onSubmit)}
                className="space-y-4 px-24"
              >
                {/* New Password */}
                <FormField
                  control={form.control}
                  name="newPassword"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>New Password</FormLabel>
                      <FormControl>
                        <Input
                          type="password"
                          placeholder="At least 10 characters"
                          {...field}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                {/* Confirm Password */}
                <FormField
                  control={form.control}
                  name="confirmPassword"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Confirm Password</FormLabel>
                      <FormControl>
                        <Input
                          type="password"
                          placeholder="Re-enter new password"
                          {...field}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <Button
                  type="submit"
                  className="w-full bg-blue-600 text-white"
                  isLoading={loader}
                >
                  Reset Password
                </Button>
              </form>
            </Form>
          </>
        )}
        {!vtoken && (
          <div className="text-center">
            <div className="font-medium text-3.2xl text-center mb-4">
              Send Reset Password Link
            </div>
            {!sentInvite && (
              <Form {...emailForm}>
                <form
                  onSubmit={emailForm.handleSubmit(sendResetLink)}
                  className="space-y-4 px-24"
                >
                  <FormField
                    control={emailForm.control}
                    name="email"
                    render={({ field }) => (
                      <FormItem className="text-start">
                        <FormLabel>Email</FormLabel>
                        <FormControl>
                          <Input
                            type="email"
                            placeholder="Enter email"
                            {...field}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <Button
                    type="submit"
                    className="w-full bg-blue-600 text-white"
                    isLoading={loader}
                  >
                    Send Link
                  </Button>
                </form>
              </Form>
            )}

            {sentInvite && (
              <div className="text-lg">
                Link to reset password has been shared on your email.
              </div>
            )}
          </div>
        )}
      </div>
      <div className="w-1/2 relative loginScreen">
        <div className="h-screen">
          <Image
            src={bgimage}
            alt="signup page image"
            className="h-full w-full"
          />
        </div>
        <div className="absolute top-1/2 text-white ms-[74px]">
          <div className="font-medium text-6.5xl">
            Futurize your Enterprise Data
          </div>
          <div className="font-normal text-base">
            Hunt powerful insights by unifying unstructured and structured data.
          </div>
        </div>
      </div>
    </div>
  );
};

export default ForgotPassword;
