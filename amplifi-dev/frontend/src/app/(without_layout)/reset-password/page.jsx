/**
 * The `ResetPassword` function in this JavaScript React code handles the form submission for resetting
 * a user's password with validation using Zod schema.
 */
"use client";
import React, { useEffect, useState } from "react";
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
import { showError, showSuccess } from "@/utils/toastUtils";
import { useRouter } from "next/navigation";
import { constants } from "@/lib/constants";
import { removeCookie } from "@/utils/cookieHelper";
import { resetPassword } from "@/api/login";

// Define your form schema using Zod

const formSchema = z
  .object({
    oldPassword: z.string().min(6, {
      message:
        "Old password must be at least 6 characters. Provided in the ema",
    }),
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

const ResetPassword = () => {
  const [loader, setLoader] = useState(false);
  const [formLoading, setFormLoading] = useState(false);

  const form = useForm({
    resolver: zodResolver(formSchema),
    defaultValues: {
      oldPassword: "",
      newPassword: "",
      confirmPassword: "",
    },
  });

  const router = useRouter();

  useEffect(() => {
    if (typeof window !== "undefined") {
      const token = sessionStorage.getItem(constants.AUTH_TOKEN);
      if (!token) {
        router.push("/login");
      } else {
        setLoader(true);
      }
    }
  }, []);

  // Handle form submission
  async function onSubmit(values) {
    setFormLoading(true);
    // Make API call here
    const payload = {
      oldPassword: values.oldPassword,
      newPassword: values.newPassword,
    };

    try {
      const response = await resetPassword(
        payload,
        sessionStorage.getItem(constants.AUTH_TOKEN),
      );
      if (response.status === 200) {
        setFormLoading(false);
        showSuccess(`${response.data?.message}`);
        removeCookie(constants.AUTH_TOKEN);
        removeCookie(constants.REFRESH_TOKEN);
        removeCookie(constants.JWT_TOKEN);
        sessionStorage.removeItem(constants.AUTH_TOKEN);
        router.push(`login`);
      }
    } catch (error) {
      setFormLoading(false);
      showError(`${error?.response?.data?.detail}`);
      console.error("Failed to reset password:", error);
    } finally {
      setFormLoading(false);
    }
  }

  if (loader) {
    return (
      <div className="flex flex-row w-full h-full">
        <div className="bg-white flex flex-col justify-evenly loginScreen w-1/2">
          <div className="flex justify-between items-center w-full">
            <Image src={icon} alt="amplify-icon" className="ms-24" />
          </div>
          <div className="font-medium text-3.2xl   text-center">
            Reset Password
          </div>
          <Form {...form}>
            <form
              onSubmit={form.handleSubmit(onSubmit)}
              className="space-y-4 px-24"
            >
              {/* Old Password */}
              <FormField
                control={form.control}
                name="oldPassword"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Old Password</FormLabel>
                    <FormControl>
                      <Input
                        type="password"
                        placeholder="Enter old password Provided in Email"
                        {...field}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

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
                isLoading={formLoading}
              >
                Reset Password
              </Button>
            </form>
          </Form>
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
              Hunt powerful insights by unifying unstructured and structured
              data.
            </div>
          </div>
        </div>
      </div>
    );
  }
};

export default ResetPassword;
