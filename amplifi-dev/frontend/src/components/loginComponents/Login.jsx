"use client";
import React, { useEffect, useState } from "react";
import bgimage from "@/assets/images/signup-bg.png";
import icon from "@/assets/icons/BrandLogo.svg";
import largeIcon from "@/assets/icons/logo-big.svg";
import largeIcon2 from "@/assets/icons/logo-big-2.svg";
import Image from "next/image";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { CheckIcon } from "@radix-ui/react-icons"; // Import Radix checkmark icon
import { Button } from "../ui/button";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "../ui/form";
import { Input } from "../ui/input";
import { Checkbox } from "@radix-ui/react-checkbox";
import { useRouter } from "next/navigation";
import { loginApi } from "@/api/login";
import Cookies from "universal-cookie";
import { showError, showSuccess } from "@/utils/toastUtils";
import { constants } from "@/lib/constants";
import { setCookie } from "@/utils/cookieHelper";
import {
  identifyUserFromObject,
  hashString,
  captureEvent,
} from "@/utils/posthogUtils";
import { decodeToken } from "@/components/utility/decodeJwtToken";

// Define your form schema using Zod
const formSchema = z.object({
  username: z.string().email({ message: "Invalid email address." }),
  password: z
    .string()
    .min(1, { message: "Password must be at least 10 characters." }),
  // agree: z.boolean().refine((value) => value === true, {
  //   message: "You must agree to the privacy policy and terms of service.",
  // }),
});

const Login = () => {
  const [showLoginScreen, setShowLoginScreen] = useState(false);
  const router = useRouter();
  const cookies = new Cookies();
  const form = useForm({
    resolver: zodResolver(formSchema),
    defaultValues: {
      username: "",
      password: "",
      // agree: false,
    },
  });

  // Handle form submission
  async function onSubmit(values) {
    try {
      const response = await loginApi(values);

      if (response?.status === 200) {
        setCookie(constants.AUTH_TOKEN, response?.data?.access_token);

        // Track login success event
        const userDetails = decodeToken(response?.data?.access_token);
        if (userDetails) {
          identifyUserFromObject(userDetails);
          captureEvent("login_success", {
            user_id_hash: hashString(userDetails?.clientId || ""),
            login_method: "email_password",
            org_id_hash: hashString(userDetails?.orgId || ""),
            description: "Login completed",
          });
        }

        router.push("/set-up");

        // cookies.set("email", values?.email);
        // cookies.set("password", values?.password);
        // cookies.set("role", response?.data?.data?.user?.role?.name);
        // cookies.set("refreshtoken", response?.data?.data?.refresh_token);
        // cookies.set("id", response?.data?.data?.user?.id);
      }
    } catch (error) {
      showError(`${error.response.data.detail}`);
    }
  }

  // Effect to switch screens after 1000ms
  useEffect(() => {
    const timer = setTimeout(() => {
      setShowLoginScreen(true); // Show login screen after 1000ms
    }, 10000);

    return () => clearTimeout(timer); // Cleanup timer on unmount
  }, []);

  return (
    <div className="flex flex-row w-full h-full">
      {showLoginScreen ? ( // Conditional rendering based on state
        <>
          <div className="w-1/2 flex flex-col justify-evenly loginScreen">
            <div className="flex justify-between items-center w-full">
              <Image src={icon} alt="amplify-icon" className="ms-24" />
              <div className="text-sm font-medium">
                Have an account?{" "}
                <span className="underline text-blue-700 me-14">Signin</span>
              </div>
            </div>
            <div className="w-2/3 ms-24">
              <div className="font-medium text-3.2xl mb-10">
                Create an <span className="font-bold">amplifi</span> account
              </div>
              <Form {...form}>
                <form
                  onSubmit={form.handleSubmit(onSubmit)}
                  className="space-y-4"
                >
                  <FormField
                    control={form.control}
                    name="username"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Email</FormLabel>
                        <FormControl>
                          <Input placeholder="john@jill.com" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="password"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Password</FormLabel>
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
                  <Button type="submit" className="w-full bg-blue-600 !mt-10">
                    Get Started
                  </Button>
                  <FormField
                    control={form.control}
                    name="agree"
                    render={({ field }) => (
                      <FormItem className="flex flex-col items-start">
                        {" "}
                        {/* Changed to flex-col for vertical stacking */}
                        <div className="flex items-center">
                          <FormControl>
                            <Checkbox
                              checked={field.value}
                              onCheckedChange={field.onChange}
                              className="w-5 h-5 border-2 border-gray-400 rounded-sm flex items-center justify-center"
                            >
                              {field.value && <CheckIcon className="w-4 h-4" />}{" "}
                              {/* Show checkmark */}
                            </Checkbox>
                          </FormControl>
                          <FormDescription
                            className="w-2/3 ml-2 font-normal text-sm text-black-10"
                            onClick={() => field.onChange(!field.value)}
                          >
                            By signing up, I agree to the amplify{" "}
                            <span className="font-medium">
                              Privacy Policy & Terms of Service
                            </span>
                          </FormDescription>
                        </div>
                        {/* Error message on the next line */}
                        <FormMessage className="mt-2" />{" "}
                        {/* Add margin-top to give some space from the checkbox */}
                      </FormItem>
                    )}
                  />
                </form>
              </Form>
            </div>
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
        </>
      ) : (
        <div className="startScreen w-full h-screen flex flex-col items-center justify-around gap-3">
          <Image src={largeIcon} alt="amplify-icon" />
          <Image src={largeIcon2} alt="amplify-icon" />
        </div>
      )}
    </div>
  );
};

export default Login;
