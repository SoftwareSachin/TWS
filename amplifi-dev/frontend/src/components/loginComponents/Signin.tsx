"use client";
import React, { useEffect, useState } from "react";
import bgimage from "@/assets/images/signup-bg.png";
import icon from "@/assets/icons/BrandLogo.svg";
import Image from "next/image";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm, SubmitHandler } from "react-hook-form";
import { Button } from "../ui/button";
import { useUser } from "@/context_api/userContext";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "../ui/form";
import { Input } from "../ui/input";
import { useRouter, useSearchParams } from "next/navigation";
import { checkForFirstTime, verifyMailId, loginApi } from "@/api/login";
import { showError, showSuccess } from "@/utils/toastUtils";
import { ForgotPasswordLink } from "@/components/loginComponents/ForgotPassword";
import { decodeToken } from "@/components/utility/decodeJwtToken";
import { constants } from "@/lib/constants";
import { removeCookie, setCookie } from "@/utils/cookieHelper";
import PageLoader from "@/components/ui/pageLoader";
import { LoginResponse, UserLoginSchema, UserLoginType } from "@/types/Users";
import { ERROR_MSG } from "@/lib/error-msg";
import {
  identifyUserFromObject,
  hashString,
  captureEvent,
} from "@/utils/posthogUtils";

const Signin: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [pageLoading, setPageLoading] = useState(true);
  const router = useRouter();
  const { setUser } = useUser();
  const searchParams = useSearchParams();

  const form = useForm<UserLoginType>({
    resolver: zodResolver(UserLoginSchema),
    defaultValues: {
      username: "",
      password: "",
    },
  });

  useEffect(() => {
    if (searchParams.size && searchParams.get("vtoken")) {
      const vtoken = searchParams.get("vtoken");
      const payload = { vtoken };
      if (vtoken) {
        loginWithLoginRadius(payload, "verifyEmail");
      }
    } else {
      setPageLoading(false);
    }
  }, [searchParams]);

  const redirectToDashboard = (jwtToken: string) => {
    const userDetails = decodeToken(jwtToken);
    // console
    setUser(userDetails);
    localStorage.setItem(constants.USER, JSON.stringify(userDetails));

    // Track login success event
    identifyUserFromObject(userDetails);
    captureEvent("login_success", {
      user_id_hash: hashString(userDetails?.clientId || ""),
      login_method: "email_password",
      org_id_hash: hashString(userDetails?.orgId || ""),
      description: "Login completed",
    });

    const redirectPath =
      window.location.hostname === process.env.NEXT_PUBLIC_CHAT_HOST_NAME
        ? `/chatapp`
        : `/workspace?id=${userDetails.clientId}`;
    router.push(redirectPath);
    showSuccess("Login successful");
  };

  const mfaLogin: SubmitHandler<UserLoginType> = async (values) => {
    setLoading(true);
    removeCookie(constants.AUTH_TOKEN);
    removeCookie(constants.REFRESH_TOKEN);
    removeCookie(constants.JWT_TOKEN);
    localStorage.removeItem(constants.USER);
    try {
      const response: LoginResponse = await loginApi(values);
      setLoading(false);
      // Removed sensitive login response logging for security
      if (
        !response.isMfaEnabled &&
        response.FirstLogin &&
        !response.jwt_token
      ) {
        sessionStorage.setItem(constants.AUTH_TOKEN, response.access_token);
        router.push("/reset-password");
        return;
      }
      if (!response.isMfaEnabled) {
        setCookie(constants.AUTH_TOKEN, response.access_token);
        setCookie(constants.REFRESH_TOKEN, response.refresh_token);
        setCookie(constants.JWT_TOKEN, response.jwt_token || "");

        redirectToDashboard(response.jwt_token || "");
      } else {
        sessionStorage.setItem(constants.AUTH_TOKEN, response.access_token);
        if (response.QrCode && response.QrCode != "null")
          sessionStorage.setItem("qrCode", response.QrCode);
        if (
          response.secondFactorAuthenticationToken &&
          response.secondFactorAuthenticationToken != "null"
        )
          sessionStorage.setItem(
            "secondFactorAuthenticationToken",
            response.secondFactorAuthenticationToken,
          );
        router.push("/authenticator-setup");
      }
    } catch (error: any) {
      setLoading(false);
      console.log("error in login", error);
      showError(error?.response?.data?.detail || "Login failed");
    }
  };

  const loginWithLoginRadius = async (
    payload: any,
    action: "verifyEmail" | "login",
  ) => {
    setLoading(true);
    let firstTimeUser = false;
    let access_token = "";
    removeCookie(constants.AUTH_TOKEN);
    removeCookie(constants.REFRESH_TOKEN);
    removeCookie(constants.JWT_TOKEN);

    try {
      const response = await verifyMailId(payload);

      if (response?.status === 200) {
        access_token = response?.data?.access_token;
        setCookie(constants.AUTH_TOKEN, access_token);
        // setCookie(constants.REFRESH_TOKEN, refresh_token);
        const res = await checkForFirstTime({ email: response?.data?.email });
        firstTimeUser = res?.data?.IsPasswordResetRequired;
        if (firstTimeUser) {
          removeCookie(constants.AUTH_TOKEN);
          sessionStorage.setItem(constants.AUTH_TOKEN, access_token);
          router.push("/reset-password");
          return;
        }
      }
    } catch (error: any) {
      setLoading(false);
      setPageLoading(false);
      if (action === "verifyEmail") {
        console.log("error in verify email", error);
        showError(error?.response?.data?.detail || "Login failed");
        // showError(ERROR_MSG.EMAIL_VERIFICATION);
      } else {
        showError(error?.response?.data?.detail || "Login failed");
      }
    }
  };

  return (
    <>
      {pageLoading && <PageLoader />}
      {!pageLoading && (
        <div className="flex flex-row w-full h-full">
          <div className="w-1/2 flex flex-col justify-evenly loginScreen">
            <div className="flex justify-between items-center w-full">
              <Image src={icon} alt="amplify-icon" className="ms-24" />
            </div>
            <div className="w-2/3 ms-24">
              <div className="font-medium text-3.2xl mb-10">
                Login to your <span className="font-bold">amplifi</span> account
              </div>
              <Form {...form}>
                <form
                  onSubmit={form.handleSubmit(mfaLogin)}
                  className="space-y-4"
                >
                  <FormField
                    control={form.control}
                    name="username"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Email</FormLabel>
                        <FormControl>
                          <Input
                            placeholder="Enter your email"
                            {...field}
                            onChange={(e) =>
                              field.onChange(e.target.value.trim())
                            }
                          />
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
                            placeholder="Enter your password"
                            {...field}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <ForgotPasswordLink />
                  <Button
                    type="submit"
                    className="w-full bg-blue-600 !mt-10 flex items-center justify-center"
                    isLoading={loading}
                  >
                    Login
                  </Button>
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
        </div>
      )}
    </>
  );
};

export default Signin;
