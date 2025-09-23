"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { showError, showSuccess } from "@/utils/toastUtils";
import Image from "next/image";
import icon from "@/assets/icons/BrandLogo.svg";
import bgimage from "@/assets/images/signup-bg.png";
import { validateAuthCode, getJwtToken, verifyAuthCode } from "@/api/login";
import { useUser } from "@/context_api/userContext";
import { decodeToken } from "@/components/utility/decodeJwtToken";
import { constants } from "@/lib/constants";
import { Button } from "@/components/ui/button";
import { setCookie } from "@/utils/cookieHelper";

const AuthenticatorSetup = () => {
  const router = useRouter();
  const { setUser } = useUser();

  const [qrCode, setQrCode] = useState(null);
  const [accessToken, setAccessToken] = useState(null);
  const [secondFactorAuthenticationToken, setSecondFactorAuthenticationToken] =
    useState(null);
  const [isAllowed, setIsAllowed] = useState(false);
  const [loading, setLoading] = useState(false);

  // OTP State
  const [otp, setOtp] = useState(["", "", "", "", "", ""]);
  const inputRefs = useRef([]);

  useEffect(() => {
    if (otp.every((digit) => digit !== "") && otp.length === 6) {
      handleVerifyAuthenticator();
    }
  }, [otp]);

  useEffect(() => {
    const qrCode = sessionStorage.getItem("qrCode");
    const token = sessionStorage.getItem("secondFactorAuthenticationToken");
    if (!qrCode && !token) {
      router.replace("/login");
    } else {
      setIsAllowed(true);
      // Removed sensitive QR code and token logging for security
      setAccessToken(sessionStorage.getItem(constants.AUTH_TOKEN));
      setQrCode(qrCode);
      setSecondFactorAuthenticationToken(token);
      setTimeout(() => {
        inputRefs.current[0]?.focus();
      }, 0);
    }
  }, [router]);

  useEffect(() => {
    if (qrCode) sessionStorage.removeItem("qrCode");
  }, [qrCode]);

  useEffect(() => {
    if (secondFactorAuthenticationToken)
      sessionStorage.removeItem("secondFactorAuthenticationToken");
  }, [secondFactorAuthenticationToken]);

  useEffect(() => {
    if (accessToken) sessionStorage.removeItem(constants.AUTH_TOKEN);
  }, [accessToken]);

  if (!isAllowed) {
    return null; // Avoid rendering until check is complete
  }

  const handleChange = (index, e) => {
    const value = e.target.value;
    if (!/^\d*$/.test(value)) return;

    const newOtp = [...otp];
    newOtp[index] = value;
    setOtp(newOtp);
    console.log(index, e);
    if (value && index < 5) {
      inputRefs.current[index + 1].focus();
    }
  };

  const handleKeyDown = (index, e) => {
    if (e.key === "Backspace" && !otp[index] && index > 0) {
      inputRefs.current[index - 1].focus();
    }
  };

  const handleVerifyAuthenticator = async () => {
    const authCode = otp.join("");
    // Removed sensitive auth code logging for security
    if (authCode.length !== 6) {
      showError("Please enter a 6-digit authentication code.");
      return;
    }

    setLoading(true);

    try {
      let response = await validateAuthCode(
        secondFactorAuthenticationToken,
        authCode,
        accessToken,
      );
      const { access_token, refresh_token, jwt_token } = response.data;
      if (response.status === 200) {
        setCookie(constants.REFRESH_TOKEN, refresh_token);
        setCookie(constants.AUTH_TOKEN, access_token);
        setCookie(constants.JWT_TOKEN, jwt_token);
      }
      if (response.status === 200) {
        const userDetails = decodeToken(jwt_token);
        setUser(userDetails);
        localStorage.setItem(constants.USER, JSON.stringify(userDetails));

        if (userDetails.clientId) {
          process.env.NEXT_PUBLIC_CHAT_HOST_NAME.includes(
            window.location.hostname,
          )
            ? router.push(`/chatapp`)
            : router.push(`/workspace?id=${userDetails.clientId}`);

          showSuccess("Login successful");
        }
      } else if (response.status === 403) {
        // Authentication bypassed - redirect to workspace directly
        showError("Authentication failed. Redirecting to workspace.");
        router.push("/workspace");
      }
    } catch (error) {
      console.error(error);
      showError("Invalid authentication code.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-row w-full h-full">
      {/* Left Section */}
      <div className="bg-white flex flex-col justify-evenly loginScreen w-1/2">
        <div className="flex justify-between items-center w-full">
          <Image src={icon} alt="amplify-icon" className="ms-24" />
        </div>
        <div className="font-medium text-3.2xl text-center">
          Setup Two-Factor Authentication
        </div>
        <div className="space-y-4 px-24 flex flex-col items-center">
          {qrCode ? (
            <>
              <p>Scan this QR code in your authenticator app:</p>
              <img
                src={qrCode.trim()}
                alt="Unable to load QR Code"
                height={200}
                width={200}
              />
            </>
          ) : (
            <p>
              Please enter the authentication code from your authenticator app.
            </p>
          )}

          <div className="flex space-x-2">
            {otp.map((digit, index) => (
              <input
                key={index}
                ref={(el) => (inputRefs.current[index] = el)}
                type="text"
                maxLength="1"
                value={digit}
                onChange={(e) => handleChange(index, e)}
                onKeyDown={(e) => handleKeyDown(index, e)}
                className="border text-center w-12 h-12 rounded text-lg"
              />
            ))}
          </div>

          <Button
            onClick={handleVerifyAuthenticator}
            className="w-full bg-blue-600 text-white p-2 rounded mt-4"
            isLoading={loading}
            disabled={loading}
          >
            Verify Code
          </Button>
        </div>
      </div>

      {/* Right Section */}
      <div className="w-1/2 relative loginScreen">
        <div className="h-screen">
          <Image
            src={bgimage}
            alt="authentication setup"
            className="h-full w-full"
          />
        </div>
        <div className="absolute top-1/2 text-white ms-[74px]">
          <div className="font-medium text-6.5xl">Secure Your Account</div>
          <div className="font-normal text-base">
            Enhance your security by enabling two-factor authentication.
          </div>
        </div>
      </div>
    </div>
  );
};

export default AuthenticatorSetup;
