"use client";
import Image from "next/image";
import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { z } from "zod";
import loginImage from "@/assets/images/login-image.png";
import icon from "@/assets/icons/BrandLogo.svg";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { createOrganisation } from "@/api/organisation";
import { showError, showSuccess } from "@/utils/toastUtils";
import Cookies from "universal-cookie";
import { setCookie } from "@/utils/cookieHelper";

// Define Zod schema for form validation
const formSchema = z.object({
  orgName: z.string().min(1, { message: "Organization name is required" }),
  domain: z.string().min(1, { message: "Domain is required" }),
});

const Setup = () => {
  const [orgName, setOrgName] = useState("");
  const [domain, setDomain] = useState("");
  const [errors, setErrors] = useState({});
  const router = useRouter();
  const cookies = new Cookies();

  const handleSubmit = async (e) => {
    e.preventDefault();
    // Clear previous errors
    setErrors({});
    // Validate the form values using Zod schema
    const result = formSchema.safeParse({ orgName, domain });
    if (!result.success) {
      // Extract and set the validation errors
      const validationErrors = result.error.flatten().fieldErrors;
      setErrors(validationErrors);
      return;
    }
    const body = {
      name: orgName,
      domain: domain,
      description: null,
    };
    try {
      const response = await createOrganisation(body);
      if (response.status === 200) {
        const id = response?.data?.data?.id;
        setCookie("orgId", id);
        showSuccess(`${response?.data?.message}`);
        router.push(`/get-started`);
      }
    } catch (error) {
      showError(`${error.response.data.detail}`);
    }
  };

  return (
    <div className="w-full h-full bg-gradient-radial from-red-500 via-blue-400 to-black at-center">
      <div className="w-full py-[56px]">
        <Image src={icon} alt="amplify-icon" className="ms-24" />
      </div>
      <div className="flex justify-between w-full gap-28 items-center">
        <div className="w-1/3 ms-24">
          <form onSubmit={handleSubmit}>
            <h3 className="mb-14 text-4xl font-medium text-gray-900">
              Set up your organization
            </h3>
            <div className="grid gap-4 mb-10 sm:grid-cols-1">
              <div>
                <label
                  htmlFor="orgname"
                  className="block mb-2 text-sm font-medium text-gray-900"
                >
                  Organization Name
                </label>
                <Input
                  type="text"
                  id="orgname"
                  className={`w-full ${
                    errors.orgName ? "border-red-500" : "border-gray-400"
                  }`}
                  placeholder="Enter your org name"
                  value={orgName}
                  onChange={(e) => setOrgName(e.target.value)}
                />
                {errors.orgName && (
                  <p className="text-red-500 text-sm">{errors.orgName[0]}</p>
                )}
              </div>
              <div>
                <label
                  htmlFor="domain"
                  className="block mb-2 text-sm font-medium text-gray-900"
                >
                  Domain
                </label>
                <Input
                  type="text"
                  id="domain"
                  className={`w-full ${
                    errors.domain ? "border-red-500" : "border-gray-400"
                  }`}
                  placeholder="Enter domain"
                  value={domain}
                  onChange={(e) => setDomain(e.target.value)}
                />
                {errors.domain && (
                  <p className="text-red-500 text-sm">{errors.domain[0]}</p>
                )}
              </div>
            </div>
            <Button type="submit" className="w-full bg-blue-10">
              Proceed
            </Button>
          </form>
        </div>
        <div className="w-2/3 h-[600px]">
          <Image
            src={loginImage}
            alt="image in login page"
            className="w-full h-full"
          />
        </div>
      </div>
    </div>
  );
};

export default Setup;
