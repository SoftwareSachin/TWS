import MultiStepForm from "@/components/loginComponents/stepperForm";
import Image from "next/image";
import icon from "@/assets/icons/BrandLogo.svg";
import React from "react";

const GetStarted = () => {
  return (
    <div className="w-full h-full">
      <div className="w-full py-[56px]">
        <Image src={icon} alt="amplify-icon" className="ms-24" />
      </div>
      <div className="flex justify-center w-full gap-28 items-center">
        <MultiStepForm />
      </div>
    </div>
  );
};

export default GetStarted;
