import Image from "next/image";
import React from "react";
import { Button } from "../ui/button";

const NoDataScreen = ({ title, subtitle, buttonText, image, onClick }) => {
  return (
    <div className="flex flex-col gap-4 w-full h-screen items-center pt-[25vh]">
      <div className="flex flex-col gap-4 justify-center items-center">
        <Image src={image} alt="destination empty screen image" />
        <div className="text-base font-medium text-center">{title}</div>
        <div className="font-normal text-sm text-gray-800 w-2/3 text-center">
          {subtitle}
        </div>
        <Button
          className="bg-blue-500 rounded-lg font-medium text-sm text-white px-4 py-2"
          onClick={onClick}
        >
          {buttonText}
        </Button>
      </div>
    </div>
  );
};

export default NoDataScreen;
