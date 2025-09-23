import Image from "next/image";
import Link from "next/link";
import React from "react";

const DatasetStatus = ({ statusImage, subtitle, title, status }) => {
  return (
    <div className="flex justify-center items-center flex-col w-full h-full">
      <Image src={statusImage} alt="status icon" className="w-8 h-8" />
      <div className="font-medium text-base mt-4">{title}</div>
      <div className="font-normal text-sm mb-4">{subtitle}</div>
      {status === "failed" ? (
        <div className="text-blue-10">Upload again</div>
      ) : (
        ""
      )}
    </div>
  );
};

export default DatasetStatus;
