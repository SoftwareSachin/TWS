/* This code snippet is a React component named `DetailCardWorkflow`. It is a card component that
displays workflow details. */
"use client";
import React from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import Image from "next/image";
import rightArrow from "@/assets/icons/right-single-arrow-blue.svg";
import clock from "@/assets/icons/clock-check.svg";
import { useRouter } from "next/navigation";
import Skeleton from "react-loading-skeleton";
import "react-loading-skeleton/dist/skeleton.css";

const DetailCardWorkflow = ({ item, index, loading = false }) => {
  const router = useRouter();
  return (
    <>
      {loading ? (
        <Card className="rounded-none">
          <CardHeader className="p-4">
            <CardTitle>
              <Skeleton height={60} />
            </CardTitle>
          </CardHeader>
          <hr />
          <CardContent className="p-4 text-xs font-normal flex justify-between items-center text-gray-500">
            <div className="flex flex-col">
              <div className="mb-2">
                <CardDescription>
                  <Skeleton count={2} height={15} />
                </CardDescription>
              </div>
              <div className="flex items-center space-x-2">
                <Skeleton width={80} height={20} />
              </div>
            </div>
            <div className="font-medium self-end cursor-pointer">
              <Skeleton width={80} height={20} />
            </div>
          </CardContent>
        </Card>
      ) : (
        <Card className="rounded-none" key={index}>
          <CardHeader className="p-4">
            <CardTitle>{item?.name}</CardTitle>
            <CardDescription className="text-black-0">
              {item?.description}
            </CardDescription>
          </CardHeader>
          <hr />
          <CardContent className="p-4 text-xs font-normal">
            <div className="text-gray-500 mb-4 flex items-center">
              <span>
                <Image
                  src={clock}
                  alt="document icon"
                  className="inline me-2"
                />
              </span>
              Last run at :{" "}
              <span className="text-black-10">
                {item?.lastUpdate || "9:00 AM  22 Aug,2024"}
              </span>
            </div>
            <div className="flex gap-4 justify-between text-black-20">
              <div className="flex items-center gap-1">
                <div
                  className={`h-2 w-2 rounded-full ${
                    item?.is_active ? "bg-green-10" : "bg-red-800"
                  }`}
                ></div>
                {item?.is_active ? "Currently running" : "Stopped"}
              </div>
              <div
                className="font-medium text-blue-10 align-middle cursor-pointer"
                onClick={() => router.push(`/workflows/${item?.id}`)}
              >
                View{" "}
                <Image
                  src={rightArrow}
                  alt="right arrow icon"
                  className="inline"
                />
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </>
  );
};

export default DetailCardWorkflow;
