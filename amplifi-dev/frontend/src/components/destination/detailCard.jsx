import React from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";
import Image from "next/image";
import documentIcon from "@/assets/icons/document-icon.svg";
import rightArrow from "@/assets/icons/right-single-arrow-blue.svg";
import dots from "@/assets/icons/dots-vertical.svg";
import avathar from "@/assets/icons/Avatar-Alphabates.svg";
import Skeleton from "react-loading-skeleton";
import "react-loading-skeleton/dist/skeleton.css";

const DetailCard = ({ item, onClick, onEdit, onDelete, loading = false }) => {
  return (
    <>
      {loading ? (
        <Card className="rounded-none">
          <CardHeader className="p-4">
            <CardTitle className="flex items-center justify-between">
              <div>
                <Skeleton circle={true} width={32} height={32} />
                <div className="pt-4">
                  <Skeleton width={`60%`} height={20} />
                </div>
              </div>
              <Skeleton width={20} height={20} />
            </CardTitle>
            <CardDescription>
              <Skeleton count={2} height={15} />
            </CardDescription>
          </CardHeader>
          <hr />
          <CardContent className="p-4 text-xs font-normal flex justify-between items-center text-gray-500">
            <div className="flex flex-col">
              <div className="mb-2">
                <Skeleton width={60} height={15} />
              </div>
              <div className="flex items-center space-x-2">
                <Skeleton circle={true} width={32} height={32} />
                <Skeleton circle={true} width={32} height={32} />
                <Skeleton circle={true} width={32} height={32} />
              </div>
            </div>
            <div className="font-medium self-end cursor-pointer">
              <Skeleton width={80} height={20} />
            </div>
          </CardContent>
        </Card>
      ) : (
        <Card className="rounded-none">
          <CardHeader className="p-4">
            <CardTitle className="flex gap-4 items-center justify-between">
              <div>
                <Image src={avathar} alt="card icon" className="inline me-2" />
                {item?.name}
              </div>
              <DropdownMenu>
                <DropdownMenuTrigger className="focus:outline-none">
                  <Image
                    src={dots}
                    alt="options"
                    className="self-start cursor-pointer"
                  />
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start" className="w-28">
                  <DropdownMenuItem
                    onClick={onDelete}
                    className="hover:!bg-blue-100"
                  >
                    Delete
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </CardTitle>
            <CardDescription>{item?.description}</CardDescription>
          </CardHeader>
          <hr />
          <CardContent className="p-4 text-xs font-normal flex justify-between items-center text-gray-500">
            <div>
              <span>
                <Image
                  src={documentIcon}
                  alt="document icon"
                  className="inline me-2"
                />
              </span>
              Last updated on :{" "}
              <span className="text-black-10">{item?.lastUpdate}</span>
            </div>
            <div
              className="font-medium text-blue-10 align-middle cursor-default"
              onClick={onClick}
            >
              View{" "}
              <Image
                src={rightArrow}
                alt="right arrow icon"
                className="inline"
              />
            </div>
          </CardContent>
        </Card>
      )}
    </>
  );
};

export default DetailCard;
