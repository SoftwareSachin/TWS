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
import rightArrow from "@/assets/icons/right-single-arrow-blue.svg";
import bgimage from "@/assets/images/workspace/workspace-default.svg";
import Skeleton from "react-loading-skeleton";
import dots from "@/assets/icons/dots-vertical.svg";
import "react-loading-skeleton/dist/skeleton.css";

const WorkspaceCard = ({
  item,
  onClick,
  KeyToChild,
  loading = false,
  onDelete,
  onEdit,
}) => {
  const membersToShow = item?.members?.slice(0, 5) || [];
  const remainingCount = (item?.members?.length || 0) - membersToShow.length;

  // Define background and text color pairs
  const colorPairs = [
    { bg: "bg-custom-blueBg", text: "text-custom-blueText" },
    { bg: "bg-custom-redBg", text: "text-custom-redText" },
    { bg: "bg-custom-pinkBg", text: "text-custom-pinkText" },
    { bg: "bg-custom-tealBg", text: "text-custom-tealText" },
    { bg: "bg-custom-yellowBg", text: "text-custom-yellowText" },
    { bg: "bg-custom-purpleBg", text: "text-custom-purpleText" },
  ];

  // Function to randomly select a color pair
  const getRandomColorPair = () => {
    return colorPairs[Math.floor(Math.random() * colorPairs.length)];
  };

  return (
    <>
      {loading ? (
        <Card className="rounded-none" key={KeyToChild}>
          <CardHeader className="p-4">
            <CardTitle>
              <Skeleton height={120} />
              <div className="pt-4">
                <Skeleton width={`60%`} height={20} />
              </div>
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
        <Card
          className="rounded-none h-[300px] flex flex-col justify-between"
          key={KeyToChild}
          onClick={onClick}
        >
          <div className="absolute top-3 right-3 z-10 bg-white/70 p-1">
            <DropdownMenu>
              <DropdownMenuTrigger className="focus:outline-none">
                <Image src={dots} alt="options" className="cursor-pointer" />
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-28 z-50">
                {onEdit && (
                  <DropdownMenuItem
                    onClick={(e) => {
                      e.stopPropagation();
                      onEdit?.();
                    }}
                    className="hover:!bg-blue-100"
                  >
                    Edit
                  </DropdownMenuItem>
                )}
                {onDelete && (
                  <DropdownMenuItem
                    onClick={(e) => {
                      e.stopPropagation();
                      onDelete();
                    }}
                    className="hover:!bg-blue-100"
                  >
                    Delete
                  </DropdownMenuItem>
                )}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
          <CardHeader className="p-4 flex flex-col gap-2 overflow-hidden">
            <CardTitle>
              <Image
                src={bgimage}
                alt="card icon"
                className="me-2 bg-gray-20 w-full object-cover h-[100px]"
              />
              <div className="pt-2 text-base font-semibold line-clamp-1">
                {item?.name}
              </div>
            </CardTitle>
            <CardDescription>
              <p className="text-sm text-muted-foreground break-words whitespace-normal line-clamp-3">
                {item?.description}
              </p>
            </CardDescription>
          </CardHeader>
          <hr />
          <CardContent className="p-4 text-xs font-normal flex justify-between items-center text-gray-500">
            <div className="flex flex-col">
              <div className="mb-2">Members</div>
              <div className="flex items-center -space-x-3">
                {membersToShow.map((member, index) => {
                  const { bg, text } = getRandomColorPair();
                  return (
                    <div
                      key={index}
                      className={`w-8 h-8 flex items-center justify-center text-sm rounded-full border-2 border-white ${bg} ${text}`}
                    >
                      {member.toUpperCase()}
                    </div>
                  );
                })}
                {remainingCount > 0 && (
                  <div className="w-8 h-8 flex items-center justify-center text-sm text-black rounded-full border-2 border-white bg-gray-300">
                    +{remainingCount}
                  </div>
                )}
              </div>
            </div>
            <div className="font-medium text-blue-500 self-end cursor-pointer">
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

export default WorkspaceCard;
