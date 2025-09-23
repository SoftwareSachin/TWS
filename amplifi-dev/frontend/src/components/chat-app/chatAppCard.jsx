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
import bgimage from "@/assets/images/workspace/workspace-default.svg";
import Skeleton from "react-loading-skeleton";
import "react-loading-skeleton/dist/skeleton.css";

const ChatAppCard = ({ item, KeyToChild, loading = false }) => {
  // Define background and text color pairs
  const chatAppType = {
    sql_chat_app: "SQL Chat App",
    unstructured_chat_app: "Unstructured Chat App",
  };

  const llmModels = {
    GPT35: "GPT3.5",
    GPT40: "GPT4.o",
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
        <Card className="rounded-none" key={KeyToChild}>
          <CardHeader className="p-4">
            <CardTitle>
              <Image
                src={bgimage}
                alt="card icon"
                className="me-2 bg-gray-20 w-full"
              />
              <div className="pt-4">{item?.name}</div>
            </CardTitle>
            <CardDescription>{item?.description}</CardDescription>
          </CardHeader>
          <hr />
          <CardContent className="p-4 text-xs font-normal flex justify-between items-center text-gray-500">
            <div className="flex flex-col">
              <div className="mb-2">{chatAppType[item?.chat_app_type]}</div>
              <div className="mb-2">
                {llmModels[item?.generation_config?.llm_model]}
              </div>
            </div>
            <div className="font-medium text-blue-500 self-end cursor-pointer">
              <a
                target={"_blank"}
                href={`${window.location.origin}/chatapp/${item?.id}`}
                rel="noopener noreferrer"
              >
                View{" "}
                <Image
                  src={rightArrow}
                  alt="right arrow icon"
                  className="inline"
                />
              </a>
            </div>
          </CardContent>
        </Card>
      )}
    </>
  );
};

export default ChatAppCard;
