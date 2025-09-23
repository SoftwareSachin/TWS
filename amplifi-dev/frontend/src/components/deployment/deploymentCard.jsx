"use client";

import React from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { SquareMenu } from "lucide-react";
import { identifyUserFromObject, captureEvent } from "@/utils/posthogUtils";

const DeploymentCard = ({ item, key, type, user = null }) => {
  const renderButton = () => {
    if (type === "current") {
      return (
        <button
          className="bg-blue-500 text-white px-3 py-2 rounded-md"
          onClick={() => {
            try {
              identifyUserFromObject(user);

              captureEvent("upgrade_button_clicked", {
                button_type: "upgrade",
                deployment_title: item?.title || "",
                deployment_subtitle: item?.subtitle || "",
                user_identified: !!user,
              });

              console.log(
                "Upgrade button clicked! Check PostHog dashboard for event.",
              );
            } catch (error) {
              console.error("Error in upgrade button click:", error);
            }
          }}
        >
          Upgrade
        </button>
      );
    } else if (type === "read") {
      return (
        <button
          className="px-3 py-2 rounded-md flex items-center text-black-10 border gap-4"
          onClick={() => {
            try {
              // PostHog: Capture the button click event (no user identification)
              captureEvent("documentation_button_clicked", {
                button_type: "read_documentation",
                deployment_title: item?.title || "",
                deployment_subtitle: item?.subtitle || "",
                user_agent: navigator.userAgent || "",
              });

              console.log(
                "Documentation button clicked! Check PostHog dashboard for event.",
              );
            } catch (error) {
              console.error("Error in documentation button click:", error);
            }
          }}
        >
          Read Documentation <SquareMenu className="h-4 w-4" />
        </button>
      );
    } else {
      return (
        <button
          className="bg-gray-200 text-gray-500 px-3 py-2 rounded-md disabled:border-gray-400 border"
          disabled
        >
          Coming Soon
        </button>
      );
    }
  };

  return (
    <Card className="rounded-md p-4" key={key}>
      <CardHeader className="p-0 pb-4">
        <CardTitle className="flex flex-col gap-2">
          <div className="font-semibold text-base">{item?.title}</div>
          <div className="text-sm font-medium">{item?.subtitle}</div>
        </CardTitle>
        <CardDescription>{item?.desc}</CardDescription>
      </CardHeader>
      <hr />
      <CardContent className="p-0 pt-4 text-xs font-normal flex justify-between items-center text-gray-500">
        {renderButton()}
      </CardContent>
    </Card>
  );
};

export default DeploymentCard;
