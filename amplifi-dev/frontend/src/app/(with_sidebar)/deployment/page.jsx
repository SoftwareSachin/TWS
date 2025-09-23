"use client";

import DeploymentCard from "@/components/deployment/deploymentCard";
import React from "react";
import { useUser } from "@/context_api/userContext";

const DeploymentPage = () => {
  const { user } = useUser();

  const data = [
    {
      title: "Amplifi Cloud Pro",
      subtitle: "Time at 9:00 AM",
      status: "current",
    },
    {
      title: "Microsoft Azure",
      subtitle: "Subtext text here",
      status: "read",
    },
    {
      title: "Google Cloud Platform",
      subtitle: "Subtext text here",
      status: "read",
    },
    {
      title: "Amazon Web Services",
      subtitle: "Subtext text here",
      status: "read",
    },
    {
      title: "On Premise",
      subtitle: "Subtext text here",
      status: "option",
    },
  ];

  const currentDeployments = data.filter((item) => item.status === "current");
  const deploymentOptions = data.filter((item) => item.status !== "current");

  return (
    <div className="p-8 flex flex-col gap-8">
      <div className="flex justify-between">
        <div className="font-semibold text-2xl">Deployment</div>
      </div>

      {/* Current Deployment Section */}
      {currentDeployments.length > 0 && (
        <div>
          <div className="text-base font-medium text-black-20 mb-4">
            Current Deployment
          </div>
          <div className="grid grid-cols-3 gap-4">
            {currentDeployments.map((item, index) => (
              <DeploymentCard
                item={item}
                key={index}
                type={item?.status}
                user={user}
              />
            ))}
          </div>
        </div>
      )}

      {/* Deployment Options Section */}
      {deploymentOptions.length > 0 && (
        <div>
          <div className="text-base font-medium text-black-20 mb-4">
            Deployment Options
          </div>
          <div className="grid grid-cols-3 gap-4">
            {deploymentOptions.map((item, index) => (
              <DeploymentCard
                item={item}
                key={index}
                type={item?.status}
                user={user}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default DeploymentPage;
