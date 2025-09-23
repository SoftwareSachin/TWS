import React from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { SquareMenu } from "lucide-react";
import Link from "next/link";
import InfoIcon from "@/assets/icons/info-circle.svg";
import Image from "next/image";

const BillingCard = ({ item, key, onClick, type }) => {
  return (
    <Card className="rounded-md p-4 w-2/3" key={key}>
      <CardHeader className="p-0 pb-4">
        <CardTitle className="flex flex-col gap-2">
          <div className="font-medium text-base text-gray-500">
            Current plan
          </div>
          <div className="font-medium text-3xl">Pro plus</div>
          <div className="flex justify-between bg-sky-200 py-2 px-4 rounded-sm border border-sky-500">
            <div className="text-sm font-medium">
              {" "}
              <Image
                src={InfoIcon}
                alt="info icon"
                className="inline w-4 h-4 text-white"
              />{" "}
              Your next bill is $545.70 on Jul 15, 2025. Your card ending in
              ••••3748 will be charged
            </div>
            <div className="text-sm font-medium">Pay now</div>
          </div>
        </CardTitle>
        <CardDescription>{item?.desc}</CardDescription>
      </CardHeader>
      <CardContent className="p-0 text-xs font-normal">
        View all plans & features on the{" "}
        <Link href="/" className="text-blue-10">
          Pricing page
        </Link>{" "}
        •{" "}
        <Link href="/" className="text-blue-10">
          Upgrade plan
        </Link>
      </CardContent>
    </Card>
  );
};

export default BillingCard;
