import BillingCard from "@/components/billing/billingCard";
import { Button } from "@/components/ui/button";
import React from "react";
import paymentEdit from "@/assets/icons/payment-method.svg";
import Image from "next/image";

const billingPage = () => {
  return (
    <div className="p-8">
      <div className="flex justify-between">
        <div className="font-semibold text-2xl">Billing</div>
        <div>
          <Button className="border bg-white text-black-10 shadow-none me-4 text-sm font-medium hover:text-white">
            Cancel plan
          </Button>
          <Button className="bg-blue-10">
            <Image
              src={paymentEdit}
              alt="payment method edit icon"
              className="me-2"
            />
            Manage Payment Method
          </Button>
        </div>
      </div>
      <div>
        <BillingCard />
      </div>
    </div>
  );
};

export default billingPage;
