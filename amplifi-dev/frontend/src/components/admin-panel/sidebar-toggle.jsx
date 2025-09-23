import { ChevronLeft } from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

export function SidebarToggle({ isOpen, setIsOpen }) {
  return (
    <div className="invisible lg:visible absolute top-[48px] -right-[14px] z-20">
      <Button
        onClick={() => setIsOpen?.()}
        className="rounded-full w-[24px] h-[24px]"
        variant="outline"
        size="icon"
      >
        <ChevronLeft
          className={cn(
            "h-4 w-4 transition-transform ease-in-out duration-700",
            isOpen === false ? "rotate-180" : "rotate-0",
          )}
        />
      </Button>
    </div>
  );
}
