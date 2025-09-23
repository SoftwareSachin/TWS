"use client";
import Link from "next/link";
import { ArrowLeft, Ellipsis } from "lucide-react";
import { useParams, usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { CollapseMenuButton } from "@/components/admin-panel/collapse-menu-button";
import {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
  TooltipProvider,
} from "@/components/ui/tooltip";
import { workspaceMenuList } from "@/lib/workspace-list";
import Cookies from "universal-cookie";
import { useUser } from "@/context_api/userContext";
import { getWorkSpaceByID } from "@/api/Workspace";
import { useEffect, useState } from "react";
import { showError } from "@/utils/toastUtils";

export function WorkspaceMenu({ isOpen }) {
  const id = useParams();
  const cookies = new Cookies();
  const pathname = usePathname();
  const menuList = workspaceMenuList(id?.workspaceId, pathname);
  const { user } = useUser();
  const [activeWorkspace, setActiveWorkspace] = useState(null);
  const getWorkspaceDetails = async () => {
    const data = {
      id: user?.clientId,
      workspaceId: id?.workspaceId,
    };
    try {
      const response = await getWorkSpaceByID(data);

      if (response.status === 200) {
        setActiveWorkspace(response.data.data);
      }
    } catch (error) {
      showError(`${error?.response?.data?.detail}`);
    }
  };
  useEffect(() => {
    getWorkspaceDetails();
  }, []);

  return (
    <div className="h-full flex flex-col">
      <ScrollArea className="h-full">
        <nav className="h-full w-full">
          <ul className="flex flex-col h-full w-full items-start space-y-1 px-2">
            <li className="flex items-center gap-2 text-sm font-medium my-6 ms-2">
              <Link href={`/workspace/?id=${user?.clientId}`}>
                <ArrowLeft className="w-4 h-4 cursor-pointer" />
              </Link>
              {isOpen && (
                <span className="text-base font-semibold">
                  {activeWorkspace?.name}
                </span>
              )}
            </li>
            {menuList.map(({ groupLabel, menus }, index) => (
              <li
                className={cn("w-full", groupLabel ? "pt-5" : "")}
                key={index}
              >
                {(isOpen && groupLabel) || isOpen === undefined ? (
                  <p className="text-sm font-medium text-muted-foreground px-4 pb-2 max-w-[248px] truncate">
                    {groupLabel}
                  </p>
                ) : !isOpen && isOpen !== undefined && groupLabel ? (
                  <TooltipProvider>
                    <Tooltip delayDuration={100}>
                      <TooltipTrigger className="w-full">
                        <div className="w-full flex justify-center items-center">
                          <Ellipsis className="h-5 w-5" />
                        </div>
                      </TooltipTrigger>
                      <TooltipContent side="right">
                        <p>{groupLabel}</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                ) : (
                  <p className="pb-2"></p>
                )}
                {menus.map(
                  ({ href, label, icon: Icon, active, submenus }, index) =>
                    !submenus || submenus.length === 0 ? (
                      <div className="w-full" key={index}>
                        <TooltipProvider disableHoverableContent>
                          <Tooltip delayDuration={100}>
                            <TooltipTrigger asChild>
                              <Button
                                variant={
                                  (active === undefined &&
                                    pathname.startsWith(href)) ||
                                  active
                                    ? "secondary"
                                    : "ghost"
                                }
                                className="w-full justify-start h-10 mb-1"
                                asChild
                              >
                                <Link href={href}>
                                  <span
                                    className={cn(
                                      isOpen === false ? "" : "mr-4",
                                    )}
                                  >
                                    {typeof Icon === "object" &&
                                    (Icon?.src || Icon?.default) ? (
                                      <img
                                        src={Icon?.src || Icon?.default}
                                        alt={`${label} icon`}
                                        className="w-[18px] h-[18px]"
                                      />
                                    ) : null}
                                  </span>
                                  <p
                                    className={cn(
                                      "max-w-[200px] truncate",
                                      isOpen === false
                                        ? "-translate-x-96 opacity-0"
                                        : "translate-x-0 opacity-100",
                                    )}
                                  >
                                    {label}
                                  </p>
                                </Link>
                              </Button>
                            </TooltipTrigger>
                            {isOpen === false && (
                              <TooltipContent side="right">
                                {label}
                              </TooltipContent>
                            )}
                          </Tooltip>
                        </TooltipProvider>
                      </div>
                    ) : (
                      <div className="w-full" key={index}>
                        <CollapseMenuButton
                          icon={Icon}
                          label={label}
                          active={
                            active === undefined
                              ? pathname.startsWith(href)
                              : active
                          }
                          submenus={submenus}
                          isOpen={isOpen}
                        />
                      </div>
                    ),
                )}
              </li>
            ))}
          </ul>
        </nav>
      </ScrollArea>
    </div>
  );
}
