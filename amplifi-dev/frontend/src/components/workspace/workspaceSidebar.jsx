"use client";
import { SidebarToggle } from "@/components/admin-panel/sidebar-toggle";
import { useSidebar } from "@/hooks/use-sidebar";
import { useStore } from "@/hooks/use-store";
import { useIsLargeScreen } from "@/hooks/use-window-size";
import { cn } from "@/lib/utils";
import { WorkspaceMenu } from "./workspaceMenu";

export function WorkspaceSidebar() {
  const sidebar = useStore(useSidebar, (x) => x);
  const isLargeScreen = useIsLargeScreen();
  // Provide fallback state during hydration instead of returning null
  const { 
    isOpen = true, 
    toggleOpen = () => {}, 
    getOpenState = (isLarge = true) => true, 
    setIsHover = () => {}, 
    settings = { disabled: false } 
  } = sidebar || {};
  return (
    <aside
      className={cn(
        "mt-6 fixed top-0 left-0 z-0 h-screen translate-x-0 transition-[width] ease-in-out duration-300 bg-white pt-[16px]",
        !getOpenState(isLargeScreen) ? "w-[90px]" : "w-[240px]",
        settings.disabled && "hidden",
      )}
    >
      <SidebarToggle isOpen={isOpen} setIsOpen={toggleOpen} />
      <div
        onMouseEnter={() => setIsHover(true)}
        onMouseLeave={() => setIsHover(false)}
        className="relative h-full flex flex-col px-4 py-6 overflow-y-auto shadow-md dark:shadow-zinc-800"
      >
        <WorkspaceMenu isOpen={getOpenState(isLargeScreen)} />
      </div>
    </aside>
  );
}
