"use client";
import { Sidebar } from "@/components/admin-panel/sidebar";
import { useSidebar } from "@/hooks/use-sidebar";
import { useStore } from "@/hooks/use-store";
import { useIsLargeScreen } from "@/hooks/use-window-size";
import { cn } from "@/lib/utils";
import { WorkspaceSidebar } from "./workspaceSidebar";

export default function WorkspacePanelLayout({ children }) {
  const sidebar = useStore(useSidebar, (x) => x);
  const isLargeScreen = useIsLargeScreen();
  if (!sidebar) return null;
  const { getOpenState, settings } = sidebar;
  return (
    <>
      <WorkspaceSidebar />
      <main
        className={cn(
          "min-h-[calc(100vh)] bg-zinc-50 dark:bg-zinc-900 transition-[margin-left] ease-in-out duration-300",
          !settings.disabled &&
            (!getOpenState(isLargeScreen) ? "ml-[90px]" : "ml-[240px]"),
        )}
      >
        {children}
      </main>
    </>
  );
}
