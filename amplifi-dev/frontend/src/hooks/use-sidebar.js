import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import { produce } from "immer";

export const useSidebar = create(
  persist(
    (set, get) => ({
      isOpen: true,
      isHover: false,
      settings: { disabled: false, isHoverOpen: false },
      toggleOpen: () => {
        set({ isOpen: !get().isOpen });
      },
      setIsOpen: (isOpen) => {
        set({ isOpen });
      },
      setIsHover: (isHover) => {
        set({ isHover });
      },
      getOpenState: (isLargeScreen = true) => {
        const state = get();
        // On smaller screens, force collapsed state regardless of isOpen
        if (!isLargeScreen) {
          return state.settings.isHoverOpen && state.isHover;
        }
        return state.isOpen || (state.settings.isHoverOpen && state.isHover);
      },
      setSettings: (settings) => {
        set(
          produce((state) => {
            state.settings = { ...state.settings, ...settings };
          }),
        );
      },
    }),
    {
      name: "sidebar",
      storage: createJSONStorage(() => localStorage),
    },
  ),
);
