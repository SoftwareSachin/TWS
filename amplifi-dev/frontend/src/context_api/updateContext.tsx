// UpdateContext.tsx
import { createContext, useContext } from "react";

export const UpdateContext = createContext({
  triggerUpdate: () => {},
});

export const useUpdate = () => useContext(UpdateContext);
