/**
 * The CommonDataWrapper component wraps its children components with the UserProvider from the
 * userContext context API.
 * @returns The CommonDataWrapper component is being returned, which wraps the children components with
 * the UserProvider component from the userContext context API.
 */
import React from "react";
import { UserProvider } from "@/context_api/userContext";
import { GraphProvider } from "@/context_api/graphContext";

function CommonDataWrapper({ children }) {
  return (
    <UserProvider>
      <GraphProvider>{children}</GraphProvider>
    </UserProvider>
  );
}

export default CommonDataWrapper;
