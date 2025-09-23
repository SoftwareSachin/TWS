import * as React from "react";

export const FormLabel: React.FC<{
  htmlFor?: string;
  children: React.ReactNode;
}> = ({ htmlFor, children }) => (
  <label htmlFor={htmlFor} className="block font-medium pb-2">
    {children}
  </label>
);
