"use client";

import { decodeToken } from "@/components/utility/decodeJwtToken";
import {
  createContext,
  useContext,
  useState,
  useEffect,
  ReactNode,
} from "react";
import Cookies from "universal-cookie";
import { constants } from "@/lib/constants";

// 1. Define your User type (update fields as needed)
export interface User {
  email: string;
  clientId: string;
  // Add any other user fields here
  roles?: string[]; // Optional roles field
}

// 2. Define the context shape
interface UserContextType {
  user: User | null;
  setUser: (user: User | null) => void;
}

// 3. Create the context with proper type
const UserContext = createContext<UserContextType | undefined>(undefined);

// 4. Provider with typed props
export const UserProvider = ({ children }: { children: ReactNode }) => {
  const cookies = new Cookies();
  const [user, setUser] = useState<User | null>(null);
  const token = cookies.get(constants.JWT_TOKEN);
  useEffect(() => {
    // Check localStorage on mount
    const storedUser = localStorage.getItem(constants.USER);
    if (storedUser) {
      try {
        const parsedUser = JSON.parse(storedUser);
        setUser(parsedUser);
      } catch (error) {
        console.error("UserContext - Error parsing localStorage user:", error);
      }
    }

    const handleStorageChange = (event: StorageEvent) => {
      if (event.key === constants.USER) {
        const newUser = event.newValue ? JSON.parse(event.newValue) : null;
        setUser(newUser);
      }
    };
    window.addEventListener("storage", handleStorageChange);
    return () => window.removeEventListener("storage", handleStorageChange);
  }, []);
  useEffect(() => {
    if (token) {
      const userDetails = decodeToken(token) as User;
      if (userDetails) {
        setUser(userDetails);
        // Also update localStorage to keep it in sync
        localStorage.setItem(constants.USER, JSON.stringify(userDetails));
      }
    }
  }, [token]);

  return (
    <UserContext.Provider value={{ user, setUser }}>
      {children}
    </UserContext.Provider>
  );
};

// 5. Custom hook
export const useUser = (): UserContextType => {
  const context = useContext(UserContext);
  if (!context) {
    throw new Error("useUser must be used within a UserProvider");
  }
  return context;
};
