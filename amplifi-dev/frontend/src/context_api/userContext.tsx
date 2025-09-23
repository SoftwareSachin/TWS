"use client";

import {
  createContext,
  useContext,
  useState,
  useEffect,
  ReactNode,
} from "react";

// *** AUTHENTICATION BYPASSED FOR TESTING ***
// Mock user data to bypass authentication

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

// Mock user data for testing
const MOCK_USER: User = {
  email: "test@amplifi.com",
  clientId: "test-org-123", 
  roles: ["admin", "developer", "user"]
};

// 3. Create the context with proper type
const UserContext = createContext<UserContextType | undefined>(undefined);

// 4. Provider with typed props - BYPASSED VERSION
export const UserProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(MOCK_USER);

  useEffect(() => {
    // Always set mock user for testing
    console.log("Authentication bypassed - using mock user:", MOCK_USER);
    setUser(MOCK_USER);
    
    // Store mock user in localStorage for consistency
    if (typeof window !== 'undefined') {
      localStorage.setItem("amplifi_user", JSON.stringify(MOCK_USER));
    }
  }, []);

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
