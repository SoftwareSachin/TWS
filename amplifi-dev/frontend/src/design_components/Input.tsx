import * as React from "react";
import { cn } from "@/lib/utils";
import { Eye, EyeOff } from "lucide-react";

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  state?: "default" | "active" | "error" | "disabled";
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, state = "default", ...props }, ref) => {
    const [showPassword, setShowPassword] = React.useState(false);
    const isPasswordType = type === "password";
    const timerRef = React.useRef<NodeJS.Timeout | null>(null);

    const handleToggle = () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      setShowPassword(true);
      timerRef.current = setTimeout(() => {
        setShowPassword(false);
      }, 3000);
    };

    React.useEffect(() => {
      return () => {
        if (timerRef.current) clearTimeout(timerRef.current);
      };
    }, []);

    const getStateStyles = () => {
      switch (state) {
        case "error":
          return "border-red-500 focus:border-red-500 focus:ring-red-500";
        case "active":
          return "border-blue-500 focus:border-blue-500 focus:ring-blue-500";
        case "disabled":
          return "bg-gray-100 border-gray-300 text-gray-500 cursor-not-allowed";
        default:
          return "border-input focus:border-blue-500 focus:ring-blue-500";
      }
    };

    return (
      <div className="relative w-full">
        <input
          type={isPasswordType && showPassword ? "text" : type}
          className={cn(
            "flex h-9 w-full rounded-md border bg-transparent shadow-sm transition-colors",
            type === "range" ? "cursor-pointer" : "px-3 py-1 pr-10",
            "file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-foreground",
            "placeholder:text-muted-foreground focus:outline-none focus:ring-1",
            "disabled:cursor-not-allowed disabled:opacity-50",
            getStateStyles(),
            className,
          )}
          ref={ref}
          disabled={state === "disabled"}
          {...props}
        />
        {isPasswordType && (
          <button
            type="button"
            onClick={handleToggle}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
            tabIndex={-1}
          >
            {showPassword ? (
              <Eye className="h-4 w-4" />
            ) : (
              <EyeOff className="h-4 w-4" />
            )}
          </button>
        )}
      </div>
    );
  },
);

Input.displayName = "Input";

export { Input };
