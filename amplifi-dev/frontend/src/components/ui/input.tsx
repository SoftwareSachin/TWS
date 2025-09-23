import * as React from "react";
import { cn } from "@/lib/utils";
import { Eye, EyeOff } from "lucide-react";

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => {
    const [showPassword, setShowPassword] = React.useState(false);
    const isPasswordType = type === "password";
    const timerRef = React.useRef<NodeJS.Timeout | null>(null);

    const handleToggle = () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      setShowPassword(true);

      // Hide again after 3 seconds
      timerRef.current = setTimeout(() => {
        setShowPassword(false);
      }, 3000);
    };

    React.useEffect(() => {
      return () => {
        if (timerRef.current) clearTimeout(timerRef.current);
      };
    }, []);

    return (
      <div className="relative w-full">
        <input
          type={isPasswordType && showPassword ? "text" : type}
          className={cn(
            "flex h-9 w-full rounded-md border border-input bg-transparent shadow-sm transition-colors",
            type === "range" ? "cursor-pointer" : "px-3 py-1 pr-10",
            "file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50",
            className,
          )}
          ref={ref}
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
