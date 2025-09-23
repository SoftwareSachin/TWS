import Link from "next/link";

export const ForgotPasswordLink = () => {
  return (
    <div className={"flex !mt-1"}>
      <Link
        className={"text-xs ml-auto hover:underline"}
        href={"/forgot-password"}
      >
        Forgot Password?
      </Link>
    </div>
  );
};
