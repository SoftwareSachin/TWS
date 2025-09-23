/* This JavaScript code snippet is setting up middleware for handling authentication in a Next.js
application. Here's a breakdown of what it does: */
import { NextResponse } from "next/server";
import { constants } from "@/lib/constants";
import Cookies from "universal-cookie";
import { getJwtToken } from "@/api/login";

const unprotectedRoutes = [
  "/login",
  "/forgot-password",
  "/reset-password",
  "/sucsses-page",
  "/authenticator-setup",
];

export const config = {
  matcher: ["/((?!api|_next|static).*)"],
};

export async function middleware(request) {
  const { pathname, searchParams } = request.nextUrl;
  const res = NextResponse.next();

  // Add security headers to all responses
  res.headers.set("X-Frame-Options", "DENY");
  res.headers.set("X-Content-Type-Options", "nosniff");
  res.headers.set("Referrer-Policy", "strict-origin-when-cross-origin");
  res.headers.set("X-XSS-Protection", "1; mode=block");

  // *** AUTHENTICATION BYPASSED FOR TESTING ***
  // Allow access to all routes without authentication checks
  console.log('Authentication bypassed - allowing access to:', pathname);
  
  // Redirect from login page to workspace if someone tries to access login
  if (pathname === '/login') {
    const workspaceUrl = request.nextUrl.clone();
    workspaceUrl.pathname = "/workspace";
    return NextResponse.redirect(workspaceUrl);
  }
  
  return res;
}
