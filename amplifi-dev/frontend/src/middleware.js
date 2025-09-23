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
  // const urlAuthToken = searchParams.get(constants.AUTH_TOKEN);
  // const urlRefreshToken = searchParams.get(constants.REFRESH_TOKEN);
  const token = request.cookies.get(constants.AUTH_TOKEN);
  let jwtToken = request.cookies.get(constants.JWT_TOKEN);
  const isAuthenticated = !!token;
  const res = NextResponse.next();

  // Add security headers to all responses
  res.headers.set("X-Frame-Options", "DENY");
  res.headers.set("X-Content-Type-Options", "nosniff");
  res.headers.set("Referrer-Policy", "strict-origin-when-cross-origin");
  res.headers.set("X-XSS-Protection", "1; mode=block");

  // if (urlRefreshToken && urlAuthToken) {
  //   res.cookies.set(constants.AUTH_TOKEN, urlAuthToken);
  //   res.cookies.set(constants.REFRESH_TOKEN, urlRefreshToken);
  // }
  // Allow access to unprotected routes
  // Removed sensitive token logging for security
  if (isAuthenticated && (!jwtToken || jwtToken.value === "")) {
    // jwtToken = await getJwtToken(token.value);
    // res.cookies.set(constants.JWT_TOKEN, jwtToken);
  }

  if (unprotectedRoutes.some((route) => pathname.startsWith(route))) {
    // Removed authentication status logging for security
    if (!isAuthenticated) {
      return res;
    } else {
      const workspaceUrl = request.nextUrl.clone();
      workspaceUrl.pathname = "/workspace";
      return NextResponse.redirect(workspaceUrl);
    }
  }

  // Redirect unauthenticated users to the login page
  if (!isAuthenticated) {
    const loginUrl = request.nextUrl.clone();
    loginUrl.pathname = "/login";
    return NextResponse.redirect(loginUrl);
  }

  // Allow authenticated users to access any route
  return res;
}
