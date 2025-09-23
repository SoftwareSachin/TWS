import Cookies from "universal-cookie";
import { CookieSetOptions } from "universal-cookie/cjs/types";

const cookies = new Cookies();

const isLocalhost = () => {
  if (typeof window === "undefined") return false;
  const host = window.location.hostname;
  return host === "localhost" || host === "127.0.0.1";
};

const defaultOptions: Partial<CookieSetOptions> = {
  path: "/",
  domain: isLocalhost()
    ? undefined
    : process.env.NEXT_PUBLIC_COOKIE_DOMAIN || ".dataamplifi.com",
  secure: !isLocalhost(),
  httpOnly: false,
  sameSite: isLocalhost() ? "lax" : "none", // 'none' is needed for cross-subdomain with secure
};

export const setCookie = (
  name: string,
  value: string,
  options: Partial<CookieSetOptions> = {},
): void => {
  cookies.set(name, value, { ...defaultOptions, ...options });
};

export const getCookie = (name: string): string | undefined => {
  return cookies.get(name);
};

export const removeCookie = (
  name: string,
  options: Partial<CookieSetOptions> = {},
): void => {
  cookies.remove(name, { ...defaultOptions, ...options });
};
