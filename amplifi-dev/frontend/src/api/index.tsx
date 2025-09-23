import axios from "axios";
import Cookies from "universal-cookie";
import { constants } from "@/lib/constants";
import { getCookie, removeCookie, setCookie } from "@/utils/cookieHelper";
import { decodeToken } from "@/components/utility/decodeJwtToken";

const cookies = new Cookies();

// Use relative API paths for better portability across environments
export const baseURL = {
  v1: '/api/v1',
  v2: '/api/v2',
};

let isRefreshing = false;
let failedQueue: any[] = [];

const processQueue = (error: any, token: string | null = null) => {
  failedQueue.forEach((prom) => {
    if (error) {
      prom.reject(error);
    } else {
      prom.resolve(token);
    }
  });
  failedQueue = [];
};

// Common request interceptor - AUTHENTICATION DISABLED
const requestInterceptor = (config: any) => {
  // Authentication disabled - no Authorization headers added
  return config;
};

// Common error handler - AUTHENTICATION DISABLED
const errorHandler = async (error: any, instance: any) => {
  // Authentication disabled - no token refresh or login redirects
  // Just pass through the error without any auth handling
  return Promise.reject(error);
};

// Axios instance for v1 (for file uploads and workspace operations)
const http = axios.create({ baseURL: baseURL.v1 });
http.interceptors.request.use(requestInterceptor, Promise.reject);
http.interceptors.response.use(
  (response) => response,
  (error) => errorHandler(error, http),
);

// Axios instance for v2
const httpV2 = axios.create({ baseURL: baseURL.v2 });
httpV2.interceptors.request.use(requestInterceptor, Promise.reject);
httpV2.interceptors.response.use(
  (response) => response,
  (error) => errorHandler(error, httpV2),
);

export { http, httpV2 };
