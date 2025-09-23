import axios from "axios";
import Cookies from "universal-cookie";
import { constants } from "@/lib/constants";
import { getCookie, removeCookie, setCookie } from "@/utils/cookieHelper";
import { decodeToken } from "@/components/utility/decodeJwtToken";

const cookies = new Cookies();

export const baseURL = {
  v1: `${process.env.NEXT_PUBLIC_BASE_URL}/v1`,
  v2: `${process.env.NEXT_PUBLIC_BASE_URL}/v2`,
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

// Common request interceptor
const requestInterceptor = (config: any) => {
  const token = getCookie(constants.AUTH_TOKEN);
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
};

// Common error handler
const errorHandler = async (error: any, instance: any) => {
  const originalRequest = error.config;
  const statusCode = error?.response?.status;

  if (statusCode === 401 && !originalRequest._retry) {
    originalRequest._retry = true;

    if (!isRefreshing) {
      isRefreshing = true;

      try {
        const refreshToken = getCookie(constants.REFRESH_TOKEN);
        if (!refreshToken) {
          console.warn("No refresh token, redirecting...");
          removeCookie(constants.AUTH_TOKEN);
          removeCookie(constants.REFRESH_TOKEN);
          window.location.href = "/login";
          return Promise.reject(error);
        }

        const response = await axios.post(
          `${baseURL.v1}/refresh_access_token`,
          { refresh_token: refreshToken },
        );

        const { access_token, refresh_token, jwt_token } = response.data;

        setCookie(constants.AUTH_TOKEN, access_token);
        setCookie(constants.REFRESH_TOKEN, refresh_token);
        setCookie(constants.JWT_TOKEN, jwt_token);
        const userDetails = decodeToken(jwt_token);
        localStorage.setItem(constants.USER, JSON.stringify(userDetails));
        processQueue(null, access_token);

        originalRequest.headers.Authorization = `Bearer ${access_token}`;
        return instance(originalRequest);
      } catch (refreshError) {
        processQueue(refreshError, null);
        removeCookie(constants.AUTH_TOKEN);
        removeCookie(constants.REFRESH_TOKEN);
        removeCookie(constants.JWT_TOKEN);
        window.location.href = "/login";
        return Promise.reject(refreshError);
      } finally {
        isRefreshing = false;
      }
    }

    return new Promise((resolve, reject) => {
      failedQueue.push({
        resolve: (token: string) => {
          originalRequest.headers.Authorization = `Bearer ${token}`;
          resolve(instance(originalRequest));
        },
        reject: (err: any) => reject(err),
      });
    });
  }

  return Promise.reject(error);
};

// Axios instance for v1 pointing to v2 (for backward compatibility)
const http = axios.create({ baseURL: baseURL.v2 });
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
