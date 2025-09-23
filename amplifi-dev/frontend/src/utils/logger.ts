// utils/logger.ts
export const isDebugMode = process.env.NEXT_PUBLIC_DEBUG === "true";

export function debugLog(...args: any[]) {
  if (isDebugMode) {
    console.log(...args);
  }
}
