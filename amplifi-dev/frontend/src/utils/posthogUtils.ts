import posthog from "posthog-js";

// Define User interface for type safety
interface User {
  id?: string;
  email?: string;
  name?: string;
  role?: string;
  clientId?: string;
}

// Define event properties interface
interface EventProperties {
  [key: string]: any;
}

/**
 * Hash function for UUIDs and strings
 * @param input - String to hash (e.g., UUID)
 * @returns string - Hashed string
 */
export const hashString = (input: string | null): string => {
  if (!input) return "";
  let hash = 0;
  if (input.length === 0) return hash.toString();

  for (let i = 0; i < input.length; i++) {
    const char = input.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash; // Convert to 32-bit integer
  }

  return Math.abs(hash).toString(36);
};

/**
 * Identify user with standardized properties from user object
 * @param user - User object
 * @param shouldHashId - Whether to hash the user ID (default: true)
 */
export const identifyUserFromObject = (
  user: User | null,
  shouldHashId: boolean = true,
) => {
  if (!user || !user.clientId) {
    console.warn("PostHog: User object is invalid or missing clientId");
    return;
  }

  try {
    const rawUserId = user.clientId;
    const userId = shouldHashId ? hashString(rawUserId) : rawUserId;

    const userProps = {
      email: user.email || "",
      name: user.name || "",
      role: user.role || "",
      client_id: userId,
    };

    posthog.identify(userId, userProps);
  } catch (error) {
    console.error("PostHog identifyUserFromObject error:", error);
  }
};

/**
 * Capture event without user identification
 * @param eventName - Event name
 * @param eventProps - Event properties
 */
export const captureEvent = (
  eventName: string,
  eventProps: EventProperties = {},
) => {
  try {
    posthog.capture(eventName, {
      ...eventProps,
      timestamp: new Date().toISOString(),
      page_url: window.location.href,
    });
  } catch (error) {
    console.error("PostHog capture error:", error);
  }
};
