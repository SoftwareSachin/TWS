/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    domains: ["api.loginradius.com"],
    remotePatterns: [
      {
        protocol: "https",
        hostname: "api.loginradius.com",
      },
    ],
  },
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_BASE_URL: process.env.NEXT_PUBLIC_BASE_URL,
    NEXT_PUBLIC_SPEECH_KEY: process.env.SPEECH_KEY,
    NEXT_PUBLIC_CHAT_HOST_NAME: process.env.NEXT_PUBLIC_CHAT_HOST_NAME,
    NEXT_PUBLIC_POSTHOG_KEY: process.env.NEXT_PUBLIC_POSTHOG_KEY,
    NEXT_PUBLIC_POSTHOG_HOST: process.env.NEXT_PUBLIC_POSTHOG_HOST,
  },
  experimental: {
    typedRoutes: false,
    missingSuspenseWithCSRBailout: false,
  },
  async headers() {
    return [
      {
        // Apply security headers to all routes
        source: "/(.*)",
        headers: [
          // Strict Transport Security (HSTS)
          {
            key: "Strict-Transport-Security",
            value: "max-age=31536000; includeSubDomains; preload",
          },
          // Content Security Policy
          {
            key: "Content-Security-Policy",
            value: [
              "default-src 'self'",
              "script-src 'self' 'unsafe-inline' 'unsafe-eval' blob: https://cdn.jsdelivr.net https://unpkg.com https://cdn.plot.ly https://cdnjs.cloudflare.com https://us-assets.i.posthog.com https://us.i.posthog.com https://*.posthog.com https://*.posthog.io",
              "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net",
              "img-src 'self' data: blob: https: http:",
              "font-src 'self' https://fonts.gstatic.com data:",
              "connect-src 'self' http://localhost:8085 https://*.dataamplifi.com https://*.posthog.com https://*.posthog.io https://edge.api.flagsmith.com https://*.flagsmith.com https://*.tile.openstreetmap.org https://tile.openstreetmap.org https://cdn.plot.ly wss: ws:",
              "frame-src 'self'",
              "object-src 'none'",
              "base-uri 'self'",
              "form-action 'self'",
              "frame-ancestors 'none'",
              "upgrade-insecure-requests",
            ].join("; "),
          },
          // X-Frame-Options
          {
            key: "X-Frame-Options",
            value: "DENY",
          },
          // X-Content-Type-Options
          {
            key: "X-Content-Type-Options",
            value: "nosniff",
          },
          // Referrer Policy
          {
            key: "Referrer-Policy",
            value: "strict-origin-when-cross-origin",
          },
          // Permissions Policy
          {
            key: "Permissions-Policy",
            value: [
              "camera=()",
              "microphone=(self)",
              "geolocation=()",
              "interest-cohort=()",
              "payment=()",
              "usb=()",
              "magnetometer=()",
              "gyroscope=()",
              "accelerometer=()",
            ].join(", "),
          },
          // X-XSS-Protection (for older browsers)
          {
            key: "X-XSS-Protection",
            value: "1; mode=block",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
