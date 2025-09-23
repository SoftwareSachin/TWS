/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "api.loginradius.com",
      },
    ],
  },
  reactStrictMode: true,
  // Allow all dev origins for Replit proxy compatibility (moved out of experimental for Next.js 15)
  allowedDevOrigins: [
    '*.replit.dev',
    '*.repl.co', 
    'localhost:*',
    '127.0.0.1:*'
  ],
  experimental: {
    serverActions: {
      allowedOrigins: [
        '*.replit.dev',
        '*.repl.co',
        'localhost:*',
        '127.0.0.1:*'
      ],
    },
  },
  env: {
    NEXT_PUBLIC_BASE_URL: process.env.NEXT_PUBLIC_BASE_URL,
    NEXT_PUBLIC_SPEECH_KEY: process.env.SPEECH_KEY,
    NEXT_PUBLIC_CHAT_HOST_NAME: process.env.NEXT_PUBLIC_CHAT_HOST_NAME,
    NEXT_PUBLIC_POSTHOG_KEY: process.env.NEXT_PUBLIC_POSTHOG_KEY,
    NEXT_PUBLIC_POSTHOG_HOST: process.env.NEXT_PUBLIC_POSTHOG_HOST,
  },
  typedRoutes: false,
  async headers() {
    return [
      {
        // Apply security headers to all routes with CORS and host bypass for Replit
        source: "/(.*)",
        headers: [
          // CORS headers for Replit proxy
          {
            key: 'Access-Control-Allow-Origin',
            value: '*',
          },
          {
            key: 'Access-Control-Allow-Methods',
            value: 'GET, POST, PUT, DELETE, OPTIONS',
          },
          {
            key: 'Access-Control-Allow-Headers',
            value: 'Content-Type, Authorization',
          },
          // Content Security Policy - relaxed for Replit
          {
            key: "Content-Security-Policy",
            value: [
              "default-src 'self' *",
              "script-src 'self' 'unsafe-inline' 'unsafe-eval' blob: https: *",
              "style-src 'self' 'unsafe-inline' https: *",
              "img-src 'self' data: blob: https: http: *",
              "font-src 'self' https: data: *",
              "connect-src 'self' http: https: ws: wss: *",
              "frame-src 'self' *",
              "object-src 'none'",
              "base-uri 'self'",
              "form-action 'self'",
            ].join("; "),
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
        ],
      },
    ];
  },
};

export default nextConfig;
