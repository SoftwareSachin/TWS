/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable support for Replit proxy environment
  allowedDevOrigins: [
    // Allow all Replit domains
    'https://*.replit.dev',
    'https://*.replit.app',
    'https://*.replit.com',
    // Allow localhost for development
    'http://localhost:5000',
    'http://0.0.0.0:5000',
    // Allow all origins in development
    '*'
  ],
  
  // Configure headers for CORS support
  async headers() {
    return [
      {
        // Apply headers to all API routes
        source: '/api/:path*',
        headers: [
          { key: 'Access-Control-Allow-Origin', value: '*' },
          { key: 'Access-Control-Allow-Methods', value: 'GET, POST, PUT, DELETE, OPTIONS' },
          { key: 'Access-Control-Allow-Headers', value: 'Content-Type, Authorization' },
        ],
      },
      {
        // Apply headers to all pages
        source: '/:path*',
        headers: [
          { key: 'X-Frame-Options', value: 'SAMEORIGIN' },
          { key: 'X-Content-Type-Options', value: 'nosniff' },
        ],
      },
    ];
  },

  // Configure rewrites for backend API
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
    ];
  },

  // Optimize for production
  experimental: {
    optimizeCss: true,
  },

  // Configure images for better performance
  images: {
    domains: ['localhost', '0.0.0.0'],
    formats: ['image/webp', 'image/avif'],
  },

  // Disable telemetry for cleaner logs
  telemetry: false,
};

module.exports = nextConfig;