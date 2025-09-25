/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable support for Replit proxy environment  
  allowedDevOrigins: [
    'https://*.replit.dev',
    'https://*.replit.app', 
    'https://*.replit.com',
    'https://*.worf.replit.dev',
    'http://localhost:5000',
    'http://0.0.0.0:5000'
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

  // Experimental features (disabled for stability)
  experimental: {},

  // Configure images for better performance
  images: {
    domains: ['localhost', '0.0.0.0'],
    formats: ['image/webp', 'image/avif'],
  },

};

module.exports = nextConfig;