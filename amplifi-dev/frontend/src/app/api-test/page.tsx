"use client";

import { useState } from 'react';

export default function ApiTest() {
  const [status, setStatus] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const testBackendConnection = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const baseUrl = process.env.NEXT_PUBLIC_BASE_URL || 'http://localhost:8000';
      console.log('Testing connection to:', `${baseUrl}/api/v1/status`);
      
      const response = await fetch(`${baseUrl}/api/v1/status`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      setStatus(data);
      console.log('Backend response:', data);
    } catch (err: any) {
      console.error('API Test Error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-8 bg-gray-50">
      <div className="max-w-2xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">Frontend-Backend API Test</h1>
        
        <div className="bg-white p-6 rounded-lg shadow-md mb-6">
          <h2 className="text-xl font-semibold mb-4">Configuration</h2>
          <p className="text-gray-600">
            <strong>Backend URL:</strong> {process.env.NEXT_PUBLIC_BASE_URL || 'http://localhost:8000'}
          </p>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4">Connection Test</h2>
          
          <button
            onClick={testBackendConnection}
            disabled={loading}
            className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:bg-gray-400"
          >
            {loading ? 'Testing...' : 'Test Backend Connection'}
          </button>

          {error && (
            <div className="mt-4 p-4 bg-red-100 border-l-4 border-red-500 text-red-700">
              <strong>Error:</strong> {error}
            </div>
          )}

          {status && (
            <div className="mt-4 p-4 bg-green-100 border-l-4 border-green-500">
              <h3 className="text-lg font-semibold text-green-700 mb-2">Success! Backend Response:</h3>
              <pre className="text-sm bg-gray-100 p-3 rounded overflow-auto">
                {JSON.stringify(status, null, 2)}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}