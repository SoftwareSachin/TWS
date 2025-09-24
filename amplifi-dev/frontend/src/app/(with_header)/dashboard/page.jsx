"use client";
import React from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { BarChart3, Upload, TrendingUp, FileSpreadsheet, Zap, ArrowRight } from "lucide-react";

const DashboardPage = () => {
  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Welcome to Amplifi Dashboard
          </h1>
          <p className="text-gray-600">
            Your AI-powered data platform for analytics and reporting
          </p>
        </div>

        {/* Featured Power BI Reports Section */}
        <div className="mb-12">
          <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl p-8 text-white relative overflow-hidden">
            <div className="absolute inset-0 bg-black/10"></div>
            <div className="relative z-10">
              <div className="flex items-center justify-between">
                <div className="space-y-4 max-w-2xl">
                  <div className="flex items-center gap-3">
                    <BarChart3 className="h-10 w-10" />
                    <h2 className="text-3xl font-bold">Power BI Report Generator</h2>
                  </div>
                  <p className="text-xl opacity-90">
                    Transform your CSV data into stunning, interactive Power BI-style reports with AI-powered insights
                  </p>
                  
                  <div className="flex flex-wrap gap-4 mt-6">
                    <div className="flex items-center gap-2 bg-white/20 rounded-full px-4 py-2">
                      <Upload className="h-4 w-4" />
                      <span className="text-sm font-medium">Easy CSV Upload</span>
                    </div>
                    <div className="flex items-center gap-2 bg-white/20 rounded-full px-4 py-2">
                      <Zap className="h-4 w-4" />
                      <span className="text-sm font-medium">AI-Powered Charts</span>
                    </div>
                    <div className="flex items-center gap-2 bg-white/20 rounded-full px-4 py-2">
                      <TrendingUp className="h-4 w-4" />
                      <span className="text-sm font-medium">Interactive Reports</span>
                    </div>
                  </div>
                </div>
                
                <div className="hidden lg:block">
                  <div className="w-32 h-32 bg-white/20 rounded-full flex items-center justify-center">
                    <BarChart3 className="h-16 w-16" />
                  </div>
                </div>
              </div>
              
              <div className="mt-8 flex gap-4">
                <Link href="/power-bi-reports">
                  <Button size="lg" variant="secondary" className="font-medium">
                    <Upload className="h-5 w-5 mr-2" />
                    Create Your First Report
                    <ArrowRight className="h-4 w-4 ml-2" />
                  </Button>
                </Link>
                <Link href="/power-bi-reports">
                  <Button size="lg" variant="outline" className="text-white border-white hover:bg-white/10">
                    View All Reports
                  </Button>
                </Link>
              </div>
            </div>
          </div>
        </div>

        {/* How It Works Section */}
        <div className="mb-12">
          <h3 className="text-2xl font-bold text-gray-900 mb-6">How Power BI Reports Work</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="border-2 border-blue-100 hover:border-blue-200 transition-colors">
              <CardHeader className="text-center">
                <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Upload className="h-8 w-8 text-blue-600" />
                </div>
                <CardTitle className="text-lg">1. Upload CSV File</CardTitle>
              </CardHeader>
              <CardContent className="text-center">
                <p className="text-gray-600">
                  Simply drag and drop your CSV file or browse to upload. Our system automatically analyzes your data structure and column types.
                </p>
              </CardContent>
            </Card>

            <Card className="border-2 border-green-100 hover:border-green-200 transition-colors">
              <CardHeader className="text-center">
                <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <FileSpreadsheet className="h-8 w-8 text-green-600" />
                </div>
                <CardTitle className="text-lg">2. Select Columns</CardTitle>
              </CardHeader>
              <CardContent className="text-center">
                <p className="text-gray-600">
                  Choose which columns to analyze, configure chart types, and set up your axes. Our AI suggests the best visualization for your data.
                </p>
              </CardContent>
            </Card>

            <Card className="border-2 border-purple-100 hover:border-purple-200 transition-colors">
              <CardHeader className="text-center">
                <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <BarChart3 className="h-8 w-8 text-purple-600" />
                </div>
                <CardTitle className="text-lg">3. Generate Report</CardTitle>
              </CardHeader>
              <CardContent className="text-center">
                <p className="text-gray-600">
                  Get beautiful, interactive Power BI-style reports with charts, tables, and insights. Download or share your reports instantly.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
          <Card>
            <CardContent className="p-6 text-center">
              <div className="text-3xl font-bold text-blue-600 mb-2">∞</div>
              <div className="text-sm text-gray-600">CSV Files Supported</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6 text-center">
              <div className="text-3xl font-bold text-green-600 mb-2">5+</div>
              <div className="text-sm text-gray-600">Chart Types Available</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6 text-center">
              <div className="text-3xl font-bold text-purple-600 mb-2">AI</div>
              <div className="text-sm text-gray-600">Smart Recommendations</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6 text-center">
              <div className="text-3xl font-bold text-orange-600 mb-2">⚡</div>
              <div className="text-sm text-gray-600">Lightning Fast</div>
            </CardContent>
          </Card>
        </div>

        {/* Call to Action */}
        <Card className="bg-gray-100 border-0">
          <CardContent className="p-8 text-center">
            <h3 className="text-2xl font-bold text-gray-900 mb-4">
              Ready to Transform Your Data?
            </h3>
            <p className="text-gray-600 mb-6 max-w-2xl mx-auto">
              Upload your CSV file now and create professional Power BI-style reports in minutes. 
              No technical skills required - our AI does the heavy lifting for you.
            </p>
            <Link href="/power-bi-reports">
              <Button size="lg" className="font-medium">
                <BarChart3 className="h-5 w-5 mr-2" />
                Start Creating Reports
                <ArrowRight className="h-4 w-4 ml-2" />
              </Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default DashboardPage;
