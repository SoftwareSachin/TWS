"use client";
import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
// Simple tabs implementation since Tabs component doesn't exist
import { BarChart3, Upload, FileSpreadsheet, TrendingUp } from "lucide-react";
import CSVUpload from "@/components/power-bi/CSVUpload";
import ColumnSelection from "@/components/power-bi/ColumnSelection";
import ReportDisplay from "@/components/power-bi/ReportDisplay";
import { showError, showSuccess } from "@/utils/toastUtils";

const PowerBIReportsPage = () => {
  const [currentStep, setCurrentStep] = useState(1);
  const [loading, setLoading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [generatedReport, setGeneratedReport] = useState(null);
  const [allReports, setAllReports] = useState([]);
  const [activeTab, setActiveTab] = useState("create");

  useEffect(() => {
    fetchAllReports();
  }, []);

  const fetchAllReports = async () => {
    try {
      const response = await fetch('/api/v1/csv/reports');
      if (response.ok) {
        const data = await response.json();
        setAllReports(data.data || []);
      }
    } catch (error) {
      console.error('Error fetching reports:', error);
    }
  };

  const handleUploadSuccess = (uploadResult) => {
    // uploadResult contains { files: [...], errors: [...] }
    if (uploadResult.files && uploadResult.files.length > 0) {
      // For now, use the first uploaded file
      setUploadedFile(uploadResult.files[0]);
      setCurrentStep(2);
      showSuccess("File uploaded successfully! Select columns to analyze.");
    } else {
      showError("No files were uploaded successfully.");
    }
  };

  const handleGenerateReport = async (reportConfig) => {
    setLoading(true);
    try {
      const response = await fetch('/api/v1/csv/generate-report', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(reportConfig),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate report');
      }

      const report = await response.json();
      setGeneratedReport(report);
      setCurrentStep(3);
      await fetchAllReports(); // Refresh reports list
      showSuccess("Power BI report generated successfully!");
    } catch (error) {
      console.error('Error generating report:', error);
      showError(error.message || 'Failed to generate report');
    } finally {
      setLoading(false);
    }
  };

  const handleStartOver = () => {
    setCurrentStep(1);
    setUploadedFile(null);
    setGeneratedReport(null);
    setActiveTab("create");
  };

  const handleReportDelete = (reportId) => {
    setAllReports(prev => prev.filter(report => report.report_id !== reportId));
  };

  const StepIndicator = ({ currentStep }) => (
    <div className="flex items-center justify-center space-x-4 mb-8">
      {[
        { step: 1, label: "Upload CSV", icon: Upload },
        { step: 2, label: "Select Columns", icon: FileSpreadsheet },
        { step: 3, label: "View Report", icon: BarChart3 },
      ].map(({ step, label, icon: Icon }) => (
        <div key={step} className="flex items-center">
          <div
            className={`flex items-center justify-center w-10 h-10 rounded-full border-2 ${
              currentStep >= step
                ? "bg-blue-600 border-blue-600 text-white"
                : "border-gray-300 text-gray-400"
            }`}
          >
            {currentStep > step ? (
              <span className="text-sm">✓</span>
            ) : (
              <Icon className="h-5 w-5" />
            )}
          </div>
          <span
            className={`ml-2 text-sm font-medium ${
              currentStep >= step ? "text-blue-600" : "text-gray-400"
            }`}
          >
            {label}
          </span>
          {step < 3 && (
            <div
              className={`w-12 h-0.5 ml-4 ${
                currentStep > step ? "bg-blue-600" : "bg-gray-300"
              }`}
            />
          )}
        </div>
      ))}
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
                <BarChart3 className="h-8 w-8 text-blue-600" />
                Power BI Report Generator
              </h1>
              <p className="text-gray-600 mt-2">
                Upload CSV files, select columns, and generate beautiful Power BI-style reports
              </p>
            </div>
            <div className="flex items-center gap-2">
              <span className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
                AI Powered
              </span>
              <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
                Interactive Charts
              </span>
            </div>
          </div>
        </div>

        {/* Simple Tabs Implementation */}
        <div className="space-y-6">
          <div className="grid w-full grid-cols-2 bg-gray-100 p-1 rounded-lg">
            <button
              onClick={() => setActiveTab("create")}
              className={`flex items-center justify-center gap-2 py-2 px-4 rounded-md transition-colors ${
                activeTab === "create" 
                  ? "bg-white text-blue-600 shadow-sm" 
                  : "text-gray-600 hover:text-gray-800"
              }`}
            >
              <Upload className="h-4 w-4" />
              Create New Report
            </button>
            <button
              onClick={() => setActiveTab("reports")}
              className={`flex items-center justify-center gap-2 py-2 px-4 rounded-md transition-colors ${
                activeTab === "reports" 
                  ? "bg-white text-blue-600 shadow-sm" 
                  : "text-gray-600 hover:text-gray-800"
              }`}
            >
              <TrendingUp className="h-4 w-4" />
              View All Reports ({allReports.length})
            </button>
          </div>

          {activeTab === "create" && (
            <div className="space-y-6">
            <StepIndicator currentStep={currentStep} />

            {currentStep === 1 && (
              <div className="max-w-2xl mx-auto">
                <CSVUpload
                  onUploadSuccess={handleUploadSuccess}
                  loading={loading}
                  setLoading={setLoading}
                />
                
                {/* Feature Highlights */}
                <Card className="mt-6">
                  <CardHeader>
                    <CardTitle className="text-lg">What You Can Do</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="text-center p-4">
                        <FileSpreadsheet className="h-8 w-8 mx-auto mb-2 text-blue-500" />
                        <h3 className="font-medium mb-1">CSV Analysis</h3>
                        <p className="text-sm text-gray-600">
                          Upload any CSV file and get instant column analysis and data insights
                        </p>
                      </div>
                      <div className="text-center p-4">
                        <BarChart3 className="h-8 w-8 mx-auto mb-2 text-green-500" />
                        <h3 className="font-medium mb-1">Smart Charts</h3>
                        <p className="text-sm text-gray-600">
                          AI automatically selects the best chart type based on your data
                        </p>
                      </div>
                      <div className="text-center p-4">
                        <TrendingUp className="h-8 w-8 mx-auto mb-2 text-purple-500" />
                        <h3 className="font-medium mb-1">Interactive Reports</h3>
                        <p className="text-sm text-gray-600">
                          Generate interactive Power BI-style reports you can share and download
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            {currentStep === 2 && uploadedFile && (
              <div className="max-w-4xl mx-auto">
                <ColumnSelection
                  fileData={uploadedFile}
                  onGenerateReport={handleGenerateReport}
                  loading={loading}
                />
              </div>
            )}

            {currentStep === 3 && generatedReport && (
              <div className="max-w-6xl mx-auto">
                <div className="mb-6 flex justify-between items-center">
                  <h2 className="text-2xl font-bold">Your Power BI Report</h2>
                  <div className="space-x-2">
                    <Button variant="outline" onClick={() => setActiveTab("reports")}>
                      View All Reports
                    </Button>
                    <Button onClick={handleStartOver}>
                      Create Another Report
                    </Button>
                  </div>
                </div>
                <ReportDisplay
                  report={generatedReport}
                  onDelete={handleReportDelete}
                  onRefresh={fetchAllReports}
                />
              </div>
            )}
            </div>
          )}

          {activeTab === "reports" && (
            <div className="space-y-6">
            <div className="flex justify-between items-center">
              <h2 className="text-2xl font-bold">All Reports ({allReports.length})</h2>
              <Button onClick={() => setActiveTab("create")}>
                <Upload className="h-4 w-4 mr-2" />
                Create New Report
              </Button>
            </div>

            {allReports.length === 0 ? (
              <Card className="text-center py-12">
                <CardContent>
                  <BarChart3 className="h-16 w-16 mx-auto mb-4 text-gray-300" />
                  <h3 className="text-lg font-medium mb-2">No Reports Yet</h3>
                  <p className="text-gray-600 mb-4">
                    Create your first Power BI report by uploading a CSV file
                  </p>
                  <Button onClick={() => setActiveTab("create")}>
                    <Upload className="h-4 w-4 mr-2" />
                    Upload CSV File
                  </Button>
                </CardContent>
              </Card>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {allReports.map((report) => (
                  <ReportDisplay
                    key={report.report_id}
                    report={report}
                    onDelete={handleReportDelete}
                    onRefresh={fetchAllReports}
                  />
                ))}
              </div>
            )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PowerBIReportsPage;