"use client";
import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  Search, 
  MessageSquare, 
  TrendingUp, 
  Users, 
  BarChart3, 
  PieChart, 
  Activity,
  Lightbulb,
  Sparkles
} from "lucide-react";
import { showError, showSuccess } from "@/utils/toastUtils";

const QueryBasedReports = ({ uploadedFiles, onReportGenerated }) => {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [generatedReport, setGeneratedReport] = useState(null);

  const queryExamples = [
    {
      icon: <TrendingUp className="h-4 w-4" />,
      text: "Show me the top 10 customers by revenue",
      category: "Top Analysis"
    },
    {
      icon: <Users className="h-4 w-4" />,
      text: "Top 5 users with most activity",
      category: "User Analysis"
    },
    {
      icon: <BarChart3 className="h-4 w-4" />,
      text: "Monthly sales trend for last 6 months",
      category: "Trend Analysis"
    },
    {
      icon: <PieChart className="h-4 w-4" />,
      text: "Distribution of products by category",
      category: "Distribution"
    },
    {
      icon: <Activity className="h-4 w-4" />,
      text: "Average order value by customer segment",
      category: "Metrics"
    }
  ];

  const handleQuerySubmit = async (queryText = query) => {
    if (!queryText.trim() || !uploadedFiles || uploadedFiles.length === 0) {
      showError("Please enter a query and ensure files are uploaded");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/v1/csv/generate-query-report', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          file_ids: uploadedFiles.map(f => f.file_id),
          query: queryText,
          report_type: "auto"
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate query-based report');
      }

      const report = await response.json();
      setGeneratedReport(report);
      
      if (onReportGenerated) {
        onReportGenerated(report);
      }

      showSuccess("Query-based report generated successfully!");
    } catch (error) {
      console.error('Error generating query report:', error);
      showError(error.message || 'Failed to generate query-based report');
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (example) => {
    setQuery(example.text);
    handleQuerySubmit(example.text);
  };

  if (!uploadedFiles || uploadedFiles.length === 0) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-gray-500">
            <MessageSquare className="h-5 w-5" />
            Query-Based Reports
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500">
            <Search className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>Upload CSV files to enable query-based report generation</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <MessageSquare className="h-5 w-5" />
            AI-Powered Query Reports
          </div>
          <Badge variant="secondary" className="flex items-center gap-1">
            <Sparkles className="h-3 w-3" />
            Smart Analytics
          </Badge>
        </CardTitle>
        <p className="text-sm text-gray-600">
          Ask questions about your data in natural language and get instant visualizations
        </p>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Query Input */}
        <div className="space-y-4">
          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium text-gray-700">
              What would you like to analyze?
            </label>
            <div className="flex gap-2">
              <div className="relative flex-1">
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Ask a question about your data... (e.g., 'Show me the top 5 customers by revenue')"
                  className="w-full p-3 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  rows={3}
                />
                <div className="absolute bottom-3 right-3">
                  <Lightbulb className="h-4 w-4 text-gray-400" />
                </div>
              </div>
            </div>
          </div>
          
          <Button
            onClick={() => handleQuerySubmit()}
            disabled={loading || !query.trim()}
            className="w-full"
            size="lg"
          >
            {loading ? (
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                Analyzing Data...
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <Search className="h-4 w-4" />
                Generate Report
              </div>
            )}
          </Button>
        </div>

        {/* Query Examples */}
        <div className="space-y-3">
          <h4 className="font-medium text-gray-900 flex items-center gap-2">
            <Lightbulb className="h-4 w-4" />
            Try These Examples
          </h4>
          <div className="grid gap-3">
            {queryExamples.map((example, index) => (
              <button
                key={index}
                onClick={() => handleExampleClick(example)}
                disabled={loading}
                className="flex items-center gap-3 p-3 bg-gray-50 hover:bg-gray-100 rounded-lg border text-left transition-colors disabled:opacity-50"
              >
                <div className="p-2 bg-blue-100 text-blue-600 rounded-lg">
                  {example.icon}
                </div>
                <div className="flex-1">
                  <div className="font-medium text-sm text-gray-900">
                    {example.text}
                  </div>
                  <div className="text-xs text-gray-500">
                    {example.category}
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Data Context */}
        <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
          <h4 className="font-medium text-blue-900 mb-2 flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Available Data Sources
          </h4>
          <div className="grid gap-2">
            {uploadedFiles.map((file, index) => (
              <div key={index} className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <span className="font-medium text-blue-800">
                    {file.filename.replace('.csv', '')}
                  </span>
                </div>
                <div className="text-blue-600">
                  {file.row_count} rows, {file.columns.length} columns
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Generated Report Preview */}
        {generatedReport && (
          <div className="space-y-4">
            <h4 className="font-medium text-gray-900 flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Query Results
            </h4>
            <div className="p-4 bg-white border rounded-lg">
              <div className="mb-3">
                <Badge variant="outline" className="mb-2">
                  Query: "{generatedReport.query}"
                </Badge>
              </div>
              
              {generatedReport.chart_config && (
                <div className="space-y-3">
                  <h5 className="font-medium text-sm">
                    {generatedReport.chart_config.title}
                  </h5>
                  
                  {generatedReport.chart_config.type === "summary" ? (
                    <div className="grid grid-cols-2 gap-4">
                      {Object.entries(generatedReport.result_data).map(([file, stats]) => (
                        <div key={file} className="p-3 bg-gray-50 rounded-lg">
                          <div className="font-medium text-sm mb-2">{file}</div>
                          <div className="space-y-1 text-xs text-gray-600">
                            <div>Rows: {stats.rows}</div>
                            <div>Columns: {stats.columns}</div>
                            <div>Numeric: {stats.numeric_columns}</div>
                            <div>Text: {stats.text_columns}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : generatedReport.chart_config.type === "bar" && generatedReport.chart_config.data ? (
                    <div className="space-y-2">
                      {generatedReport.chart_config.data.slice(0, 5).map((item, index) => (
                        <div key={index} className="flex items-center justify-between">
                          <span className="text-sm">{item.x}</span>
                          <div className="flex items-center gap-2">
                            <div 
                              className="bg-blue-500 h-2 rounded" 
                              style={{ width: `${(item.y / Math.max(...generatedReport.chart_config.data.map(d => d.y))) * 100}px` }}
                            />
                            <span className="text-sm font-medium">{item.y}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-sm text-gray-600">
                      <pre className="bg-gray-50 p-3 rounded text-xs overflow-auto">
                        {JSON.stringify(generatedReport.result_data, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {loading && (
          <div className="text-center py-8">
            <div className="flex items-center justify-center gap-2 text-blue-600">
              <div className="w-5 h-5 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
              <span>Processing your query and analyzing data...</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default QueryBasedReports;