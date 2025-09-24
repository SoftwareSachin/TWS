"use client";
import React, { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Download, Eye, Trash2, Calendar, BarChart3 } from "lucide-react";
import dynamic from "next/dynamic";
import { showSuccess, showError } from "@/utils/toastUtils";

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const ReportDisplay = ({ report, onDelete, onRefresh }) => {
  const [plotlyData, setPlotlyData] = useState(null);

  useEffect(() => {
    if (report?.chart_data) {
      try {
        // Parse the chart data if it's a string
        const chartData = typeof report.chart_data === 'string' 
          ? JSON.parse(report.chart_data) 
          : report.chart_data;
        setPlotlyData(chartData);
      } catch (error) {
        console.error('Error parsing chart data:', error);
      }
    }
  }, [report]);

  const handleDownload = () => {
    if (plotlyData) {
      // Create a downloadable JSON file of the report
      const reportData = {
        report_name: report.report_name,
        created_at: report.created_at,
        columns_used: report.columns_used,
        chart_data: report.chart_data
      };
      
      const dataStr = JSON.stringify(reportData, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(dataBlob);
      
      const link = document.createElement('a');
      link.href = url;
      link.download = `${report.report_name.replace(/\s+/g, '_')}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      
      showSuccess('Report downloaded successfully');
    }
  };

  const handleDelete = async () => {
    try {
      const response = await fetch(`/api/v1/csv/reports/${report.report_id}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error('Failed to delete report');
      }

      showSuccess('Report deleted successfully');
      if (onDelete) onDelete(report.report_id);
      if (onRefresh) onRefresh();
    } catch (error) {
      console.error('Error deleting report:', error);
      showError('Failed to delete report');
    }
  };

  const formatDate = (dateString) => {
    try {
      return new Date(dateString).toLocaleString();
    } catch {
      return dateString;
    }
  };

  if (!report) return null;

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex justify-between items-start">
          <div className="space-y-2">
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              {report.report_name}
            </CardTitle>
            <div className="flex items-center gap-2 text-sm text-gray-600">
              <Calendar className="h-4 w-4" />
              Created: {formatDate(report.created_at)}
            </div>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={handleDownload}>
              <Download className="h-4 w-4" />
            </Button>
            <Button variant="outline" size="sm" onClick={handleDelete}>
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent>
        {/* Report Metadata */}
        <div className="mb-6 p-4 bg-gray-50 rounded-lg">
          <h4 className="font-medium mb-2">Report Details</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <p className="font-medium">Report ID</p>
              <p className="text-gray-600 font-mono text-xs">{report.report_id}</p>
            </div>
            <div>
              <p className="font-medium">Chart Type</p>
              <p className="text-gray-600">{report.chart_type || 'Auto'}</p>
            </div>
            <div>
              <p className="font-medium">Columns Used</p>
              <div className="flex flex-wrap gap-1 mt-1">
                {report.columns_used?.map((column, index) => (
                  <Badge key={index} variant="secondary" className="text-xs">
                    {column}
                  </Badge>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Chart Display */}
        <div className="border rounded-lg p-4 bg-white">
          {plotlyData ? (
            <div className="w-full">
              <Plot
                data={plotlyData.data || []}
                layout={{
                  ...plotlyData.layout,
                  autosize: true,
                  responsive: true,
                  margin: { t: 50, r: 20, b: 50, l: 50 },
                }}
                config={{
                  displayModeBar: true,
                  displaylogo: false,
                  modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
                  responsive: true
                }}
                style={{ width: '100%', height: '500px' }}
                useResizeHandler={true}
              />
            </div>
          ) : (
            <div className="flex items-center justify-center h-64 text-gray-500">
              <div className="text-center">
                <BarChart3 className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                <p>No chart data available</p>
              </div>
            </div>
          )}
        </div>

        {/* Raw Data Preview */}
        {report.chart_data && (
          <details className="mt-4">
            <summary className="cursor-pointer text-sm font-medium text-gray-600 hover:text-gray-800">
              View Raw Chart Data
            </summary>
            <pre className="mt-2 p-3 bg-gray-100 rounded text-xs overflow-auto max-h-40">
              {typeof report.chart_data === 'string' 
                ? report.chart_data 
                : JSON.stringify(report.chart_data, null, 2)}
            </pre>
          </details>
        )}
      </CardContent>
    </Card>
  );
};

export default ReportDisplay;