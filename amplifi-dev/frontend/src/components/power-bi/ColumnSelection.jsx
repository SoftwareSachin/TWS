"use client";
import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { BarChart3, PieChart, TrendingUp, Table, Zap } from "lucide-react";
import { showError } from "@/utils/toastUtils";

const ColumnSelection = ({ fileData, onGenerateReport, loading }) => {
  const [selectedColumns, setSelectedColumns] = useState([]);
  const [chartType, setChartType] = useState("auto");
  const [xAxis, setXAxis] = useState("");
  const [yAxis, setYAxis] = useState([]);
  const [columnDetails, setColumnDetails] = useState(null);

  useEffect(() => {
    if (fileData?.file_id) {
      fetchColumnDetails();
    }
  }, [fileData]);

  const fetchColumnDetails = async () => {
    try {
      const response = await fetch(`/api/v1/csv/${fileData.file_id}/columns`);
      if (!response.ok) throw new Error('Failed to fetch column details');
      
      const data = await response.json();
      setColumnDetails(data);
    } catch (error) {
      console.error('Error fetching column details:', error);
      showError('Failed to fetch column details');
    }
  };

  const handleColumnToggle = (columnName, checked) => {
    if (checked) {
      setSelectedColumns(prev => [...prev, columnName]);
    } else {
      setSelectedColumns(prev => prev.filter(col => col !== columnName));
      // Remove from axis selections if unchecked
      if (xAxis === columnName) setXAxis("");
      if (yAxis.includes(columnName)) {
        setYAxis(prev => prev.filter(axis => axis !== columnName));
      }
    }
  };

  const handleGenerateReport = () => {
    if (selectedColumns.length === 0) {
      showError("Please select at least one column");
      return;
    }

    const reportConfig = {
      file_id: fileData.file_id,
      selected_columns: selectedColumns,
      chart_type: chartType,
      x_axis: xAxis || undefined,
      y_axis: yAxis.length > 0 ? yAxis : undefined,
    };

    onGenerateReport(reportConfig);
  };

  const getColumnTypeIcon = (type) => {
    switch (type) {
      case 'numeric': return '🔢';
      case 'text': return '📝';
      case 'date': return '📅';
      default: return '❓';
    }
  };

  const getChartTypeIcon = (type) => {
    switch (type) {
      case 'bar': return <BarChart3 className="h-4 w-4" />;
      case 'pie': return <PieChart className="h-4 w-4" />;
      case 'line': return <TrendingUp className="h-4 w-4" />;
      case 'scatter': return <BarChart3 className="h-4 w-4" />;
      case 'table': return <Table className="h-4 w-4" />;
      default: return <Zap className="h-4 w-4" />;
    }
  };

  if (!fileData) return null;

  const numericColumns = selectedColumns.filter(col => 
    columnDetails?.columns[col]?.type === 'numeric'
  );
  const textColumns = selectedColumns.filter(col => 
    columnDetails?.columns[col]?.type === 'text'
  );

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>File Information</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="font-medium">File Name</p>
              <p className="text-gray-600">{fileData.filename}</p>
            </div>
            <div>
              <p className="font-medium">Rows</p>
              <p className="text-gray-600">{fileData.row_count.toLocaleString()}</p>
            </div>
            <div>
              <p className="font-medium">Columns</p>
              <p className="text-gray-600">{fileData.columns.length}</p>
            </div>
            <div>
              <p className="font-medium">File Size</p>
              <p className="text-gray-600">{(fileData.file_size / 1024).toFixed(1)} KB</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Select Columns for Analysis</CardTitle>
          <p className="text-sm text-gray-600">
            Choose the columns you want to include in your Power BI report
          </p>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {fileData.columns.map((column) => {
              const columnInfo = columnDetails?.columns[column];
              return (
                <div key={column} className="flex items-start space-x-3 p-3 border rounded-lg hover:bg-gray-50">
                  <Checkbox
                    id={column}
                    checked={selectedColumns.includes(column)}
                    onCheckedChange={(checked) => handleColumnToggle(column, checked)}
                  />
                  <div className="flex-1 min-w-0">
                    <label htmlFor={column} className="font-medium cursor-pointer">
                      {getColumnTypeIcon(columnInfo?.type)} {column}
                    </label>
                    <Badge variant="outline" className="ml-2 text-xs">
                      {columnInfo?.type || 'unknown'}
                    </Badge>
                    {columnInfo && (
                      <div className="mt-1 text-xs text-gray-500">
                        <p>Unique: {columnInfo.unique_count} | Nulls: {columnInfo.null_count}</p>
                        {columnInfo.type === 'numeric' && columnInfo.mean && (
                          <p>Range: {columnInfo.min?.toFixed(2)} - {columnInfo.max?.toFixed(2)} (avg: {columnInfo.mean?.toFixed(2)})</p>
                        )}
                        {columnInfo.sample_values && columnInfo.sample_values.length > 0 && (
                          <p>Sample: {columnInfo.sample_values.slice(0, 3).join(', ')}...</p>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {selectedColumns.length > 0 && (
            <div className="mt-6 p-4 bg-blue-50 rounded-lg">
              <h4 className="font-medium mb-2">Selected Columns ({selectedColumns.length})</h4>
              <div className="flex flex-wrap gap-2">
                {selectedColumns.map((column) => (
                  <Badge key={column} variant="default" className="text-xs">
                    {getColumnTypeIcon(columnDetails?.columns[column]?.type)} {column}
                  </Badge>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {selectedColumns.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Chart Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Chart Type</label>
              <Select value={chartType} onValueChange={setChartType}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="auto">
                    <div className="flex items-center gap-2">
                      <Zap className="h-4 w-4" />
                      Auto-detect
                    </div>
                  </SelectItem>
                  <SelectItem value="bar">
                    <div className="flex items-center gap-2">
                      <BarChart3 className="h-4 w-4" />
                      Bar Chart
                    </div>
                  </SelectItem>
                  <SelectItem value="line">
                    <div className="flex items-center gap-2">
                      <TrendingUp className="h-4 w-4" />
                      Line Chart
                    </div>
                  </SelectItem>
                  <SelectItem value="scatter">
                    <div className="flex items-center gap-2">
                      <BarChart3 className="h-4 w-4" />
                      Scatter Plot
                    </div>
                  </SelectItem>
                  <SelectItem value="pie">
                    <div className="flex items-center gap-2">
                      <PieChart className="h-4 w-4" />
                      Pie Chart
                    </div>
                  </SelectItem>
                  <SelectItem value="table">
                    <div className="flex items-center gap-2">
                      <Table className="h-4 w-4" />
                      Data Table
                    </div>
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            {(chartType === "bar" || chartType === "line" || chartType === "scatter") && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">X-Axis</label>
                  <Select value={xAxis} onValueChange={setXAxis}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select X-axis column" />
                    </SelectTrigger>
                    <SelectContent>
                      {selectedColumns.map((column) => (
                        <SelectItem key={column} value={column}>
                          {getColumnTypeIcon(columnDetails?.columns[column]?.type)} {column}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Y-Axis</label>
                  <Select 
                    value={yAxis[0] || ""} 
                    onValueChange={(value) => setYAxis(value ? [value] : [])}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select Y-axis column" />
                    </SelectTrigger>
                    <SelectContent>
                      {selectedColumns.map((column) => (
                        <SelectItem key={column} value={column}>
                          {getColumnTypeIcon(columnDetails?.columns[column]?.type)} {column}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
            )}

            <Button 
              onClick={handleGenerateReport}
              disabled={loading || selectedColumns.length === 0}
              className="w-full"
              size="lg"
            >
              {getChartTypeIcon(chartType)}
              <span className="ml-2">
                {loading ? "Generating Report..." : "Generate Power BI Report"}
              </span>
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default ColumnSelection;