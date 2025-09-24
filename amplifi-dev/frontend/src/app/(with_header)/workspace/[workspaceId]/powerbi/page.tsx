'use client';

import React, { useState, useCallback, useEffect } from 'react';
import { Upload, FileText, Database, BarChart3, Settings, Play, Download, Search, Filter, Plus, Trash2, Eye, Edit3 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Checkbox } from '@/components/ui/checkbox';
import { useToast } from '@/components/ui/use-toast';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import dynamic from 'next/dynamic';

// Dynamically import Plot component to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface CSVFile {
  file_id: string;
  filename: string;
  columns: string[];
  row_count: number;
  column_types: Record<string, string>;
  potential_keys: string[];
  preview_data: any[];
}

interface Relationship {
  from_table: string;
  from_column: string;
  to_table: string;
  to_column: string;
  relationship_type: string;
  confidence: number;
  match_count: number;
}

interface Dashboard {
  dashboard_id: string;
  name: string;
  description: string;
  charts: any[];
  created_at: string;
  total_charts: number;
}

export default function PowerBIPage({ params }: { params: { workspaceId: string } }) {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState('upload');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [csvFiles, setCsvFiles] = useState<CSVFile[]>([]);
  const [relationships, setRelationships] = useState<Relationship[]>([]);
  const [graphData, setGraphData] = useState<any>(null);
  const [dashboards, setDashboards] = useState<Dashboard[]>([]);
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [queryText, setQueryText] = useState('');
  const [queryResult, setQueryResult] = useState<any>(null);
  const [selectedColumns, setSelectedColumns] = useState<Record<string, string[]>>({});

  // Initialize session
  useEffect(() => {
    createSession();
  }, []);

  const createSession = async () => {
    try {
      const response = await fetch('/api/v1/csv/session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: `PowerBI Session - ${new Date().toLocaleDateString()}`,
          description: 'Professional PowerBI analysis session',
          workspace_id: params.workspaceId
        })
      });
      
      const data = await response.json();
      setSessionId(data.session_id);
      toast({
        title: "Session Created",
        description: "PowerBI analysis session initialized successfully."
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to create PowerBI session.",
        variant: "destructive"
      });
    }
  };

  const handleFileUpload = useCallback(async (files: FileList) => {
    if (!sessionId) return;
    
    setIsUploading(true);
    setUploadProgress(0);
    
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (!file.name.toLowerCase().endsWith('.csv')) {
        toast({
          title: "Invalid File",
          description: `${file.name} is not a CSV file.`,
          variant: "destructive"
        });
        continue;
      }

      try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`/api/v1/csv/session/${sessionId}/upload`, {
          method: 'POST',
          body: formData
        });
        
        if (response.ok) {
          const fileData = await response.json();
          setCsvFiles(prev => [...prev, fileData]);
          toast({
            title: "File Uploaded",
            description: `${file.name} uploaded successfully.`
          });
        }
        
        setUploadProgress(((i + 1) / files.length) * 100);
      } catch (error) {
        toast({
          title: "Upload Error",
          description: `Failed to upload ${file.name}.`,
          variant: "destructive"
        });
      }
    }
    
    setIsUploading(false);
  }, [sessionId, toast]);

  const detectRelationships = async () => {
    if (!sessionId || csvFiles.length < 2) {
      toast({
        title: "Insufficient Data",
        description: "At least 2 CSV files are required for relationship detection.",
        variant: "destructive"
      });
      return;
    }

    try {
      const response = await fetch(`/api/v1/csv/session/${sessionId}/detect-relationships`, {
        method: 'POST'
      });
      
      const data = await response.json();
      setRelationships(data.relationships);
      setGraphData(data.graph_data);
      
      toast({
        title: "Relationships Detected",
        description: `Found ${data.total_relationships} potential relationships.`
      });
      setActiveTab('relationships');
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to detect relationships.",
        variant: "destructive"
      });
    }
  };

  const generateQueryReport = async () => {
    if (!sessionId || !queryText.trim()) {
      toast({
        title: "Invalid Query",
        description: "Please enter a query to generate a report.",
        variant: "destructive"
      });
      return;
    }

    try {
      const response = await fetch('/api/v1/csv/query-report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          query: queryText
        })
      });
      
      const data = await response.json();
      setQueryResult(data);
      
      toast({
        title: "Report Generated",
        description: "Query-based report created successfully."
      });
      setActiveTab('reports');
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to generate query report.",
        variant: "destructive"
      });
    }
  };

  const createDashboard = async () => {
    if (!sessionId || Object.keys(selectedColumns).length === 0) {
      toast({
        title: "No Data Selected",
        description: "Please select columns to create a dashboard.",
        variant: "destructive"
      });
      return;
    }

    try {
      const charts = Object.entries(selectedColumns).map(([fileId, columns]) => {
        const file = csvFiles.find(f => f.file_id === fileId);
        return {
          table_id: fileId,
          type: 'bar',
          title: `Analysis for ${file?.filename}`,
          x_column: columns[0],
          y_column: columns[1] || columns[0],
          filters: [],
          aggregation: {
            group_by: columns[0],
            metric: columns[1] || columns[0],
            operation: 'count'
          }
        };
      });

      const response = await fetch('/api/v1/csv/dashboard/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          name: `Dashboard - ${new Date().toLocaleDateString()}`,
          description: 'Professional PowerBI Dashboard',
          charts
        })
      });
      
      const dashboard = await response.json();
      setDashboards(prev => [...prev, dashboard]);
      
      toast({
        title: "Dashboard Created",
        description: "Professional dashboard generated successfully."
      });
      setActiveTab('dashboards');
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to create dashboard.",
        variant: "destructive"
      });
    }
  };

  const EntityRelationshipGraph = () => {
    if (!graphData || !graphData.nodes) return null;

    const plotData = [
      {
        x: graphData.nodes.map((n: any) => n.x),
        y: graphData.nodes.map((n: any) => n.y),
        text: graphData.nodes.map((n: any) => `${n.label}<br>Columns: ${n.columns.length}<br>Rows: ${n.row_count}`),
        mode: 'markers+text',
        type: 'scatter',
        marker: {
          size: graphData.nodes.map((n: any) => n.size),
          color: '#374AF1',
          line: { width: 2, color: '#ffffff' }
        },
        textposition: 'top center',
        textfont: { size: 12, color: '#374151' }
      }
    ];

    // Add relationship lines
    if (graphData.edges) {
      graphData.edges.forEach((edge: any) => {
        const fromNode = graphData.nodes.find((n: any) => n.id === edge.from);
        const toNode = graphData.nodes.find((n: any) => n.id === edge.to);
        
        if (fromNode && toNode) {
          plotData.push({
            x: [fromNode.x, toNode.x],
            y: [fromNode.y, toNode.y],
            mode: 'lines',
            type: 'scatter',
            line: { 
              width: Math.max(2, edge.weight * 5),
              color: edge.color
            },
            showlegend: false,
            hoverinfo: 'text',
            text: `${edge.from_column} → ${edge.to_column}<br>Confidence: ${(edge.weight * 100).toFixed(1)}%`
          });
        }
      });
    }

    const layout = {
      title: {
        text: 'Entity Relationship Graph',
        font: { size: 16, color: '#111827' }
      },
      xaxis: { showgrid: false, zeroline: false, showticklabels: false },
      yaxis: { showgrid: false, zeroline: false, showticklabels: false },
      showlegend: false,
      plot_bgcolor: '#fafafa',
      paper_bgcolor: '#ffffff',
      font: { family: 'Inter, sans-serif' },
      margin: { t: 60, r: 30, b: 30, l: 30 }
    };

    return (
      <div className="w-full h-96">
        <Plot
          data={plotData as any}
          layout={layout}
          style={{ width: '100%', height: '100%' }}
          config={{ displayModeBar: false, responsive: true }}
        />
      </div>
    );
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">PowerBI Analytics</h1>
          <p className="text-gray-600 mt-1">Professional data analysis and visualization platform</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={detectRelationships} disabled={csvFiles.length < 2}>
            <Database className="w-4 h-4 mr-2" />
            Detect Relationships
          </Button>
          <Button onClick={createDashboard} disabled={Object.keys(selectedColumns).length === 0}>
            <BarChart3 className="w-4 h-4 mr-2" />
            Create Dashboard
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="upload" className="flex items-center gap-2">
            <Upload className="w-4 h-4" />
            Upload Data
          </TabsTrigger>
          <TabsTrigger value="relationships" className="flex items-center gap-2">
            <Database className="w-4 h-4" />
            Relationships
          </TabsTrigger>
          <TabsTrigger value="columns" className="flex items-center gap-2">
            <Settings className="w-4 h-4" />
            Select Columns
          </TabsTrigger>
          <TabsTrigger value="dashboards" className="flex items-center gap-2">
            <BarChart3 className="w-4 h-4" />
            Dashboards
          </TabsTrigger>
          <TabsTrigger value="query" className="flex items-center gap-2">
            <Search className="w-4 h-4" />
            Query Reports
          </TabsTrigger>
        </TabsList>

        <TabsContent value="upload" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="w-5 h-5" />
                CSV File Upload
              </CardTitle>
              <CardDescription>
                Upload multiple CSV files to analyze relationships and create professional dashboards
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div 
                  className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors cursor-pointer"
                  onDrop={(e) => {
                    e.preventDefault();
                    handleFileUpload(e.dataTransfer.files);
                  }}
                  onDragOver={(e) => e.preventDefault()}
                  onClick={() => document.getElementById('file-upload')?.click()}
                >
                  <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-lg font-medium text-gray-700">Drop CSV files here or click to browse</p>
                  <p className="text-sm text-gray-500 mt-2">Supports multiple file upload for relationship analysis</p>
                  <input
                    id="file-upload"
                    type="file"
                    multiple
                    accept=".csv"
                    className="hidden"
                    onChange={(e) => e.target.files && handleFileUpload(e.target.files)}
                  />
                </div>
                
                {isUploading && (
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm text-gray-600">
                      <span>Uploading files...</span>
                      <span>{Math.round(uploadProgress)}%</span>
                    </div>
                    <Progress value={uploadProgress} className="w-full" />
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {csvFiles.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Uploaded Files ({csvFiles.length})</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4">
                  {csvFiles.map((file) => (
                    <div key={file.file_id} className="flex items-center justify-between p-4 border rounded-lg">
                      <div className="flex items-center gap-3">
                        <FileText className="w-5 h-5 text-blue-600" />
                        <div>
                          <p className="font-medium">{file.filename}</p>
                          <p className="text-sm text-gray-500">
                            {file.row_count.toLocaleString()} rows, {file.columns.length} columns
                          </p>
                        </div>
                      </div>
                      <div className="flex gap-2">
                        {file.potential_keys.map((key) => (
                          <Badge key={key} variant="secondary">{key}</Badge>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="relationships" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database className="w-5 h-5" />
                Entity Relationship Graph
              </CardTitle>
              <CardDescription>
                Visual representation of relationships between your data tables
              </CardDescription>
            </CardHeader>
            <CardContent>
              {graphData ? (
                <EntityRelationshipGraph />
              ) : (
                <div className="text-center py-12">
                  <Database className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                  <p className="text-gray-500">No relationships detected yet. Upload files and click "Detect Relationships".</p>
                </div>
              )}
            </CardContent>
          </Card>

          {relationships.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Detected Relationships ({relationships.length})</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {relationships.slice(0, 10).map((rel, index) => (
                    <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                      <div className="flex-1">
                        <p className="font-medium">
                          {rel.from_column} ({csvFiles.find(f => f.file_id === rel.from_table)?.filename})
                          →
                          {rel.to_column} ({csvFiles.find(f => f.file_id === rel.to_table)?.filename})
                        </p>
                        <p className="text-sm text-gray-500">{rel.relationship_type}</p>
                      </div>
                      <div className="text-right">
                        <Badge variant={rel.confidence > 0.8 ? "default" : "secondary"}>
                          {(rel.confidence * 100).toFixed(1)}%
                        </Badge>
                        <p className="text-xs text-gray-500 mt-1">{rel.match_count} matches</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="columns" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="w-5 h-5" />
                Column Selection
              </CardTitle>
              <CardDescription>
                Select columns from your data tables to include in analysis and dashboards
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {csvFiles.map((file) => (
                  <div key={file.file_id} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="font-medium">{file.filename}</h3>
                      <Badge variant="outline">{file.columns.length} columns</Badge>
                    </div>
                    <div className="grid grid-cols-3 gap-3">
                      {file.columns.map((column) => (
                        <div key={column} className="flex items-center space-x-2">
                          <Checkbox
                            id={`${file.file_id}-${column}`}
                            checked={selectedColumns[file.file_id]?.includes(column) || false}
                            onCheckedChange={(checked) => {
                              if (checked) {
                                setSelectedColumns(prev => ({
                                  ...prev,
                                  [file.file_id]: [...(prev[file.file_id] || []), column]
                                }));
                              } else {
                                setSelectedColumns(prev => ({
                                  ...prev,
                                  [file.file_id]: (prev[file.file_id] || []).filter(c => c !== column)
                                }));
                              }
                            }}
                          />
                          <label htmlFor={`${file.file_id}-${column}`} className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">
                            {column}
                          </label>
                          <Badge variant="secondary" className="text-xs">
                            {file.column_types[column]}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="dashboards" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5" />
                Professional Dashboards
              </CardTitle>
              <CardDescription>
                Create and manage professional PowerBI-style dashboards
              </CardDescription>
            </CardHeader>
            <CardContent>
              {dashboards.length > 0 ? (
                <div className="grid gap-4">
                  {dashboards.map((dashboard) => (
                    <div key={dashboard.dashboard_id} className="border rounded-lg p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <h3 className="font-medium">{dashboard.name}</h3>
                          <p className="text-sm text-gray-500">{dashboard.description}</p>
                        </div>
                        <div className="flex gap-2">
                          <Badge variant="outline">{dashboard.total_charts} charts</Badge>
                          <Button size="sm" variant="outline">
                            <Eye className="w-4 h-4 mr-1" />
                            View
                          </Button>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        {dashboard.charts.slice(0, 2).map((chart: any) => (
                          <div key={chart.id} className="h-48 border rounded bg-gray-50 flex items-center justify-center">
                            <p className="text-gray-500">{chart.title}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <BarChart3 className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                  <p className="text-gray-500">No dashboards created yet. Select columns and create your first dashboard.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="query" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Search className="w-5 h-5" />
                Natural Language Queries
              </CardTitle>
              <CardDescription>
                Generate reports using natural language queries like "show me top 5 users" or "analyze sales trends"
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex gap-3">
                  <Textarea
                    placeholder="Enter your query (e.g., 'show me top 5 users by revenue', 'analyze trends over time', 'compare categories')"
                    value={queryText}
                    onChange={(e) => setQueryText(e.target.value)}
                    className="flex-1"
                    rows={3}
                  />
                  <Button onClick={generateQueryReport} disabled={!queryText.trim() || !sessionId}>
                    <Play className="w-4 h-4 mr-2" />
                    Generate Report
                  </Button>
                </div>
                
                <div className="flex flex-wrap gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setQueryText("show me top 5 users")}
                  >
                    Top 5 Users
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setQueryText("analyze sales by category")}
                  >
                    Sales by Category
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setQueryText("show revenue trends")}
                  >
                    Revenue Trends
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setQueryText("compare regional performance")}
                  >
                    Regional Comparison
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {queryResult && (
            <Card>
              <CardHeader>
                <CardTitle>Query Results</CardTitle>
                <CardDescription>Generated report for: "{queryResult.query}"</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {queryResult.charts && queryResult.charts.map((chart: any) => (
                    <div key={chart.id} className="border rounded-lg p-4">
                      <h3 className="font-medium mb-3">{chart.title}</h3>
                      <div className="h-64 bg-gray-50 rounded flex items-center justify-center">
                        <p className="text-gray-500">Chart: {chart.type}</p>
                      </div>
                    </div>
                  ))}
                  
                  {!queryResult.charts?.length && (
                    <div className="text-center py-8">
                      <p className="text-gray-500">Query processed. Results are being prepared...</p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}