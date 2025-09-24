"use client";
import React, { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  Network, 
  Table, 
  Key, 
  ArrowRight, 
  Database, 
  Maximize2, 
  ZoomIn, 
  ZoomOut,
  RefreshCw 
} from "lucide-react";
import { showError, showSuccess } from "@/utils/toastUtils";

const EntityRelationshipGraph = ({ uploadedFiles, onRelationshipsDetected }) => {
  const [relationships, setRelationships] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedTable, setSelectedTable] = useState(null);
  const [graphData, setGraphData] = useState(null);
  const svgRef = useRef(null);

  useEffect(() => {
    if (uploadedFiles && uploadedFiles.length > 1) {
      detectRelationships();
    }
  }, [uploadedFiles]);

  const detectRelationships = async () => {
    if (!uploadedFiles || uploadedFiles.length < 2) {
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/v1/csv/detect-relationships', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          file_ids: uploadedFiles.map(f => f.file_id)
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to detect relationships');
      }

      const data = await response.json();
      setRelationships(data.relationships || []);
      setGraphData(data.graph_data || null);
      
      if (onRelationshipsDetected) {
        onRelationshipsDetected(data);
      }

      showSuccess(`Detected ${data.relationships?.length || 0} potential relationships`);
    } catch (error) {
      console.error('Error detecting relationships:', error);
      showError('Failed to detect relationships between files');
    } finally {
      setLoading(false);
    }
  };

  const getTableColor = (tableId) => {
    const colors = [
      '#3B82F6', // Blue
      '#10B981', // Green  
      '#F59E0B', // Orange
      '#EF4444', // Red
      '#8B5CF6', // Purple
      '#06B6D4', // Cyan
      '#84CC16', // Lime
      '#F97316', // Orange-red
    ];
    const index = uploadedFiles.findIndex(f => f.file_id === tableId);
    return colors[index % colors.length];
  };

  const renderTableNode = (file, x, y) => {
    const color = getTableColor(file.file_id);
    const isSelected = selectedTable === file.file_id;
    
    return (
      <g
        key={file.file_id}
        transform={`translate(${x}, ${y})`}
        className="cursor-pointer"
        onClick={() => setSelectedTable(isSelected ? null : file.file_id)}
      >
        {/* Table box */}
        <rect
          width="200"
          height={Math.max(120, file.columns.length * 20 + 60)}
          fill="white"
          stroke={color}
          strokeWidth={isSelected ? 3 : 2}
          rx="8"
          className="drop-shadow-md"
        />
        
        {/* Table header */}
        <rect
          width="200"
          height="40"
          fill={color}
          rx="8"
        />
        <rect
          width="200"
          height="32"
          fill={color}
          rx="0"
        />
        
        {/* Table name */}
        <text
          x="100"
          y="25"
          textAnchor="middle"
          fill="white"
          className="font-semibold text-sm"
        >
          {file.filename.replace('.csv', '')}
        </text>
        
        {/* Row count */}
        <text
          x="100"
          y="55"
          textAnchor="middle"
          fill="#666"
          className="text-xs"
        >
          {file.row_count} rows
        </text>
        
        {/* Columns (show first 8) */}
        {file.columns.slice(0, 8).map((column, index) => (
          <g key={column}>
            <text
              x="15"
              y={80 + index * 18}
              fill="#333"
              className="text-xs font-mono"
            >
              {column}
            </text>
            {/* Key icon for potential key columns */}
            {(column.toLowerCase().includes('id') || column.toLowerCase().includes('key')) && (
              <circle
                cx="185"
                cy={75 + index * 18}
                r="6"
                fill="#F59E0B"
                className="opacity-80"
              />
            )}
          </g>
        ))}
        
        {/* Show more indicator */}
        {file.columns.length > 8 && (
          <text
            x="100"
            y={80 + 8 * 18}
            textAnchor="middle"
            fill="#666"
            className="text-xs"
          >
            ... {file.columns.length - 8} more columns
          </text>
        )}
      </g>
    );
  };

  const renderRelationshipLine = (rel) => {
    const fromFile = uploadedFiles.find(f => f.file_id === rel.from_table);
    const toFile = uploadedFiles.find(f => f.file_id === rel.to_table);
    
    if (!fromFile || !toFile) return null;

    const fromIndex = uploadedFiles.findIndex(f => f.file_id === rel.from_table);
    const toIndex = uploadedFiles.findIndex(f => f.file_id === rel.to_table);
    
    // Calculate positions (simplified layout)
    const fromX = 100 + (fromIndex * 250);
    const fromY = 150;
    const toX = 100 + (toIndex * 250);
    const toY = 150;
    
    return (
      <g key={`${rel.from_table}-${rel.to_table}`}>
        {/* Relationship line */}
        <line
          x1={fromX + 200}
          y1={fromY + 60}
          x2={toX}
          y2={toY + 60}
          stroke="#6B7280"
          strokeWidth="2"
          strokeDasharray={rel.confidence < 0.8 ? "5,5" : "none"}
        />
        
        {/* Arrow */}
        <polygon
          points={`${toX-8},${toY + 60 - 4} ${toX},${toY + 60} ${toX-8},${toY + 60 + 4}`}
          fill="#6B7280"
        />
        
        {/* Relationship label */}
        <text
          x={(fromX + 200 + toX) / 2}
          y={((fromY + 60) + (toY + 60)) / 2 - 10}
          textAnchor="middle"
          fill="#374151"
          className="text-xs font-medium"
        >
          {rel.relationship_type}
        </text>
        
        {/* Confidence score */}
        <text
          x={(fromX + 200 + toX) / 2}
          y={((fromY + 60) + (toY + 60)) / 2 + 5}
          textAnchor="middle"
          fill="#6B7280"
          className="text-xs"
        >
          {Math.round(rel.confidence * 100)}%
        </text>
      </g>
    );
  };

  if (!uploadedFiles || uploadedFiles.length < 2) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-gray-500">
            <Network className="h-5 w-5" />
            Entity Relationship Graph
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500">
            <Database className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>Upload at least 2 CSV files to generate relationship graph</p>
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
            <Network className="h-5 w-5" />
            Entity Relationship Graph
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={detectRelationships}
              disabled={loading}
            >
              {loading ? (
                <RefreshCw className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4" />
              )}
              Refresh
            </Button>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Relationship Summary */}
        <div className="flex items-center gap-4 p-4 bg-blue-50 rounded-lg border">
          <div className="grid grid-cols-3 gap-4 flex-1">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {uploadedFiles.length}
              </div>
              <div className="text-sm text-gray-600">Tables</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {relationships.length}
              </div>
              <div className="text-sm text-gray-600">Relationships</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                {uploadedFiles.reduce((sum, f) => sum + f.columns.length, 0)}
              </div>
              <div className="text-sm text-gray-600">Total Columns</div>
            </div>
          </div>
        </div>

        {/* Graph Visualization */}
        <div className="border rounded-lg bg-white">
          <div className="p-4 border-b bg-gray-50">
            <h4 className="font-medium flex items-center gap-2">
              <Table className="h-4 w-4" />
              Database Schema Visualization
            </h4>
          </div>
          <div className="p-4">
            <svg
              ref={svgRef}
              width="100%"
              height="400"
              viewBox={`0 0 ${uploadedFiles.length * 250 + 100} 400`}
              className="border rounded"
            >
              {/* Render table nodes */}
              {uploadedFiles.map((file, index) => 
                renderTableNode(file, 50 + index * 250, 50)
              )}
              
              {/* Render relationships */}
              {relationships.map(rel => renderRelationshipLine(rel))}
            </svg>
          </div>
        </div>

        {/* Relationship Details */}
        {relationships.length > 0 && (
          <div className="space-y-3">
            <h4 className="font-medium flex items-center gap-2">
              <ArrowRight className="h-4 w-4" />
              Detected Relationships ({relationships.length})
            </h4>
            <div className="space-y-2">
              {relationships.map((rel, index) => {
                const fromFile = uploadedFiles.find(f => f.file_id === rel.from_table);
                const toFile = uploadedFiles.find(f => f.file_id === rel.to_table);
                
                return (
                  <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border">
                    <div className="flex items-center gap-3">
                      <div className="flex items-center gap-2">
                        <div 
                          className="w-3 h-3 rounded-full" 
                          style={{ backgroundColor: getTableColor(rel.from_table) }}
                        />
                        <span className="font-mono text-sm">
                          {fromFile?.filename.replace('.csv', '')}.{rel.from_column}
                        </span>
                      </div>
                      <ArrowRight className="h-4 w-4 text-gray-400" />
                      <div className="flex items-center gap-2">
                        <div 
                          className="w-3 h-3 rounded-full" 
                          style={{ backgroundColor: getTableColor(rel.to_table) }}
                        />
                        <span className="font-mono text-sm">
                          {toFile?.filename.replace('.csv', '')}.{rel.to_column}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge 
                        variant={rel.confidence > 0.8 ? "default" : "secondary"}
                        className="text-xs"
                      >
                        {Math.round(rel.confidence * 100)}%
                      </Badge>
                      <Badge variant="outline" className="text-xs">
                        {rel.relationship_type}
                      </Badge>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {loading && (
          <div className="text-center py-8">
            <div className="flex items-center justify-center gap-2 text-blue-600">
              <RefreshCw className="h-5 w-5 animate-spin" />
              <span>Analyzing relationships between tables...</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default EntityRelationshipGraph;