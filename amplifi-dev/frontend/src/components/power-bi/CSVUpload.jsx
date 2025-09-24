"use client";
import React, { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Upload, FileText, AlertCircle, X, Database, Table } from "lucide-react";
import { showError, showSuccess } from "@/utils/toastUtils";

const CSVUpload = ({ onUploadSuccess, loading, setLoading }) => {
  const [dragActive, setDragActive] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const files = Array.from(e.dataTransfer.files);
    if (files && files.length > 0) {
      handleMultipleFileUpload(files);
    }
  };

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files);
    if (files && files.length > 0) {
      handleMultipleFileUpload(files);
    }
  };

  const handleMultipleFileUpload = async (files) => {
    // Validate all files first
    const invalidFiles = files.filter(file => !file.name.toLowerCase().endsWith('.csv'));
    if (invalidFiles.length > 0) {
      showError(`Invalid file types: ${invalidFiles.map(f => f.name).join(', ')}. Only CSV files are allowed.`);
      return;
    }

    const oversizedFiles = files.filter(file => file.size > 10 * 1024 * 1024);
    if (oversizedFiles.length > 0) {
      showError(`Files too large: ${oversizedFiles.map(f => f.name).join(', ')}. Maximum size is 10MB per file.`);
      return;
    }

    setLoading(true);
    const uploadResults = [];
    const errors = [];

    try {
      // Upload files one by one to maintain order and handle errors
      for (const file of files) {
        try {
          const formData = new FormData();
          formData.append('file', file);

          const response = await fetch('/api/v1/csv/upload-multiple', {
            method: 'POST',
            body: formData,
          });

          if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Upload failed');
          }

          const result = await response.json();
          uploadResults.push(result);
          showSuccess(`Successfully uploaded ${file.name}`);
        } catch (error) {
          console.error(`Upload error for ${file.name}:`, error);
          errors.push({ filename: file.name, error: error.message });
        }
      }

      if (uploadResults.length > 0) {
        setUploadedFiles(uploadResults);
        onUploadSuccess({ files: uploadResults, errors });
      }

      if (errors.length > 0) {
        showError(`Failed to upload ${errors.length} file(s): ${errors.map(e => e.filename).join(', ')}`);
      }
    } finally {
      setLoading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const removeFile = (fileId) => {
    const updatedFiles = uploadedFiles.filter(file => file.file_id !== fileId);
    setUploadedFiles(updatedFiles);
    onUploadSuccess({ files: updatedFiles, errors: [] });
  };

  const openFileDialog = () => {
    fileInputRef.current?.click();
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Database className="h-5 w-5" />
          Upload CSV Files
        </CardTitle>
        <p className="text-sm text-gray-600 mt-2">
          Upload multiple CSV files to create comprehensive data analysis and entity relationship diagrams
        </p>
      </CardHeader>
      <CardContent className="space-y-6">
        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            dragActive
              ? "border-blue-500 bg-blue-50"
              : "border-gray-300 hover:border-gray-400"
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <Upload className="h-12 w-12 mx-auto mb-4 text-gray-400" />
          <h3 className="text-lg font-medium mb-2">
            Drop your CSV files here, or{" "}
            <button
              onClick={openFileDialog}
              className="text-blue-600 hover:text-blue-700 underline font-medium"
              disabled={loading}
            >
              browse
            </button>
          </h3>
          <p className="text-sm text-gray-500 mb-4">
            Upload multiple CSV files to automatically detect relationships
          </p>
          
          <div className="flex flex-col items-center gap-2 text-sm text-gray-600">
            <div className="flex items-center gap-2">
              <AlertCircle className="h-4 w-4" />
              <span>Maximum file size: 10MB per file</span>
            </div>
            <div className="flex items-center gap-2">
              <FileText className="h-4 w-4" />
              <span>Supports: CSV format only</span>
            </div>
          </div>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          multiple
          onChange={handleFileChange}
          className="hidden"
          disabled={loading}
        />

        <Button
          onClick={openFileDialog}
          disabled={loading}
          className="w-full"
          size="lg"
        >
          {loading ? (
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              Uploading Files...
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <Upload className="h-4 w-4" />
              Select CSV Files
            </div>
          )}
        </Button>

        {uploadedFiles.length > 0 && (
          <div className="space-y-4">
            <h4 className="font-medium text-gray-900 flex items-center gap-2">
              <Table className="h-4 w-4" />
              Uploaded Files ({uploadedFiles.length})
            </h4>
            <div className="grid gap-3">
              {uploadedFiles.map((file) => (
                <div 
                  key={file.file_id} 
                  className="flex items-center justify-between p-4 bg-gray-50 rounded-lg border"
                >
                  <div className="flex items-center gap-3">
                    <FileText className="h-5 w-5 text-green-600" />
                    <div>
                      <div className="font-medium text-sm text-gray-900">
                        {file.filename}
                      </div>
                      <div className="flex items-center gap-4 text-xs text-gray-500">
                        <span>{file.row_count} rows</span>
                        <span>{file.columns.length} columns</span>
                        <span>{(file.file_size / 1024).toFixed(1)} KB</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="text-xs">
                      {file.columns.length} cols
                    </Badge>
                    <button
                      onClick={() => removeFile(file.file_id)}
                      className="p-1 hover:bg-gray-200 rounded"
                      title="Remove file"
                    >
                      <X className="h-4 w-4 text-gray-500" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default CSVUpload;