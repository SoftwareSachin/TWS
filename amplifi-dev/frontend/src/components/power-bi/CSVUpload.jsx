"use client";
import React, { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Upload, FileText, AlertCircle } from "lucide-react";
import { showError, showSuccess } from "@/utils/toastUtils";

const CSVUpload = ({ onUploadSuccess, loading, setLoading }) => {
  const [dragActive, setDragActive] = useState(false);
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

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const handleFileChange = (e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const handleFileUpload = async (file) => {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      showError("Please select a CSV file");
      return;
    }

    if (file.size > 10 * 1024 * 1024) { // 10MB limit
      showError("File size must be less than 10MB");
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/v1/csv/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const result = await response.json();
      showSuccess(`Successfully uploaded ${file.name}`);
      onUploadSuccess(result);
    } catch (error) {
      console.error('Upload error:', error);
      showError(error.message || 'Failed to upload CSV file');
    } finally {
      setLoading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const openFileDialog = () => {
    fileInputRef.current?.click();
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FileText className="h-5 w-5" />
          Upload CSV File
        </CardTitle>
      </CardHeader>
      <CardContent>
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
            Drop your CSV file here, or{" "}
            <button
              onClick={openFileDialog}
              className="text-blue-600 hover:text-blue-700 underline"
              disabled={loading}
            >
              browse
            </button>
          </h3>
          <p className="text-sm text-gray-500 mb-4">
            Maximum file size: 10MB
          </p>
          
          <div className="flex items-center justify-center gap-2 text-sm text-gray-600">
            <AlertCircle className="h-4 w-4" />
            <span>Only CSV files are supported</span>
          </div>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          onChange={handleFileChange}
          className="hidden"
          disabled={loading}
        />

        <Button
          onClick={openFileDialog}
          disabled={loading}
          className="w-full mt-4"
        >
          {loading ? "Uploading..." : "Select CSV File"}
        </Button>
      </CardContent>
    </Card>
  );
};

export default CSVUpload;