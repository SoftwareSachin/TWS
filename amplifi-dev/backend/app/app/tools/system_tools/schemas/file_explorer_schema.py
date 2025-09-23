from typing import List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class FileExplorerInput(BaseModel):
    name: Optional[str] = Field(
        None,
        description="Specific filename to locate (for file management operations)",
    )
    operation: Literal[
        "list",
        "search",
        "content_search",
        "get_metadata",
        "get_description",
        "find_datasets_for_file",
        "find_file_by_name",
    ] = Field(
        default="list",
        description="File system operation: list (browse files), search (find by filename pattern), content_search (find files by content similarity using vector search on descriptions), get_metadata (file properties), get_description (basic file overview), find_datasets_for_file (locate file across datasets), find_file_by_name (quickly locate a specific file by exact name)",
    )
    search_pattern: Optional[str] = Field(
        None,
        description="Filename pattern to match (supports wildcards like *.pdf, *report*, etc.)",
    )
    content_query: Optional[str] = Field(
        None,
        description="Natural language query to find files by content similarity (used with content_search operation)",
    )
    # Context fields for dataset operations
    dataset_ids: List[UUID] = Field(
        ..., description="List of Dataset UUIDs to scope the operation"
    )

    # Pagination parameters
    limit: Optional[int] = Field(
        10,
        description="Maximum number of files to return (default: 50, max: 200)",
        ge=1,
        le=200,
    )
    page: Optional[int] = Field(
        1, description="Page number for pagination (starts from 1)", ge=1
    )

    @model_validator(mode="after")
    def validate_input_context(self):
        """Validate input parameters and auto-detect operations."""
        # Catch common invalid operation names
        invalid_operations = {
            "list_files": "list",
            "search_files": "search",
            "find_file": "find_file_by_name",
            "get_file_metadata": "get_metadata",
            "get_file_description": "get_description",
        }

        if self.operation in invalid_operations:
            correct_operation = invalid_operations[self.operation]
            raise ValueError(
                f"Invalid operation '{self.operation}'. Did you mean '{correct_operation}'? "
                f"Valid operations: list, search, content_search, get_metadata, get_description, find_datasets_for_file, find_file_by_name"
            )

        # Auto-detect search operation when search_pattern is provided
        if self.search_pattern and self.operation == "list":
            self.operation = "search"

        # Auto-detect content_search operation when content_query is provided
        if self.content_query and self.operation == "list":
            self.operation = "content_search"

        # Content search operation requires a content query
        if self.operation == "content_search":
            if not self.content_query:
                raise ValueError(
                    "'content_query' field is required for content_search operation"
                )

        # File-specific operations require a filename
        if self.operation in [
            "get_metadata",
            "get_description",
            "find_datasets_for_file",
            "find_file_by_name",
        ]:
            if not self.name:
                raise ValueError(
                    f"'name' field is required for {self.operation} operation"
                )

        return self


class FileInfo(BaseModel):
    # Core file information
    name: str = Field(..., description="File name and extension")
    size: Optional[str] = Field(
        None, description="Human-readable file size (e.g., '1.2 MB')"
    )
    file_extension: Optional[str] = Field(
        None, description="File extension (e.g., '.pdf', '.csv')"
    )

    # Database-specific fields
    file_id: Optional[str] = Field(None, description="Internal file identifier")
    content_summary: Optional[str] = Field(
        None, description="Brief summary of file content"
    )
    mimetype: Optional[str] = Field(None, description="MIME type of the file")

    # Structured data metadata - Currently not implemented for structured data
    # estimated_rows: Optional[int] = Field(
    #     None, description="Number of rows (for tabular data)"
    # )
    # estimated_columns: Optional[int] = Field(
    #     None, description="Number of columns (for tabular data)"
    # )

    # Context information
    dataset_names: Optional[List[str]] = Field(
        None, description="Datasets containing this file"
    )


class FileExplorerOutput(BaseModel):
    current_path: str = Field(..., description="Current browsing context")
    operation_result: str = Field(
        ..., description="Summary of the file system operation performed"
    )
    items: List[FileInfo] = Field(..., description="List of files found")
    total_items: int = Field(..., description="Total number of files")
    total_size: Optional[str] = Field(None, description="Combined size of all files")
    search_results: Optional[List[str]] = Field(
        None, description="Search-specific results and matching details"
    )
    dataset_name: Optional[str] = Field(None, description="Primary dataset context")

    # Comprehensive context information
    datasets_involved: Optional[List[str]] = Field(
        None, description="All datasets involved in the operation"
    )
    operation_type: Optional[str] = Field(
        None, description="Type of operation performed (list, search, metadata, etc.)"
    )
    fuzzy_matching_used: Optional[bool] = Field(
        False, description="Whether fuzzy matching was used in the search"
    )

    # File type statistics
    file_types_summary: Optional[dict] = Field(
        None, description="Summary of file types found (extension counts)"
    )

    # Enhanced contextual information
    context_hints: Optional[List[str]] = Field(
        None, description="Helpful hints and context for the user"
    )

    # Pagination information
    pagination: Optional[dict] = Field(None, description="Pagination metadata")
    has_more: bool = Field(
        False, description="Whether there are more results available"
    )
    next_page: Optional[int] = Field(
        None, description="Next page number if has_more is True"
    )
