"""Pydantic models for the catalog service."""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime


class ColumnInfo(BaseModel):
    """Model for column information."""
    name: str
    type: str
    description: Optional[str] = None
    sample_values: Optional[List[Any]] = None


class TableInfo(BaseModel):
    """Model for table information."""
    name: str
    description: Optional[str] = None
    columns: List[ColumnInfo]
    row_count: Optional[int] = None
    last_updated: Optional[str] = None
    s3_location: Optional[str] = None


class DatabaseCatalog(BaseModel):
    """Model for database catalog."""
    database_name: str
    tables: List[TableInfo]
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class FileMetadata(BaseModel):
    """Model for file metadata."""
    file_name: str
    file_type: str
    s3_path: str
    schema_path: str
    created_at: str
    size: int


class FileInfo(BaseModel):
    """Model for file information."""
    file_name: str
    s3_path: str
    size: int
    file_type: str
    last_modified: datetime


class UploadResponse(BaseModel):
    """Response model for file upload."""
    status: str
    message: str
    file_name: str
    s3_path: str
    table_name: Optional[str] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class DescriptiveQueryRequest(BaseModel):
    """Request model for descriptive query."""
    query: str
    table_name: Optional[str] = None


class DescriptiveQueryResponse(BaseModel):
    """Response model for descriptive query."""
    status: str
    query: Optional[str] = None
    results: Optional[List[Any]] = None
    message: Optional[str] = None 