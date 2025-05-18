from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from datetime import datetime


class DatabaseCatalog(BaseModel):
    name: str
    description: str
    created_at: datetime
    updated_at: datetime


class TableInfo(BaseModel):
    name: str
    schema: Dict[str, Any]
    location: str
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class FileMetadata(BaseModel):
    name: str
    size: int
    last_modified: datetime
    format: str
    location: str
    schema: Optional[Dict[str, Any]] = None
    quality: Optional[Dict[str, Any]] = None


class FileInfo(BaseModel):
    file_name: str
    s3_path: str
    format: Optional[str] = None


class FilePreview(BaseModel):
    columns: List[str]
    data: List[Dict[str, Any]]
    total_rows: int


class FileConversionRequest(BaseModel):
    target_format: str
    options: Optional[Dict[str, Any]] = None


class FileConversionResponse(BaseModel):
    status: str
    message: str
    new_location: Optional[str] = None


class UploadResponse(BaseModel):
    status: str
    message: str
    file_info: Dict[str, Any]


class UploadProgress(BaseModel):
    file_name: str
    bytes_uploaded: int
    total_bytes: int
    status: str
    error: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    schema: Optional[Dict[str, Any]] = None
    sample_data: Optional[List[Dict[str, Any]]] = None
    sql: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    sql: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class ClassificationRequest(BaseModel):
    """Request model for classification."""
    client_id: str
    data_path: str
    user_instruction: str


class DescriptiveQueryRequest(BaseModel):
    query: str
    table_name: Optional[str] = None


class DescriptiveQueryResponse(BaseModel):
    status: str
    query: Optional[str] = None
    results: Optional[list] = None
    message: Optional[str] = None


class QualityMetrics(BaseModel):
    completeness: float
    accuracy: float
    consistency: float
    timeliness: float


class AuditLog(BaseModel):
    timestamp: datetime
    action: str
    details: Dict[str, Any]
    user: Optional[str] = None


class ColumnInfo(BaseModel):
    name: str
    type: str
    description: Optional[str] = None
    sample_values: Optional[List[Any]] = None 