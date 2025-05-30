from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class UploadResponse(BaseModel):
    status: str
    file_id: str
    table_name: str
    message: str

class ChatRequest(BaseModel):
    message: str
    schema: Dict[str, str]
    sample_data: List[Dict[str, Any]]

class ChatResponse(BaseModel):
    status: str
    response: str
    message: str

class DescriptiveQueryRequest(BaseModel):
    query: str
    table_name: str

class DescriptiveQueryResponse(BaseModel):
    status: str
    results: List[Dict[str, Any]]
    query: str
    message: str

class TableInfo(BaseModel):
    name: str
    description: Optional[str]
    columns: List[Dict[str, Any]]
    s3_location: str
    created_at: Optional[str]
    updated_at: Optional[str]

class DatabaseCatalog(BaseModel):
    tables: List[TableInfo]
    total_tables: int
    last_updated: str

class FileInfo(BaseModel):
    name: str
    size: int
    type: str
    last_modified: str
    s3_path: str

class FileMetadata(BaseModel):
    file_id: str
    table_name: str
    schema: Dict[str, str]
    sample_data: List[Dict[str, Any]]
    created_at: str
    updated_at: str 