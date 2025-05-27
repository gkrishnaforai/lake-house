from fastapi import (
    APIRouter, HTTPException, Depends, UploadFile, File, Query, Body, Response
)
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel
from etl_architect_agent_v2.backend.services.catalog_service import (
    CatalogService
)
from etl_architect_agent_v2.backend.models.catalog import (
    DatabaseCatalog,
    TableInfo,
    FileMetadata,
    FileInfo as CatalogFileInfo,
    FilePreview,
    FileConversionRequest,
    FileConversionResponse,
    AuditLog,
    QualityMetrics,
    UploadResponse,
    ColumnInfo  # Import ColumnInfo from catalog models
)
from etl_architect_agent_v2.backend.config import get_settings
import pandas as pd
import json
from datetime import datetime
import logging
from ..config.transformation_tools import TRANSFORMATION_TOOLS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["catalog"])

# Pydantic models for request/response validation
class FileInfo(BaseModel):
    file_name: str
    s3_path: str
    format: Optional[str] = None

class SchemaUpdate(BaseModel):
    file_name: str
    schema: str
    version: Optional[str] = None

class DataQualityRequest(BaseModel):
    file_name: str
    data: Dict[str, Any]  # JSON representation of DataFrame

class SchemaEvolutionRequest(BaseModel):
    file_name: str
    new_schema: str

class ProcessS3FileRequest(BaseModel):
    original_s3_uri: str

class DescriptiveQueryRequest(BaseModel):
    query: str
    table_name: Optional[str] = None
    user_id: str = "test_user"  # Add user_id with default
    preserve_column_names: bool = True

class DescriptiveQueryResponse(BaseModel):
    status: str
    query: Optional[str] = None
    results: Optional[list] = None
    message: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    user_id: str = "test_user"

# Add new Pydantic models for reports
class Report(BaseModel):
    id: str
    name: str
    description: str
    query: str
    schedule: Optional[str] = None
    last_run: Optional[str] = None
    created_by: str
    created_at: str
    is_favorite: bool = False

class ReportCreate(BaseModel):
    name: str
    description: str
    query: str
    schedule: Optional[str] = None

class ReportUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    query: Optional[str] = None
    schedule: Optional[str] = None
    is_favorite: Optional[bool] = None

class SQLQueryRequest(BaseModel):
    query: str
    tables: List[str]
    user_id: str = "test_user"

class SQLQueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    error: Optional[str] = None

# Dependency to get catalog service instance
def get_catalog_service():
    settings = get_settings()
    return CatalogService(
        bucket=settings.AWS_S3_BUCKET,
        aws_region=settings.AWS_REGION
    )

# Catalog Overview
@router.get("", response_model=DatabaseCatalog)
async def get_catalog(
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get the complete database catalog overview."""
    try:
        return await catalog_service.get_catalog(user_id=user_id)
    except Exception as e:
        logger.error(f"Error getting catalog: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Table Operations
@router.get("/tables", response_model=List[TableInfo])
async def list_tables(
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """List all tables in the catalog."""
    try:
        return await catalog_service.list_tables(user_id=user_id)
    except Exception as e:
        logger.error(f"Error listing tables: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tables/{table_name}", response_model=TableInfo)
async def get_table(
    table_name: str,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get detailed information about a specific table."""
    try:
        return await catalog_service.get_table(table_name, user_id=user_id)
    except Exception as e:
        logger.error(f"Error getting table: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tables/{table_name}/schema")
async def get_table_schema(
    table_name: str,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get the schema definition for a specific table."""
    try:
        return await catalog_service.get_table_schema(table_name, user_id=user_id)
    except Exception as e:
        logger.error(f"Error getting table schema: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tables/{table_name}/preview")
async def get_table_preview(
    table_name: str,
    user_id: str = Query("test_user", description="User ID"),
    rows: int = Query(5, description="Number of rows to return"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get a preview of the table data."""
    try:
        return await catalog_service.get_table_preview(table_name, user_id=user_id, rows=rows)
    except Exception as e:
        logger.error(f"Error getting table preview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# File Operations
@router.get("/files", response_model=List[FileMetadata])
async def list_files(
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """List all files in the catalog."""
    try:
        return await catalog_service.list_files(user_id=user_id)
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files/{s3_path:path}", response_model=FileMetadata)
async def get_file(
    s3_path: str,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get file metadata."""
    try:
        return await catalog_service.get_file(s3_path, user_id=user_id)
    except Exception as e:
        logger.error(f"Error getting file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/files/{s3_path:path}")
async def delete_file(
    s3_path: str,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Delete a file."""
    try:
        await catalog_service.delete_file(s3_path, user_id=user_id)
        return {"status": "success", "message": "File deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files/{s3_path:path}/details", response_model=CatalogFileInfo)
async def get_file_details(
    s3_path: str,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get detailed file information."""
    try:
        return await catalog_service.get_file(s3_path, user_id=user_id)
    except Exception as e:
        logger.error(f"Error getting file details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files/{s3_path:path}/preview", response_model=FilePreview)
async def get_file_preview(
    s3_path: str,
    rows: int = Query(5, ge=1, le=100),
    columns: Optional[List[str]] = None,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get a preview of the file contents."""
    try:
        return await catalog_service.get_file_preview(s3_path, user_id=user_id, rows=rows)
    except Exception as e:
        logger.error(f"Error getting file preview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/files/{s3_path:path}/convert", response_model=FileConversionResponse)
async def convert_file(
    s3_path: str,
    conversion: FileConversionRequest,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Convert a file to a different format."""
    try:
        return await catalog_service.convert_file(s3_path, conversion, user_id=user_id)
    except Exception as e:
        logger.error(f"Error converting file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Table File Operations
@router.post("/tables/{table_name}/files")
async def upload_file(
    table_name: str,
    file: UploadFile = File(...),
    create_new: bool = False,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Upload a file to a table."""
    try:
        return await catalog_service.upload_file(file, table_name, create_new, user_id=user_id)
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tables/{table_name}/files", response_model=List[FileMetadata])
async def get_table_files(
    table_name: str,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get all files associated with a table."""
    try:
        return await catalog_service.get_table_files(table_name, user_id=user_id)
    except Exception as e:
        logger.error(f"Error getting table files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tables/{table_name}/process_s3_file", response_model=UploadResponse)
async def trigger_s3_file_processing(
    table_name: str,
    request_data: ProcessS3FileRequest,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Process an S3 file for a table."""
    try:
        return await catalog_service.process_s3_file(request_data.original_s3_uri, table_name, user_id=user_id)
    except Exception as e:
        logger.error(f"Error processing S3 file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Quality and Audit
@router.get("/quality", response_model=QualityMetrics)
async def get_quality_metrics(
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get overall quality metrics."""
    try:
        return await catalog_service.get_quality_metrics(user_id=user_id)
    except Exception as e:
        logger.error(f"Error getting quality metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quality/{table_name}", response_model=QualityMetrics)
async def get_table_quality(
    table_name: str,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get quality metrics for a specific table."""
    try:
        return await catalog_service.get_table_quality(table_name, user_id=user_id)
    except Exception as e:
        logger.error(f"Error getting table quality: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/audit/{table_name}", response_model=List[AuditLog])
async def get_table_audit(
    table_name: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get audit logs for a table."""
    try:
        return await catalog_service.get_table_audit(table_name, start_date, end_date, user_id=user_id)
    except Exception as e:
        logger.error(f"Error getting table audit: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Query Operations
@router.post("/query")
async def execute_query(
    request: QueryRequest = Body(...),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Execute a SQL query."""
    try:
        return await catalog_service.execute_query(request.query, request.user_id)
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/descriptive_query")
async def descriptive_query(
    request: DescriptiveQueryRequest,
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Execute a descriptive query."""
    try:
        return await catalog_service.process_descriptive_query(
            request.query,
            request.table_name,
            request.preserve_column_names,
            request.user_id
        )
    except Exception as e:
        logger.error(f"Error executing descriptive query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Schema and Evolution
@router.get("/schema/{file_name}")
async def get_schema(
    file_name: str,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get schema for a file."""
    try:
        return await catalog_service.get_schema(file_name, user_id=user_id)
    except Exception as e:
        logger.error(f"Error getting schema: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/evolution/{file_name}")
async def get_schema_evolution(
    file_name: str,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get schema evolution history for a file."""
    try:
        return await catalog_service.get_schema_evolution(file_name, user_id=user_id)
    except Exception as e:
        logger.error(f"Error getting schema evolution: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# S3 Operations
@router.get("/s3-list")
async def s3_list(
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """List S3 objects."""
    try:
        return await catalog_service.list_s3_files(user_id=user_id)
    except Exception as e:
        logger.error(f"Error listing S3 files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files/list", response_model=List[str])
async def list_s3_files(
    prefix: str = "originals/",
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """List files in S3."""
    try:
        return await catalog_service.list_s3_files(prefix, user_id=user_id)
    except Exception as e:
        logger.error(f"Error listing S3 files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Transformation Tools
@router.get("/transformation/tools")
async def get_transformation_tools() -> List[Dict[str, Any]]:
    """Get available transformation tools."""
    return TRANSFORMATION_TOOLS

# Add SQL query endpoint
@router.post("/query/sql", response_model=SQLQueryResponse)
async def execute_sql_query(
    request: SQLQueryRequest = Body(...),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Execute a SQL query."""
    try:
        results = await catalog_service.execute_query(
            request.query,
            request.tables,
            request.user_id
        )
        return SQLQueryResponse(results=results)
    except Exception as e:
        logger.error(f"Error executing SQL query: {str(e)}")
        return SQLQueryResponse(results=[], error=str(e))

# Add report endpoints
@router.post("/reports", response_model=Report)
async def create_report(
    report: ReportCreate = Body(...),
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Create a new report."""
    try:
        new_report = Report(
            id=str(datetime.now().timestamp()),
            name=report.name,
            description=report.description,
            query=report.query,
            schedule=report.schedule,
            created_by=user_id,
            created_at=datetime.now().isoformat(),
            is_favorite=False
        )
        # Here you would typically save the report to a database
        return new_report
    except Exception as e:
        logger.error(f"Error creating report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reports", response_model=List[Report])
async def list_reports(
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """List all reports for a user."""
    try:
        # Here you would typically fetch reports from a database
        return []
    except Exception as e:
        logger.error(f"Error listing reports: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reports/{report_id}", response_model=Report)
async def get_report(
    report_id: str,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get a specific report."""
    try:
        # Here you would typically fetch the report from a database
        raise HTTPException(status_code=404, detail="Report not found")
    except Exception as e:
        logger.error(f"Error getting report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/reports/{report_id}", response_model=Report)
async def update_report(
    report_id: str,
    report_update: ReportUpdate = Body(...),
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Update a report."""
    try:
        # Here you would typically update the report in a database
        raise HTTPException(status_code=404, detail="Report not found")
    except Exception as e:
        logger.error(f"Error updating report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/reports/{report_id}")
async def delete_report(
    report_id: str,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Delete a report."""
    try:
        # Here you would typically delete the report from a database
        return {"message": "Report deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reports/schedule")
async def schedule_report(
    report_id: str = Body(...),
    schedule: str = Body(...),
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Schedule a report."""
    try:
        # Here you would typically schedule the report in a database
        return {"message": "Report scheduled successfully"}
    except Exception as e:
        logger.error(f"Error scheduling report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reports/share")
async def share_report(
    report_id: str = Body(...),
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Share a report."""
    try:
        # Here you would typically handle report sharing
        return {"message": "Report shared successfully"}
    except Exception as e:
        logger.error(f"Error sharing report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 