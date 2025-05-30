from fastapi import (
    APIRouter, HTTPException, Depends, UploadFile, File, Query, Body
)
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from ..services.catalog_service import CatalogService
from ..config import get_settings
from ..models.catalog import (
    DatabaseCatalog,
    TableInfo,
    FileMetadata,
    FileInfo as CatalogFileInfo,
    FilePreview,
    FileConversionRequest,
    FileConversionResponse,
    AuditLog,
    QualityMetrics,
    UploadResponse
)
from datetime import datetime
import logging
import pandas as pd
from ..dependencies import get_catalog_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["catalog"])

# Get settings instance
settings = get_settings()

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
    data: Dict[str, Any]

class SchemaEvolutionRequest(BaseModel):
    file_name: str
    new_schema: str

class ProcessS3FileRequest(BaseModel):
    original_s3_uri: str

class DescriptiveQueryRequest(BaseModel):
    query: str
    table_name: Optional[str] = None
    user_id: str = "test_user"
    preserve_column_names: bool = True

class DescriptiveQueryResponse(BaseModel):
    status: str
    query: Optional[str] = None
    results: Optional[list] = None
    message: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    user_id: str = "test_user"

class SchemaValidationRequest(BaseModel):
    schema: Dict[str, Any]
    data: Dict[str, Any]

class QualityCheckConfigRequest(BaseModel):
    """Request model for configuring quality checks."""
    enabled_metrics: List[str] = ["completeness", "uniqueness", "consistency"]
    thresholds: Dict[str, float] = {
        "completeness": 0.95,
        "uniqueness": 0.90,
        "consistency": 0.85
    }
    schedule: Optional[str] = None  # Cron expression for periodic checks

class MetricDetails(BaseModel):
    total: Optional[int] = None
    valid: Optional[int] = None
    invalid: Optional[int] = None
    missing: Optional[int] = None
    threshold: Optional[float] = None

class MetricData(BaseModel):
    score: float
    status: str
    details: Optional[MetricDetails] = None
    trend: Optional[str] = None
    lastUpdated: Optional[str] = None

class QualityMetrics(BaseModel):
    metrics: Optional[Dict[str, MetricData]] = None
    completeness: Optional[float] = None
    accuracy: Optional[float] = None
    consistency: Optional[float] = None
    timeliness: Optional[float] = None
    completeness_details: Optional[MetricDetails] = None
    accuracy_details: Optional[MetricDetails] = None
    consistency_details: Optional[MetricDetails] = None
    timeliness_details: Optional[MetricDetails] = None
    column_metrics: Optional[Dict[str, Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None

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

@router.get("/tables/{table_name}/quality", response_model=QualityMetrics)
async def get_table_quality_metrics(
    table_name: str,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get quality metrics for a specific table."""
    try:
        return await catalog_service.get_table_quality(table_name, user_id=user_id)
    except Exception as e:
        logger.error(f"Error getting table quality metrics: {str(e)}")
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
        # Get metrics for the specific table only
        metrics = await catalog_service.get_table_quality(table_name, user_id=user_id)
        
        # If metrics are in the old format (flat structure), convert to new format
        if not metrics.get("metrics") and any(key in metrics for key in ["completeness", "accuracy", "consistency", "timeliness"]):
            converted_metrics = {
                "metrics": {},
                "metadata": {
                    "table_name": table_name,
                    "checked_at": datetime.now().isoformat()
                }
            }
            
            for metric_name in ["completeness", "accuracy", "consistency", "timeliness"]:
                if metric_name in metrics:
                    converted_metrics["metrics"][metric_name] = {
                        "score": metrics[metric_name],
                        "status": "success" if metrics[metric_name] >= 0.95 else "warning" if metrics[metric_name] >= 0.85 else "error",
                        "details": metrics.get(f"{metric_name}_details")
                    }
            
            # Only preserve table-specific metadata
            for key, value in metrics.items():
                if (key not in ["completeness", "accuracy", "consistency", "timeliness"] 
                    and not key.endswith("_details")
                    and key not in ["file_metrics", "files"]):  # Exclude file-related data
                    converted_metrics[key] = value
            
            return converted_metrics
        
        # For new format, ensure we only return table-specific metrics
        if metrics.get("metrics"):
            # Filter out any file-specific metrics
            table_metrics = {
                "metrics": metrics["metrics"],
                "metadata": {
                    "table_name": table_name,
                    "checked_at": datetime.now().isoformat()
                }
            }
            
            # Preserve other table-specific metadata, excluding file-related data
            for key, value in metrics.items():
                if key not in ["metrics", "file_metrics", "files"]:
                    table_metrics[key] = value
            
            return table_metrics
        
        return metrics
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

@router.post("/validate-schema")
async def validate_schema(
    request: SchemaValidationRequest,
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Validate data against schema constraints."""
    try:
        # Convert data dict to DataFrame
        df = pd.DataFrame(request.data)
        
        # Validate schema
        result = await catalog_service.validate_schema(
            schema=request.schema,
            data=df
        )
        return result
    except Exception as e:
        logger.error(f"Error validating schema: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/check-quality")
async def check_data_quality(
    request: DataQualityRequest,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Check data quality metrics for a dataset."""
    try:
        # Convert data dict to DataFrame
        df = pd.DataFrame(request.data)
        
        # Check data quality
        result = await catalog_service._check_data_quality(
            file_name=request.file_name,
            data=df,
            user_id=user_id
        )
        return result
    except Exception as e:
        logger.error(f"Error checking data quality: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tables/{table_name}/quality/config")
async def configure_quality_checks(
    table_name: str,
    config: QualityCheckConfigRequest,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Configure quality checks for a table."""
    try:
        return await catalog_service.configure_quality_checks(
            table_name=table_name,
            config=config,
            user_id=user_id
        )
    except Exception as e:
        logger.error(f"Error configuring quality checks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tables/{table_name}/quality/run")
async def run_quality_checks(
    table_name: str,
    force: bool = Query(False, description="Force check even if recently run"),
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Run quality checks for a table."""
    try:
        return await catalog_service.run_quality_checks(
            table_name=table_name,
            user_id=user_id,
            force=force
        )
    except Exception as e:
        logger.error(f"Error running quality checks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 