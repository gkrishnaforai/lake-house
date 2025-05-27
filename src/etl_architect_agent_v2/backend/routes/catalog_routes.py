from fastapi import (
    APIRouter, HTTPException, Depends, UploadFile, File, Query, Body, Response
)
from typing import Dict, Any, Optional, List
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
        logger.info(f"Fetching catalog data for user {user_id}...")
        catalog_data = await catalog_service.get_catalog(user_id=user_id)
        logger.info(f"Raw catalog data: {json.dumps(catalog_data, indent=2)}")
        
        # Ensure required fields are present
        if not isinstance(catalog_data, dict):
            logger.error(f"Invalid catalog data type: {type(catalog_data)}")
            raise HTTPException(
                status_code=500,
                detail="Invalid catalog data format"
            )
            
        if 'name' not in catalog_data:
            catalog_data['name'] = 'Data Catalog'
        if 'description' not in catalog_data:
            catalog_data['description'] = 'Main data catalog'
        if 'created_at' not in catalog_data:
            catalog_data['created_at'] = datetime.utcnow().isoformat()
        if 'updated_at' not in catalog_data:
            catalog_data['updated_at'] = datetime.utcnow().isoformat()
            
        logger.info(f"Processed catalog data: {json.dumps(catalog_data, indent=2)}")
        return catalog_data
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
        logger.info(f"Fetching tables for user {user_id}...")
        tables = await catalog_service.list_tables(user_id=user_id)
        
        # Ensure each table has required fields
        processed_tables = []
        for table in tables:
            if not isinstance(table, dict):
                logger.error(f"Invalid table data type: {type(table)}")
                continue
                
            # Convert column dictionaries to ColumnInfo objects
            columns = [
                ColumnInfo(
                    name=col.get('Name', ''),
                    type=col.get('Type', ''),
                    description=col.get('Comment', '')
                )
                for col in table.get('StorageDescriptor', {}).get('Columns', [])
            ]
            
            # Create schema dictionary
            schema = {
                "columns": [
                    {
                        "name": col.get('Name', ''),
                        "type": col.get('Type', ''),
                        "description": col.get('Comment', '')
                    }
                    for col in table.get('StorageDescriptor', {}).get('Columns', [])
                ]
            }
            
            # Convert datetime objects to ISO format strings
            created_at = table.get('CreateTime', datetime.utcnow())
            updated_at = table.get('UpdateTime', datetime.utcnow())
            
            if isinstance(created_at, datetime):
                created_at = created_at.isoformat()
            if isinstance(updated_at, datetime):
                updated_at = updated_at.isoformat()
            
            processed_table = TableInfo(
                name=table.get('Name', 'Unknown Table'),
                schema=schema,
                location=table.get('StorageDescriptor', {}).get('Location', ''),
                description=table.get('Description', ''),
                created_at=created_at,
                updated_at=updated_at
            )
            processed_tables.append(processed_table)
            
        # Convert TableInfo objects to dictionaries for logging
        #table_dicts = [table.model_dump() for table in processed_tables]
        # logger.info(f"Processed tables data: {json.dumps(table_dicts, indent=2)}")
        return processed_tables
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
    return await catalog_service.get_table(table_name, user_id=user_id)

@router.get("/tables/{table_name}/schema")
async def get_table_schema(
    table_name: str,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get the schema definition for a specific table."""
    return await catalog_service.get_table_schema(table_name, user_id=user_id)

@router.get("/tables/{table_name}/preview")
async def get_table_preview(
    table_name: str,
    user_id: str = Query("test_user", description="User ID"),
    rows: int = Query(5, description="Number of rows to return"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get a preview of table contents."""
    try:
        logger.info(f"Fetching preview for table {table_name} for user {user_id}...")
        preview = await catalog_service.get_table_preview(table_name, user_id, rows)
        logger.info(f"Successfully fetched preview for table {table_name}")
        return preview
    except Exception as e:
        logger.error(f"Error getting table preview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# File Operations
@router.get("/files", response_model=List[FileMetadata])
async def list_files(
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """List all files in the catalog with their metadata."""
    try:
        logger.info(f"Fetching files for user {user_id}...")
        files = await catalog_service.list_files(user_id=user_id)
        logger.info(f"Raw files data: {json.dumps(files, indent=2)}")
        
        # Ensure each file has required fields
        processed_files = []
        for file in files:
            if not isinstance(file, dict):
                logger.error(f"Invalid file data type: {type(file)}")
                continue
                
            processed_file = {
                'name': file.get('name', 'Unknown File'),
                'size': file.get('size', 0),
                'last_modified': file.get('last_modified', datetime.utcnow().isoformat()),
                'format': file.get('format', 'unknown'),
                'location': file.get('location', ''),
                'schema': file.get('schema', {}),
                'quality': file.get('quality', {})
            }
            processed_files.append(processed_file)
            
        logger.info(f"Processed files data: {json.dumps(processed_files, indent=2)}")
        return processed_files
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files/{s3_path:path}", response_model=FileMetadata)
async def get_file(
    s3_path: str,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get metadata for a specific file."""
    try:
        return await catalog_service.get_file(s3_path, user_id=user_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files/{s3_path:path}/details", response_model=CatalogFileInfo)
async def get_file_details(
    s3_path: str,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get detailed information about a specific file."""
    return await catalog_service.get_file_details(s3_path, user_id=user_id)

@router.get("/files/{s3_path:path}/preview", response_model=FilePreview)
async def get_file_preview(
    s3_path: str,
    rows: int = Query(5, ge=1, le=100),
    columns: Optional[List[str]] = None,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get a preview of the file contents."""
    return await catalog_service.get_file_preview(s3_path, rows, columns, user_id=user_id)

@router.post("/files/{s3_path:path}/convert", response_model=FileConversionResponse)
async def convert_file(
    s3_path: str,
    conversion: FileConversionRequest,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Convert a file to a different format."""
    return await catalog_service.convert_file(s3_path, conversion, user_id=user_id)

@router.delete("/files/{s3_path:path}")
async def delete_file(
    s3_path: str,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Delete a file and its metadata from the catalog."""
    return await catalog_service.delete_file(s3_path, user_id=user_id)

# File Upload Operations
@router.post("/tables/{table_name}/files")
async def upload_file(
    table_name: str,
    file: UploadFile = File(...),
    create_new: bool = False,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Upload a file to a table in the catalog."""
    try:
        logger.info(f"Uploading file {file.filename} to table {table_name} for user {user_id}")
        
        # Validate file extension
        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in ['csv', 'json', 'parquet', 'xlsx', 'xls']:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_ext}. Supported formats: csv, json, parquet, xlsx, xls"
            )
        
        # Upload file
        result = await catalog_service.upload_file(
            file=file,
            table_name=table_name,
            create_new=create_new,
            user_id=user_id
        )
        
        logger.info(f"Successfully uploaded file {file.filename} to table {table_name}")
        return result
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading file: {str(e)}"
        )

@router.post("/tables/{table_name}/process_s3_file", response_model=UploadResponse)
async def trigger_s3_file_processing(
    table_name: str,
    request_data: ProcessS3FileRequest,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Process an existing S3 file: convert to Parquet, catalog, create Glue table."""
    try:
        logger.info(
            f"Received request to process S3 file: {request_data.original_s3_uri} for table {table_name} for user {user_id}"
        )
        result = await catalog_service.process_s3_file(
            original_s3_uri=request_data.original_s3_uri,
            table_name=table_name,
            user_id=user_id
        )
        
        # Check status from service layer
        if result.get("status") == "error":
            error_message = result.get("message", "Unknown error during S3 file processing")
            logger.error(f"Error processing S3 file via API: {error_message}")
            raise HTTPException(
                status_code=500,
                detail=error_message
            )
        
        logger.info(f"Successfully processed S3 file {request_data.original_s3_uri} into table {table_name}")
        return result

    except ValueError as ve:
        logger.error(f"Validation error processing S3 file via API: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error processing S3 file via API: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Server error processing S3 file: {str(e)}"
        )

# Quality and Audit
@router.get("/quality", response_model=QualityMetrics)
async def get_quality_metrics(
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get overall data quality metrics."""
    return await catalog_service.get_quality_metrics(user_id=user_id)

@router.get("/quality/{table_name}", response_model=QualityMetrics)
async def get_table_quality(
    table_name: str,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get quality metrics for a specific table."""
    return await catalog_service.get_table_quality(table_name, user_id=user_id)

@router.get("/audit/{table_name}", response_model=List[AuditLog])
async def get_table_audit(
    table_name: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get audit logs for a specific table."""
    return await catalog_service.get_table_audit(
        table_name,
        start_date,
        end_date,
        user_id=user_id
    )

# Query Operations
@router.post("/query")
async def execute_query(
    request: QueryRequest = Body(...),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Execute a SQL query against the catalog."""
    return await catalog_service.execute_query(request.query, user_id=request.user_id)

@router.get("/schema/{file_name}")
async def get_schema(
    file_name: str,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get schema for a file."""
    try:
        schema_key = f"metadata/schema/{user_id}/{file_name}.json"
        response = catalog_service.s3_client.get_object(
            Bucket=catalog_service.bucket,
            Key=schema_key
        )
        schema = json.loads(response['Body'].read())
        return schema
    except catalog_service.s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Schema not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quality/{file_name}")
async def get_file_quality_metrics(
    file_name: str,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get quality metrics for a file."""
    try:
        quality_key = f"metadata/quality/{user_id}/{file_name}.json"
        response = catalog_service.s3_client.get_object(
            Bucket=catalog_service.bucket,
            Key=quality_key
        )
        metrics = json.loads(response['Body'].read())
        return metrics
    except catalog_service.s3_client.exceptions.NoSuchKey:
        raise HTTPException(
            status_code=404,
            detail="Quality metrics not found"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/s3-list", tags=["catalog"])
async def s3_list(
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """List files in S3 bucket."""
    import boto3
    import os
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
        region_name=os.environ.get("AWS_REGION", "us-east-1")
    )
    bucket = "lambda-code-q"
    prefix = f"originals/{user_id}/"
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    files = [obj['Key'] for obj in resp.get('Contents', []) if obj.get('Key') and not obj['Key'].endswith('/')]
    return files

@router.get("/evolution/{file_name}")
async def get_schema_evolution(
    file_name: str,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get schema evolution history for a file."""
    try:
        evolution_key = f"metadata/evolution/{user_id}/{file_name}.json"
        response = catalog_service.s3_client.get_object(
            Bucket=catalog_service.bucket,
            Key=evolution_key
        )
        evolution = json.loads(response['Body'].read())
        return evolution
    except catalog_service.s3_client.exceptions.NoSuchKey:
        raise HTTPException(
            status_code=404,
            detail="Evolution history not found"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files/list", response_model=List[str])
async def list_s3_files(
    prefix: str = "originals/",
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """List all files in the given S3 prefix."""
    s3 = catalog_service.s3_client
    bucket = catalog_service.bucket
    try:
        paginator = s3.get_paginator('list_objects_v2')
        files = []
        found = False
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}{user_id}/"):
            found = True
            for obj in page.get('Contents', []):
                key = obj.get('Key')
                if key and not key.endswith('/'):
                    files.append(key)
        if not found or not files:
            return []
        return files
    except s3.exceptions.NoSuchKey:
        return []
    except Exception:
        return []

@router.post("/descriptive_query")
async def descriptive_query(
    request: DescriptiveQueryRequest,
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Process a descriptive query using LangChain."""
    try:
        result = await catalog_service.process_descriptive_query(
            query=request.query,
            table_name=request.table_name,
            preserve_column_names=request.preserve_column_names,
            user_id=request.user_id
        )
        return result
    except Exception as e:
        logger.error(f"Error processing descriptive query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing descriptive query: {str(e)}"
        )

@router.get("/tables/{table_name}/files", response_model=List[FileMetadata])
async def get_table_files(
    table_name: str,
    user_id: str = Query("test_user", description="User ID"),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get files associated with a specific table."""
    try:
        logger.info(f"Fetching files for table {table_name} for user {user_id}...")
        files = await catalog_service.get_table_files(table_name, user_id=user_id)
        logger.info(f"Found {len(files)} files for table {table_name}")
        return files
    except Exception as e:
        logger.error(f"Error getting table files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/transformation/tools")
async def get_transformation_tools() -> List[Dict[str, Any]]:
    """Get available transformation tools."""
    return list(TRANSFORMATION_TOOLS.values())

@router.get("/tables/{table_name}/export")
async def export_table_data(
    table_name: str,
    format: str = "csv",
    user_id: str = "test_user",
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Export table data in the specified format."""
    try:
        logger.info(f"Exporting table {table_name} for user {user_id}...")
        
        # Get table data
        result = await catalog_service.get_table_data(table_name, user_id=user_id)
        
        if result.get('status') != 'success':
            raise Exception(result.get('message', 'Failed to get table data'))
            
        # Get schema and data
        schema = result.get('schema', {}).get('schema', [])
        data = result.get('data', [])
        
        if not data:
            raise Exception("No data found in table")
            
        # Extract column names from schema
        column_names = [col.get('name') for col in schema]
        
        # Convert to DataFrame with proper column names
        df = pd.DataFrame(data, columns=column_names)
        
        # Convert to requested format
        if format.lower() == "csv":
            csv_data = df.to_csv(index=False)
            return Response(
                content=csv_data,
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename={table_name}.csv"
                }
            )
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
    except Exception as e:
        logger.error(f"Error exporting table data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error exporting table data: {str(e)}"
        ) 