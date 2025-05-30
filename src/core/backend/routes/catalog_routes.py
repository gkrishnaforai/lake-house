"""Catalog routes for the AWS Architect Agent."""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from src.core.backend.services.catalog_service import CatalogService
from src.core.backend.models.catalog import (
    TableInfo, DatabaseCatalog, FileMetadata, FileInfo,
    UploadResponse, DescriptiveQueryRequest, DescriptiveQueryResponse
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Dependency injection
async def get_catalog_service() -> CatalogService:
    """Get the CatalogService instance."""
    return CatalogService()

@router.get("/tables", response_model=List[TableInfo])
async def list_tables(
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """List all tables in the catalog."""
    try:
        return await catalog_service.list_tables()
    except Exception as e:
        logger.error(f"Error listing tables: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("", response_model=DatabaseCatalog)
async def get_database_catalog(
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get the full database catalog."""
    try:
        return await catalog_service.get_database_catalog()
    except Exception as e:
        logger.error(f"Error getting database catalog: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files", response_model=List[FileMetadata])
async def list_catalog_files(
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """List all files in the catalog."""
    try:
        return await catalog_service.list_files()
    except Exception as e:
        logger.error(f"Error listing catalog files: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/descriptive_query", response_model=DescriptiveQueryResponse)
async def descriptive_query(
    request: DescriptiveQueryRequest,
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Process a descriptive query."""
    try:
        return await catalog_service.process_descriptive_query(
            query=request.query,
            table_name=request.table_name
        )
    except Exception as e:
        logger.error(f"Error processing descriptive query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tables/{table_name}/files")
async def upload_table_file(
    table_name: str,
    file: UploadFile = File(...),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """
    Upload a file to a specific table in the catalog.
    """
    try:
        return await catalog_service.upload_file_to_table(table_name, file)
    except Exception as e:
        logger.error(f"Error uploading file to table {table_name}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 