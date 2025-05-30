"""File operation routes for the AWS Architect Agent."""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from typing import List
import logging
import os
from datetime import datetime

from src.core.backend.services.catalog_service import CatalogService
from src.core.backend.models.catalog import FileMetadata, UploadResponse

router = APIRouter()
logger = logging.getLogger(__name__)


async def get_catalog_service() -> CatalogService:
    """Get the CatalogService instance."""
    return CatalogService()


@router.get("", response_model=List[FileMetadata])
async def list_files(
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """List all files in the system."""
    try:
        return await catalog_service.list_files()
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Upload a file to the system."""
    try:
        return await catalog_service.upload_file(file)
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{file_id}", response_model=FileMetadata)
async def get_file(
    file_id: str,
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get file metadata by ID."""
    try:
        return await catalog_service.get_file(file_id)
    except Exception as e:
        logger.error(f"Error getting file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{file_id}")
async def delete_file(
    file_id: str,
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Delete a file by ID."""
    try:
        await catalog_service.delete_file(file_id)
        return {"message": "File deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 