"""File format conversion routes for the AWS Architect Agent."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging

from src.core.backend.services.catalog_service import CatalogService
from src.core.backend.models.catalog import FileMetadata

router = APIRouter()
logger = logging.getLogger(__name__)


async def get_catalog_service() -> CatalogService:
    """Get the CatalogService instance."""
    return CatalogService()


@router.post("/{file_id}/convert")
async def convert_file(
    file_id: str,
    target_format: str,
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Convert a file to a different format."""
    try:
        result = await catalog_service.convert_file(file_id, target_format)
        return result
    except Exception as e:
        logger.error(f"Error converting file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{file_id}/status")
async def get_conversion_status(
    file_id: str,
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get the status of a file conversion."""
    try:
        status = await catalog_service.get_conversion_status(file_id)
        return status
    except Exception as e:
        logger.error(f"Error getting conversion status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 