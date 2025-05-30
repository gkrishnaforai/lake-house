"""Data transformation routes for the AWS Architect Agent."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
import logging

from src.core.backend.services.catalog_service import CatalogService

router = APIRouter()
logger = logging.getLogger(__name__)


async def get_catalog_service() -> CatalogService:
    """Get the CatalogService instance."""
    return CatalogService()


@router.post("/{table_name}/transform")
async def transform_table(
    table_name: str,
    transformation_config: Dict[str, Any],
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Apply transformations to a table."""
    try:
        result = await catalog_service.transform_table(
            table_name, transformation_config
        )
        return result
    except Exception as e:
        logger.error(f"Error transforming table: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{table_name}/transformations")
async def list_transformations(
    table_name: str,
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """List all transformations applied to a table."""
    try:
        transformations = await catalog_service.list_transformations(table_name)
        return transformations
    except Exception as e:
        logger.error(f"Error listing transformations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{table_name}/schema")
async def get_transformed_schema(
    table_name: str,
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get the schema of a transformed table."""
    try:
        schema = await catalog_service.get_transformed_schema(table_name)
        return schema
    except Exception as e:
        logger.error(f"Error getting transformed schema: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 