"""Routes for data transformations."""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from etl_architect_agent_v2.backend.services.transformation_service import (
    TransformationService
)
from etl_architect_agent_v2.backend.config import get_settings

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["transformation"])

# Pydantic models for request/response validation
class TransformationRequest(BaseModel):
    """Request model for applying a transformation."""
    table_name: str
    tool_id: str
    source_columns: List[str]
    user_id: str = "test_user"

class TransformationResponse(BaseModel):
    """Response model for transformation results."""
    status: str
    message: Optional[str] = None
    new_columns: Optional[List[Dict[str, str]]] = None
    preview_data: Optional[List[Dict[str, Any]]] = None

# Dependency to get transformation service instance
def get_transformation_service():
    """Get transformation service instance."""
    settings = get_settings()
    return TransformationService(
        bucket=settings.AWS_S3_BUCKET,
        aws_region=settings.AWS_REGION
    )

@router.get("/tools", response_model=List[Dict[str, Any]])
async def get_available_transformations(
    transformation_service: TransformationService = Depends(get_transformation_service)
):
    """Get list of available transformation tools."""
    try:
        return await transformation_service.get_available_transformations()
    except Exception as e:
        logger.error(f"Error getting available tools: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting available tools: {str(e)}"
        )

@router.get("/tables/{table_name}/columns", response_model=List[Dict[str, str]])
async def get_table_columns(
    table_name: str,
    user_id: str = Query("test_user", description="User ID"),
    transformation_service: TransformationService = Depends(get_transformation_service)
):
    """Get available columns from a table."""
    try:
        return await transformation_service.get_table_columns(table_name, user_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/apply", response_model=TransformationResponse)
async def apply_transformation(
    request: TransformationRequest,
    user_id: str = Query("test_user", description="User ID"),
    transformation_service: TransformationService = Depends(get_transformation_service)
) -> Dict[str, Any]:
    """Apply a transformation to a table."""
    try:
        result = await transformation_service.apply_transformation(
            table_name=request.table_name,
            tool_id=request.tool_id,
            source_columns=request.source_columns,
            user_id=user_id
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error applying transformation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error applying transformation: {str(e)}"
        )

@router.get("/templates")
async def get_transformation_templates(
    user_id: str = Query("test_user", description="User ID"),
    transformation_service: TransformationService = Depends(get_transformation_service)
) -> List[Dict[str, Any]]:
    """Get list of transformation templates for a user."""
    try:
        return await transformation_service.get_transformation_templates(user_id)
    except Exception as e:
        logger.error(f"Error getting templates: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting templates: {str(e)}"
        )

@router.post("/templates")
async def save_transformation_template(
    template: Dict[str, Any],
    user_id: str = Query("test_user", description="User ID"),
    transformation_service: TransformationService = Depends(get_transformation_service)
) -> Dict[str, Any]:
    """Save a transformation template."""
    try:
        return await transformation_service.save_transformation_template(
            template, user_id
        )
    except Exception as e:
        logger.error(f"Error saving template: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error saving template: {str(e)}"
        )

@router.delete("/templates/{template_id}")
async def delete_transformation_template(
    template_id: str,
    user_id: str = Query("test_user", description="User ID"),
    transformation_service: TransformationService = Depends(get_transformation_service)
) -> Dict[str, Any]:
    """Delete a transformation template."""
    try:
        return await transformation_service.delete_transformation_template(
            template_id, user_id
        )
    except Exception as e:
        logger.error(f"Error deleting template: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting template: {str(e)}"
        ) 