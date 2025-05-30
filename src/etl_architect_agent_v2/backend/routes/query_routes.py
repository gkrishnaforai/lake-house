"""API routes for query management."""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import List, Optional
import logging

from etl_architect_agent_v2.backend.models.catalog import SavedQuery
from etl_architect_agent_v2.backend.services.query_service import QueryService
from etl_architect_agent_v2.backend.config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router without prefix (prefix is set in main.py)
router = APIRouter(tags=["queries"])

# Dependency to get query service instance
def get_query_service():
    settings = get_settings()
    logger.info(
        f"Initializing QueryService with bucket: {settings.AWS_S3_BUCKET}, "
        f"region: {settings.AWS_REGION}"
    )
    return QueryService(
        bucket=settings.AWS_S3_BUCKET,
        aws_region=settings.AWS_REGION
    )

@router.get("", response_model=List[SavedQuery])
async def list_queries(
    user_id: str = Query("test_user", description="User ID"),
    query_service: QueryService = Depends(get_query_service)
):
    """List all saved queries for a user."""
    logger.info(f"Received request to list queries for user: {user_id}")
    try:
        queries = await query_service.get_all_queries(user_id)
        logger.info(
            f"Successfully retrieved {len(queries)} queries for user: {user_id}"
        )
        return queries
    except Exception as e:
        logger.error(f"Error listing queries: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{query_id}", response_model=SavedQuery)
async def get_query(
    query_id: str,
    user_id: str = Query("test_user", description="User ID"),
    query_service: QueryService = Depends(get_query_service)
):
    """Get a specific saved query."""
    try:
        query = await query_service.get_query(user_id, query_id)
        if not query:
            raise HTTPException(status_code=404, detail="Query not found")
        return query
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("", response_model=SavedQuery)
async def save_query(
    name: str = Body(...),
    query: str = Body(...),
    tables: List[str] = Body(...),
    description: Optional[str] = Body(None),
    query_id: Optional[str] = Body(None),
    user_id: str = Query("test_user", description="User ID"),
    query_service: QueryService = Depends(get_query_service)
):
    """Save a new query or update an existing one."""
    try:
        return await query_service.save_query(
            user_id=user_id,
            name=name,
            query=query,
            tables=tables,
            description=description,
            query_id=query_id
        )
    except Exception as e:
        logger.error(f"Error saving query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{query_id}")
async def delete_query(
    query_id: str,
    user_id: str = Query("test_user", description="User ID"),
    query_service: QueryService = Depends(get_query_service)
):
    """Delete a saved query."""
    try:
        success = await query_service.delete_query(user_id, query_id)
        if not success:
            raise HTTPException(status_code=404, detail="Query not found")
        return {"status": "success", "message": "Query deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{query_id}/favorite")
async def update_query_favorite(
    query_id: str,
    is_favorite: bool = Body(...),
    user_id: str = Query("test_user", description="User ID"),
    query_service: QueryService = Depends(get_query_service)
):
    """Update query favorite status."""
    try:
        query = await query_service.update_query_favorite(
            user_id, query_id, is_favorite
        )
        if not query:
            raise HTTPException(status_code=404, detail="Query not found")
        return query
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating query favorite: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{query_id}/execute")
async def execute_query(
    query_id: str,
    user_id: str = Query("test_user", description="User ID"),
    query_service: QueryService = Depends(get_query_service)
):
    """Execute a saved query and update its execution stats."""
    try:
        query = await query_service.get_query(user_id, query_id)
        if not query:
            raise HTTPException(status_code=404, detail="Query not found")
        
        # Update execution stats
        await query_service.update_query_execution(user_id, query_id)
        
        # Execute the query using the existing query execution endpoint
        return {"status": "success", "message": "Query executed"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 