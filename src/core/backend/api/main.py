"""FastAPI backend for the AWS Architect Agent."""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import logging
from datetime import datetime
from dotenv import load_dotenv

from src.core.backend.services.catalog_service import CatalogService
from src.core.backend.services.glue_service import GlueService
from src.core.backend.models.catalog import (
    TableInfo, DatabaseCatalog, FileMetadata,
    DescriptiveQueryRequest, DescriptiveQueryResponse
)
from src.core.backend.routes import (
    catalog_routes, conversion_routes, file_routes,
    transformation_routes, healing_routes
)
from src.core.backend.api.catalog_api import router as catalog_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AWS Architect Agent API",
    description="API for AWS data lake and architecture management",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency injection
async def get_catalog_service() -> CatalogService:
    """Get the CatalogService instance."""
    return CatalogService()

async def get_glue_service() -> GlueService:
    """Get the GlueService instance."""
    return GlueService()

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

# Catalog endpoints
@app.get("/api/catalog/tables", response_model=List[TableInfo])
async def list_tables(
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """List all tables in the catalog."""
    try:
        return await catalog_service.list_tables()
    except Exception as e:
        logger.error(f"Error listing tables: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/catalog", response_model=DatabaseCatalog)
async def get_database_catalog(
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """Get the full database catalog."""
    try:
        return await catalog_service.get_database_catalog()
    except Exception as e:
        logger.error(f"Error getting database catalog: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/catalog/files", response_model=List[FileMetadata])
async def list_catalog_files(
    catalog_service: CatalogService = Depends(get_catalog_service)
):
    """List all files in the catalog."""
    try:
        return await catalog_service.list_files()
    except Exception as e:
        logger.error(f"Error listing catalog files: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/catalog/descriptive_query", response_model=DescriptiveQueryResponse)
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

# Include routers
app.include_router(catalog_router, prefix="/api/catalog", tags=["catalog"])
app.include_router(
    conversion_routes.router,
    prefix="/api/conversion",
    tags=["conversion"]
)
app.include_router(
    file_routes.router,
    prefix="/api/files",
    tags=["files"]
)
app.include_router(
    transformation_routes.router,
    prefix="/api/transformation",
    tags=["transformation"]
)
app.include_router(
    healing_routes.router,
    prefix="/api/healing",
    tags=["healing"]
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 