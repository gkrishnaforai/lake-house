import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from etl_architect_agent_v2.backend.routes.catalog_routes import router as catalog_router
from etl_architect_agent_v2.backend.routes.query_routes import router as query_router
from etl_architect_agent_v2.backend.routes.transformation_routes import router as transformation_router
from etl_architect_agent_v2.backend.routes.report_routes import router as report_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ETL Architect Agent API",
    description="API for ETL Architect Agent",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
logger.info("Registering API routers...")

# Register query router first to ensure it's properly loaded
app.include_router(
    query_router,
    prefix="/api/queries",
    tags=["queries"]
)
logger.info("Registered query router at /api/queries")

app.include_router(
    catalog_router,
    prefix="/api/catalog",
    tags=["catalog"]
)
logger.info("Registered catalog router at /api/catalog")

app.include_router(
    transformation_router,
    prefix="/api/transformations",
    tags=["transformations"]
)
logger.info("Registered transformation router at /api/transformations")

app.include_router(
    report_router,
    prefix="/api/reports",
    tags=["reports"]
)
logger.info("Registered report router at /api/reports")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up ETL Architect Agent API")
    logger.info("Available routes:")
    for route in app.routes:
        logger.info(f"  {route.methods} {route.path}")
        if hasattr(route, "name"):
            logger.info(f"    Name: {route.name}")
        if hasattr(route, "endpoint"):
            logger.info(f"    Endpoint: {route.endpoint.__name__}") 