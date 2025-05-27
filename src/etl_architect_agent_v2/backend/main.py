from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from .routes import catalog_routes, conversion_routes, file_routes, transformation_routes
from etl_architect_agent_v2.backend.config import get_settings
from etl_architect_agent_v2.backend.services.transformation_service import TransformationService
import boto3
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ETL Architect Agent API")

# Get settings
settings = get_settings()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize S3 client
s3_client = boto3.client('s3', region_name=settings.AWS_REGION)

def get_transformation_service():
    """Get transformation service instance."""
    return TransformationService(
        bucket=settings.AWS_S3_BUCKET,
        aws_region=settings.AWS_REGION
    )

def initialize_metadata():
    """
    Initialize metadata files if they don't exist
    """
    try:
        logger.info("Starting metadata initialization")
        
        # Initialize catalog index with proper structure
        catalog_index = {
            "tables": {},  # Dictionary of tables, keyed by table name
            "last_updated": datetime.utcnow().isoformat(),
            "metadata": {
                "total_tables": 0,
                "total_files": 0,
                "formats": {},
                "last_scan": None
            }
        }
        
        # Initialize conversion status index
        conversion_index = {
            "conversions": [],
            "last_updated": datetime.utcnow().isoformat(),
            "metadata": {
                "total_conversions": 0,
                "pending": 0,
                "completed": 0,
                "failed": 0
            }
        }
        
        # Initialize audit index
        audit_index = {
            "tables": {},  # Dictionary of table audit trails
            "files": {},   # Dictionary of file audit trails
            "last_updated": datetime.utcnow().isoformat(),
            "metadata": {
                "total_events": 0,
                "last_event": None
            }
        }
        
        # Create metadata directory if it doesn't exist
        try:
            logger.info("Checking metadata directory")
            s3_client.head_object(
                Bucket=settings.AWS_S3_BUCKET,
                Key="metadata/"
            )
            logger.info("Metadata directory exists")
        except s3_client.exceptions.ClientError:
            logger.info("Creating metadata directory")
            s3_client.put_object(
                Bucket=settings.AWS_S3_BUCKET,
                Key="metadata/",
                Body=""
            )
        
        # Store initial metadata
        logger.info("Storing catalog index")
        s3_client.put_object(
            Bucket=settings.AWS_S3_BUCKET,
            Key="metadata/catalog_index.json",
            Body=json.dumps(catalog_index, indent=2)
        )
        
        logger.info("Storing conversion index")
        s3_client.put_object(
            Bucket=settings.AWS_S3_BUCKET,
            Key="metadata/conversion_index.json",
            Body=json.dumps(conversion_index, indent=2)
        )
        
        logger.info("Storing audit index")
        s3_client.put_object(
            Bucket=settings.AWS_S3_BUCKET,
            Key="metadata/audit_index.json",
            Body=json.dumps(audit_index, indent=2)
        )
        
        logger.info("Successfully initialized metadata files")
        
    except Exception as e:
        logger.error(f"Error initializing metadata: {str(e)}")
        raise e


# Initialize metadata on startup
@app.on_event("startup")
async def startup_event():
    try:
        initialize_metadata()
    except Exception as e:
        logger.error(f"Failed to initialize metadata: {str(e)}")


# Include routers
app.include_router(catalog_routes.router, prefix="/api/catalog", tags=["catalog"])
app.include_router(conversion_routes.router, prefix="/api/conversion", tags=["conversion"])
app.include_router(file_routes.router, prefix="/api/files", tags=["files"])
app.include_router(transformation_routes.router, prefix="/api/transformation", tags=["transformation"], dependencies=[Depends(get_transformation_service)])


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    try:
        # Check if metadata files exist and have correct structure
        logger.info("Checking metadata files in health check")
        response = s3_client.get_object(
            Bucket=settings.AWS_S3_BUCKET,
            Key="metadata/catalog_index.json"
        )
        catalog_index = json.loads(response['Body'].read())
        
        # Validate structure
        if "tables" not in catalog_index:
            logger.error("Invalid catalog index structure")
            return {
                "status": "degraded",
                "metadata_initialized": False,
                "error": "Invalid catalog index structure"
            }
            
        return {
            "status": "healthy",
            "metadata_initialized": True,
            "total_tables": len(catalog_index["tables"]),
            "total_files": catalog_index["metadata"]["total_files"]
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "degraded",
            "metadata_initialized": False,
            "error": str(e)
        } 