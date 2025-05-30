"""FastAPI backend for the data lakehouse builder UI."""

from fastapi import (
    FastAPI, UploadFile, File, WebSocket, BackgroundTasks, HTTPException,
    Depends, Body, Header, Query
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import tempfile
import os
import logging
from datetime import datetime
import boto3
import json
import uuid
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from botocore.exceptions import (
    ClientError, NoCredentialsError, PartialCredentialsError, BotoCoreError
)
import asyncio
import re
from io import BytesIO

from etl_architect_agent_v2.agents.classification.classification_agent import (
    ClassificationAgent, ClassificationState
)
from etl_architect_agent_v2.core.llm.manager import LLMManager
from etl_architect_agent_v2.agents.catalog_agent import (
    CatalogAgent, CatalogState
)
from etl_architect_agent_v2.backend.services.catalog_service import (
    CatalogService
)
from etl_architect_agent_v2.backend.routes import catalog_routes
from etl_architect_agent_v2.backend.services.glue_service import GlueService
from etl_architect_agent_v2.backend.services.athena_service import AthenaService
from etl_architect_agent_v2.backend.services.agent_service import AgentService
from etl_architect_agent_v2.backend.models.catalog import (
    ColumnInfo, TableInfo, DatabaseCatalog, FileMetadata, FileInfo,
    FileConversionRequest, UploadProgress, UploadResponse, ChatRequest,
    ChatResponse, ClassificationRequest, DescriptiveQueryRequest,
    DescriptiveQueryResponse
)
from src.core.agents.tools.log_manager import LogManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Global placeholders for clients and services, to be initialized in lifespan
s3_client = None
glue_client = None
catalog_service_instance = None
catalog_agent = None
llm_manager = None
classification_agent = None

# Global placeholders for AWS config values that will be set in lifespan
BUCKET_NAME = None
DATABASE_NAME = None
aws_region = None

# Define a temporary file path for lifespan logging during tests
LIFESPAN_DEBUG_LOG_FILE = "lifespan_debug.log"  # Relative to workspace root

logger.info(
    "AWS configuration will be initialized during application lifespan "
    "startup."
)

# --- Lifespan Manager (defined BEFORE app instantiation) ---
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    # Attempt to open/create the debug log file immediately
    try:
        with open(LIFESPAN_DEBUG_LOG_FILE, "a") as f_debug_initial_check:
            f_debug_initial_check.write(
                f"{datetime.utcnow().isoformat()} - "
                "LIFESPAN FUNCTION ENTRY ATTEMPT (CONSTRUCTOR MODE)\n"
            )
    except Exception as e_initial_log:
        # This print might not be visible in pytest if lifespan fails very early
        print(
            f"CRITICAL_LIFESPAN_LOG_FAIL: Could not open "
            f"{LIFESPAN_DEBUG_LOG_FILE}: {str(e_initial_log)}"
        )

    global s3_client, glue_client, catalog_service_instance
    global catalog_agent, llm_manager, classification_agent
    global BUCKET_NAME, DATABASE_NAME, aws_region

    # Open the debug log file in append mode
    with open(LIFESPAN_DEBUG_LOG_FILE, "a") as f_debug:
        f_debug.write(
            f"{datetime.utcnow().isoformat()} - "
            "LIFESPAN STARTING (CONSTRUCTOR MODE, after globals)\n"
        )
        logger.info("Application lifespan startup...")
        load_dotenv()

        aws_access_key_local = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key_local = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region_local = os.getenv('AWS_REGION', 'us-east-1')
        aws_session_token_local = os.getenv('AWS_SESSION_TOKEN')
        BUCKET_NAME_local = os.getenv('AWS_S3_BUCKET', 'your-bucket-name')
        DATABASE_NAME_local = os.getenv('GLUE_DATABASE_NAME', 'data_lakehouse')

        aws_region = aws_region_local
        BUCKET_NAME = BUCKET_NAME_local
        DATABASE_NAME = DATABASE_NAME_local

        f_debug.write(
            f"{datetime.utcnow().isoformat()} - "
            f"Read AWS Region: {aws_region}\n"
        )
        logger.info(f"Read S3 Bucket: {BUCKET_NAME}")
        f_debug.write(
            f"{datetime.utcnow().isoformat()} - "
            f"Read S3 Bucket: {BUCKET_NAME}\n"
        )
        logger.info(f"Read Glue Database: {DATABASE_NAME}")
        f_debug.write(
            f"{datetime.utcnow().isoformat()} - "
            f"Read Glue Database: {DATABASE_NAME}\n"
        )

        # Format key ID for logging (first 4 and last 4 chars)
        key_id_to_log = (
            f"{aws_access_key_local[:4]}...{aws_access_key_local[-4:]}"
            if aws_access_key_local and len(aws_access_key_local) > 8
            else "Not Set or Too Short"
        )
        logger.info(
            f"AWS Access Key ID: {key_id_to_log}"
        )
        f_debug.write(
            f"{datetime.utcnow().isoformat()} - "
            f"AWS Access Key ID: {key_id_to_log}\n"
        )
        logger.info(
            f"AWS Secret Key: {'Set (masked)' if aws_secret_key_local else 'Not Set'}"
        )
        f_debug.write(
            f"{datetime.utcnow().isoformat()} - "
            f"AWS Secret Key: {'Set (masked)' if aws_secret_key_local else 'Not Set'}\n"
        )
        logger.info(
            f"AWS Session Token: {'Set' if aws_session_token_local else 'Not Set'}"
        )
        f_debug.write(
            f"{datetime.utcnow().isoformat()} - "
            f"AWS Session Token: {'Set' if aws_session_token_local else 'Not Set'}\n"
        )

        try:
            f_debug.write(
                f"{datetime.utcnow().isoformat()} - "
                "LIFESPAN_STAGE: Attempting to initialize S3 client...\n"
            )
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_local,
                aws_secret_access_key=aws_secret_key_local,
                aws_session_token=aws_session_token_local,
                region_name=aws_region
            )
            f_debug.write(
                f"{datetime.utcnow().isoformat()} - "
                "LIFESPAN_STAGE: S3 client initialized successfully.\n"
            )

            f_debug.write(
                f"{datetime.utcnow().isoformat()} - "
                "LIFESPAN_STAGE: Attempting to initialize Glue client...\n"
            )
            glue_client = boto3.client(
                'glue',
                aws_access_key_id=aws_access_key_local,
                aws_secret_access_key=aws_secret_key_local,
                aws_session_token=aws_session_token_local,
                region_name=aws_region
            )
            f_debug.write(
                f"{datetime.utcnow().isoformat()} - "
                "LIFESPAN_STAGE: Glue client initialized successfully.\n"
            )

            if not validate_aws_config_internal():
                f_debug.write(
                    f"{datetime.utcnow().isoformat()} - "
                    "LIFESPAN_VALIDATION_FAILURE: AWS configuration validation "
                    "failed.\n"
                )
            else:
                f_debug.write(
                    f"{datetime.utcnow().isoformat()} - "
                    "Lifespan: AWS configuration validated successfully.\n"
                )

            f_debug.write(
                f"{datetime.utcnow().isoformat()} - "
                "LIFESPAN_STAGE: Attempting to initialize CatalogService...\n"
            )
            catalog_service_instance = CatalogService(
                bucket=BUCKET_NAME,
                aws_region=aws_region
            )
            f_debug.write(
                f"{datetime.utcnow().isoformat()} - "
                "LIFESPAN_STAGE: CatalogService initialized successfully.\n"
            )

            f_debug.write(
                f"{datetime.utcnow().isoformat()} - "
                "LIFESPAN_STAGE: Attempting to initialize CatalogAgent...\n"
            )
            catalog_agent = CatalogAgent(
                catalog_service=catalog_service_instance
            )
            f_debug.write(
                f"{datetime.utcnow().isoformat()} - "
                "LIFESPAN_STAGE: CatalogAgent initialized successfully.\n"
            )

            f_debug.write(
                f"{datetime.utcnow().isoformat()} - "
                "LIFESPAN_STAGE: Attempting to initialize LLMManager...\n"
            )
            llm_manager = LLMManager()
            f_debug.write(
                f"{datetime.utcnow().isoformat()} - "
                "LIFESPAN_STAGE: LLMManager initialized successfully.\n"
            )

            f_debug.write(
                f"{datetime.utcnow().isoformat()} - "
                "LIFESPAN_STAGE: Attempting to initialize ClassificationAgent...\n"
            )
            classification_agent = ClassificationAgent()
            f_debug.write(
                f"{datetime.utcnow().isoformat()} - "
                "LIFESPAN_STAGE: ClassificationAgent initialized successfully.\n"
            )

            f_debug.write(f"{datetime.utcnow().isoformat()} - LIFESPAN YIELDING (CONSTRUCTOR MODE)\n")
        except Exception as e:
            with open(LIFESPAN_DEBUG_LOG_FILE, "a") as f_debug:
                f_debug.write(
                    f"{datetime.utcnow().isoformat()} - "
                    f"LIFESPAN_ERROR: {str(e)}\n"
                )
            raise

        with open(LIFESPAN_DEBUG_LOG_FILE, "a") as f_debug:
            f_debug.write(
                f"{datetime.utcnow().isoformat()} - "
                "LIFESPAN_STAGE: All services initialized successfully.\n"
            )
        yield
        with open(LIFESPAN_DEBUG_LOG_FILE, "a") as f_debug:
            f_debug.write(
                f"{datetime.utcnow().isoformat()} - "
                "LIFESPAN_STAGE: Shutting down services...\n"
            )
            # Cleanup code here if needed
            f_debug.write(
                f"{datetime.utcnow().isoformat()} - "
                "LIFESPAN_STAGE: Services shut down successfully.\n"
            )

# Initialize FastAPI app
app = FastAPI(
    title="Data Lakehouse Builder API",
    description="API for building and managing data lakehouses",
    version="1.0.0",
    lifespan=lifespan
)

# Include catalog routes
app.include_router(catalog_routes.router, prefix="/api/catalog")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Dependency Injection Helpers (defined AFTER app and lifespan) ---
async def get_catalog_service_dependency() -> CatalogService:
    if catalog_service_instance is None:
        logger.error("CatalogService not initialized via lifespan.")
        raise HTTPException(status_code=503, detail="CatalogService is not available.")
    return catalog_service_instance

async def get_catalog_agent_dependency() -> CatalogAgent:
    if catalog_agent is None:
        logger.error("CatalogAgent not initialized via lifespan.")
        raise HTTPException(status_code=503, detail="CatalogAgent is not available.")
    return catalog_agent

def get_s3_client_dependency(): # Changed to a dependency pattern for consistency if ever needed
    if s3_client is None:
        logger.error("S3 client not initialized via lifespan.")
        raise HTTPException(status_code=503, detail="S3 client not available.")
    return s3_client

def get_glue_client_dependency(): # Changed to a dependency pattern
    if glue_client is None:
        logger.error("Glue client not initialized via lifespan.")
        raise HTTPException(status_code=503, detail="Glue client not available.")
    return glue_client

def get_llm_manager_dependency() -> LLMManager:
    if llm_manager is None:
        logger.error("LLMManager not initialized via lifespan.")
        raise HTTPException(status_code=503, detail="LLM service not available.")
    return llm_manager

async def get_classification_agent_dependency() -> ClassificationAgent:
    if classification_agent is None:
        logger.error("ClassificationAgent not initialized via lifespan.")
        raise HTTPException(status_code=503, detail="ClassificationAgent is not available.")
    return classification_agent

async def get_glue_service_dependency() -> GlueService:
    """Get the GlueService instance."""
    if glue_client is None:
        logger.error("Glue client not initialized via lifespan.")
        raise HTTPException(status_code=503, detail="Glue client is not available.")
    return GlueService(glue_client=glue_client)

# --- Internal Helper Functions (using global clients set by lifespan) ---
def ensure_glue_database_internal():
    global glue_client, DATABASE_NAME # Uses globals set by lifespan
    if glue_client is None:
        logger.error("ensure_glue_database_internal: Glue client not initialized.")
        raise RuntimeError("Glue client not available for database check.")
    if DATABASE_NAME is None: # Check if DATABASE_NAME was set
        logger.error("ensure_glue_database_internal: DATABASE_NAME not set.")
        raise RuntimeError("DATABASE_NAME not configured.")
    try:
        glue_client.get_database(Name=DATABASE_NAME)
        logger.info(f"Database {DATABASE_NAME} already exists.")
    except glue_client.exceptions.EntityNotFoundException:
        logger.info(f"Database {DATABASE_NAME} not found, creating it...")
        try:
            glue_client.create_database(
                DatabaseInput={'Name': DATABASE_NAME, 'Description': 'Data lakehouse database'}
            )
            logger.info(f"Created database {DATABASE_NAME}.")
        except Exception as e:
            logger.error(f"Error creating database {DATABASE_NAME}: {str(e)}", exc_info=True)
            raise
    except Exception as e:
        logger.error(f"Error checking/creating database {DATABASE_NAME}: {str(e)}", exc_info=True)
        raise

def validate_aws_config_internal():
    global s3_client # Uses global s3_client set by lifespan
    if s3_client is None:
        logger.error("validate_aws_config_internal: S3 client not initialized.")
        return False
    try:
        # Check if essential env vars for credentials were loaded by lifespan
        # No need to re-os.getenv here if we trust lifespan's loading.
        # boto3 client would have failed to init if fundamental creds were missing.
        # A light check on the client itself:
        logger.info("Validating S3 connection (List Buckets)...")
        s3_client.list_buckets() 
        logger.info("S3 connection validated.")

        logger.info("Validating Glue connection and ensuring database exists...")
        ensure_glue_database_internal() # This uses global glue_client
        logger.info("Glue connection and database presence validated.")
        
        return True
    except (ClientError, NoCredentialsError, PartialCredentialsError, BotoCoreError) as cred_err:
        logger.error(f"AWS credential/client error during validation: {str(cred_err)}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"AWS configuration validation failed: {str(e)}", exc_info=True)
        return False

# --- Pydantic Models ---
class FileMetadata(BaseModel):
    file_name: str
    file_type: str
    s3_path: str
    schema_path: str
    created_at: str
    size: int


class ClassificationRequest(BaseModel):
    """Request model for classification."""
    client_id: str
    data_path: str
    user_instruction: str


class TableInfo(BaseModel):
    name: str
    description: Optional[str] = None
    columns: List[ColumnInfo]
    row_count: Optional[int] = None
    last_updated: Optional[str] = None
    s3_location: Optional[str] = None


class DatabaseCatalog(BaseModel):
    database_name: str
    tables: List[TableInfo]
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class FileInfo(BaseModel):
    file_name: str
    s3_path: str
    size: int
    file_type: str
    last_modified: datetime # Keep as datetime for Pydantic


class FileConversionRequest(BaseModel):
    """Request model for file conversion."""
    target_format: str
    user_id: str


class UploadProgress(BaseModel):
    """Model for tracking upload progress."""
    status: str
    progress: float
    message: str
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class UploadResponse(BaseModel):
    """Response model for file upload."""
    status: str
    message: str
    file_name: str
    s3_path: str # This is usually the PARQUET path after successful processing by new endpoint
    table_name: Optional[str] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class DescriptiveQueryRequest(BaseModel):
    query: str
    table_name: Optional[str] = None


class DescriptiveQueryResponse(BaseModel):
    status: str
    query: Optional[str] = None
    results: Optional[list] = None
    message: Optional[str] = None


# --- API Endpoints ---

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test S3 access
        s3_client.list_buckets()
        
        # Test Glue access
        glue_client.get_databases()
        
        return {
            "status": "healthy",
            "aws_region": aws_region,
            "s3_bucket": BUCKET_NAME,
            "glue_database": DATABASE_NAME
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "degraded",
            "error": str(e)
        }

def create_glue_table(
    table_name: str,
    database_name: str,
    location: str,
    input_format: str = "org.apache.hadoop.mapred.TextInputFormat",
    output_format: str = "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
    ser_de_info: dict = None,
    columns: list = None
) -> dict:
    """Create a Glue table with the specified parameters."""
    if ser_de_info is None:
        ser_de_info = {
            "SerializationLibrary": "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe",
            "Parameters": {"field.delim": ","}
        }
    
    if columns is None:
        columns = [
            {"Name": "col1", "Type": "string"},
            {"Name": "col2", "Type": "string"}
        ]
    
    try:
        response = glue_client.create_table(
            DatabaseName=database_name,
            TableInput={
                "Name": table_name,
                "StorageDescriptor": {
                    "Location": location,
                    "InputFormat": input_format,
                    "OutputFormat": output_format,
                    "SerdeInfo": ser_de_info,
                    "Columns": columns
                }
            }
        )
        return response
    except Exception as e:
        logger.error(f"Error creating Glue table: {str(e)}")
        raise

def upload_to_s3( # Old helper, ensure it uses global client
    file_path: str, file_name: str, schema: Dict[str, Any],
    s3: boto3.client = Depends(get_s3_client_dependency) # Or rely on global
) -> Dict[str, str]:
    """Upload a file to S3 and return metadata. Uses global s3_client, BUCKET_NAME."""
    global s3_client, BUCKET_NAME # Fallback to globals
    current_s3_client = s3
    if not current_s3_client: current_s3_client = get_s3_client_dependency()

    if not BUCKET_NAME:
        logger.error("upload_to_s3: BUCKET_NAME not configured.")
        raise RuntimeError("S3 BUCKET_NAME not configured for upload.")
    try:
        upload_id = str(uuid.uuid4())
        raw_s3_key = f"raw/{upload_id}/{file_name}" # S3 key, not full path
        schema_s3_key = f"schemas/{upload_id}/schema.json" # S3 key

        logger.info(f"Uploading raw file {file_name} to s3://{BUCKET_NAME}/{raw_s3_key}")
        current_s3_client.upload_file(file_path, BUCKET_NAME, raw_s3_key)
        
        schema_data = json.dumps(schema, default=str)
        logger.info(f"Uploading schema for {file_name} to s3://{BUCKET_NAME}/{schema_s3_key}")
        current_s3_client.put_object(Bucket=BUCKET_NAME, Key=schema_s3_key, Body=schema_data)
        
        # The create_glue_table expects S3 prefix for location.
        # The raw_s3_key is for a specific file. This needs care.
        # Glue table location should be like f"s3://{BUCKET_NAME}/raw/{upload_id}/"
        glue_table_s3_location = f"s3://{BUCKET_NAME}/raw/{upload_id}/"
        table_name_glue = f"table_{upload_id.replace('-', '_')}" # Make table name Glue-friendly

        # This create_glue_table is for CSV usually. Parquet is different.
        # The schema for create_glue_table needs to match the file format (e.g. CSV schema for CSV file)
        # create_glue_table(table_name_glue, schema, glue_table_s3_location) # Original call. Be careful with schema and S3 path.

        return {
            "raw_path": f"s3://{BUCKET_NAME}/{raw_s3_key}",
            "schema_path": f"s3://{BUCKET_NAME}/{schema_s3_key}",
            "table_name": table_name_glue # Name of the Glue table created (if any)
        }
    except Exception as e:
        logger.error(f"Error in upload_to_s3 for {file_name}: {str(e)}", exc_info=True)
        raise

@app.get("/api/files", response_model=List[FileInfo])
async def list_files(
    s3: boto3.client = Depends(get_s3_client_dependency),
    prefix: str = Query("", description="S3 prefix to filter files (e.g. originals/)", alias="prefix"),
    user_id: str = Header("test_user", alias="X-User-Id")
):
    """
    List all raw files in the S3 bucket for the current user. By default, only list files under originals/{user_id}/.
    """
    global BUCKET_NAME # BUCKET_NAME is set by lifespan
    if not BUCKET_NAME:
        raise HTTPException(status_code=503, detail="S3 bucket not configured.")

    listed_files = []
    # Always list only the current user's raw files unless a different prefix is provided
    if not prefix or prefix == "originals/":
        prefixes_to_scan = [f"originals/{user_id}/"]
    else:
        prefixes_to_scan = [prefix]

    logger.info(f"Listing files from S3 bucket '{BUCKET_NAME}' across prefixes: {prefixes_to_scan}")

    for scan_prefix in prefixes_to_scan:
        try:
            paginator = s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=scan_prefix):
                for obj in page.get('Contents', []):
                    key = obj.get('Key', '')
                    if not key or key.endswith('/'):
                        logger.info(f"Skipping S3 object with empty or directory key: '{key}'")
                        continue
                    listed_files.append(FileInfo(
                        file_name=os.path.basename(key),
                        s3_path=key,
                        size=obj['Size'],
                        file_type=os.path.splitext(key)[1][1:] or "unknown",
                        last_modified=obj['LastModified']
                    ))
        except Exception as e:
            logger.error(
                f"Error listing files under prefix '{scan_prefix}' for user_id '{user_id}' in bucket '{BUCKET_NAME}': {str(e)}",
                exc_info=True
            )
            logger.error(f"Exception type: {type(e).__name__}, args: {e.args}")
            # Continue to next prefix if one fails

    if not listed_files:
        logger.info("No files found in specified S3 prefixes.")
    return listed_files

@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    user_id: str = Depends(get_user_id)
):
    """Upload a file to S3 and create corresponding Glue table."""
    try:
        # Read file content
        content = await file.read()
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Upload to S3
        s3_key = f"raw/{user_id}/{file_id}/{file.filename}"
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=content
        )
        
        # Read file content for schema extraction
        df = pd.read_excel(BytesIO(content)) if file.filename.endswith('.xlsx') else pd.read_csv(BytesIO(content))
        
        # Create Glue table
        table_name = f"{user_id}_{file_id}"
        glue_service.create_table(
            database_name=DATABASE_NAME,
            table_name=table_name,
            s3_location=f"s3://{BUCKET_NAME}/{s3_key}",
            columns=df.dtypes.to_dict()
        )
        
        return UploadResponse(
            status="success",
            file_id=file_id,
            table_name=table_name,
            message="File uploaded and table created successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_agent(
    request: ChatRequest,
    user_id: str = Depends(get_user_id)
):
    """Chat with the ETL architect agent."""
    try:
        response = await agent_service.process_chat(
            user_id=user_id,
            message=request.message,
            schema=request.schema,
            sample_data=request.sample_data
        )
        
        return ChatResponse(
            status="success",
            response=response,
            message="Chat processed successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/catalog/descriptive_query", response_model=DescriptiveQueryResponse)
async def execute_descriptive_query(
    request: DescriptiveQueryRequest,
    user_id: str = Depends(get_user_id)
):
    """Execute a descriptive query using Athena."""
    try:
        # Generate and execute Athena query
        query = athena_service.generate_query(
            natural_language_query=request.query,
            table_name=request.table_name
        )
        
        results = await athena_service.execute_query(
            query=query,
            workgroup=os.getenv('ATHENA_WORKGROUP', 'etl_architect_workgroup')
        )
        
        return DescriptiveQueryResponse(
            status="success",
            results=results,
            query=query,
            message="Query executed successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/catalog/tables", response_model=List[TableInfo])
async def list_tables( # RESTORED ORIGINAL NAME
    glue: boto3.client = Depends(get_glue_client_dependency), # Use DI
    llm: LLMManager = Depends(get_llm_manager_dependency) # Use DI
):
    global DATABASE_NAME # From lifespan
    if not DATABASE_NAME:
        raise HTTPException(status_code=503, detail="Glue database not configured.")
    try:
        response = glue.get_tables(DatabaseName=DATABASE_NAME)
        tables_info_list = []
        for table_data in response.get('TableList', []):
            table_name = table_data['Name']
            # get_table_sample_data might need to be async if it uses await asyncio.sleep
            # For now, assuming it's synchronous or adapted.
            # If get_table_sample_data becomes async, this loop needs `await`
            # sample_data = await get_table_sample_data(table_name) # This can be slow
            # To avoid slow down, maybe get sample data optionally or limit its usage.
            # For now, let's pass empty sample data to analyze_table_schema for speed.
            sample_data_for_analysis = [] # Pass empty or fetch if truly needed by LLM.
            
            # analyze_table_schema is async and takes llm via DI
            table_info_analyzed = await analyze_table_schema(
                table_name,
                table_data.get('StorageDescriptor', {}).get('Columns', []),
                sample_data_for_analysis, # Pass limited/empty sample data
                llm=llm # Pass injected LLM manager
            )
            table_info_analyzed.row_count = table_data.get('Parameters', {}).get('numRows')
            table_info_analyzed.last_updated = table_data.get('UpdateTime', datetime.utcnow()).isoformat()
            tables_info_list.append(table_info_analyzed)
        return tables_info_list
    except Exception as e:
        logger.error(f"Error listing tables from Glue: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/catalog/files", response_model=List[FileMetadata])
async def list_catalog_files(s3: boto3.client = Depends(get_s3_client_dependency)): # RESTORED ORIGINAL NAME
    global BUCKET_NAME # From lifespan
    if not BUCKET_NAME:
        raise HTTPException(status_code=503, detail="S3 bucket not configured.")
    
    # This endpoint logic was very simple, fetching from a single index file.
    # This might be different from /api/files which lists from S3 directly.
    # Keeping original logic for this specific path.
    index_key = 'metadata/catalog_index.json'
    logger.info(f"Fetching catalog index file: s3://{BUCKET_NAME}/{index_key}")
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=index_key)
        index_content = json.loads(response['Body'].read().decode('utf-8'))
        # Assuming index_content is like {"files": [FileMetadata_dict, ...]}
        # Pydantic will validate if the structure matches List[FileMetadata]
        return index_content.get('files', [])
    except s3.exceptions.NoSuchKey:
        logger.warning(f"Catalog index file not found: s3://{BUCKET_NAME}/{index_key}")
        return []
    except Exception as e:
        logger.error(f"Error reading catalog index file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching catalog index: {e}")


@app.get("/api/catalog", response_model=DatabaseCatalog)
async def get_database_catalog( # RESTORED ORIGINAL NAME
    glue: boto3.client = Depends(get_glue_client_dependency), # Use DI
    llm: LLMManager = Depends(get_llm_manager_dependency) # Use DI
):
    global DATABASE_NAME, BUCKET_NAME # From lifespan
    if not DATABASE_NAME or not BUCKET_NAME: # BUCKET_NAME used for total_size
        raise HTTPException(status_code=503, detail="Glue DB or S3 Bucket not configured.")

    try:
        response = glue.get_tables(DatabaseName=DATABASE_NAME)
        tables_list = []
        total_size_bytes = 0

        for table_data in response.get('TableList', []):
            table_name = table_data['Name']
            # sample_data = await get_table_sample_data(table_name) # Potentially slow
            sample_data_for_analysis = []
            
            table_info_obj = await analyze_table_schema(
                table_name,
                table_data.get('StorageDescriptor', {}).get('Columns', []),
                sample_data_for_analysis,
                llm=llm # Pass injected LLM
            )
            table_info_obj.row_count = table_data.get('Parameters', {}).get('numRows')
            table_info_obj.last_updated = table_data.get('UpdateTime', datetime.utcnow()).isoformat()
            tables_list.append(table_info_obj)
            
            # Attempt to get size from parameters if available (might not be accurate or present)
            total_size_bytes += int(table_data.get('Parameters', {}).get('size', 0)) # 'size' is not standard Glue param

        db_desc_prompt = f"Database: {DATABASE_NAME}, Tables: {[t.name for t in tables_list]}. Describe this database."
        db_description_text = "LLM-generated DB description placeholder."
        try:
            db_description_text = await llm.generate_response(db_desc_prompt)
        except Exception as e:
             logger.error(f"LLM error generating DB description: {e}")


        return DatabaseCatalog(
            database_name=DATABASE_NAME,
            tables=tables_list,
            description=db_description_text,
            metadata={
                "total_tables": len(tables_list),
                # "total_files" is ambiguous here, might be same as tables or S3 files.
                # "total_size_mb": round(total_size_bytes / (1024 * 1024), 2) # This size is unreliable
            }
        )
    except Exception as e:
        logger.error(f"Error getting full database catalog: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/files/{s3_path:path}/details")
async def get_file_details(s3_path: str, s3: boto3.client = Depends(get_s3_client_dependency)): # RESTORED
    global BUCKET_NAME # From lifespan
    if not BUCKET_NAME:
        raise HTTPException(status_code=503, detail="S3 bucket not configured.")
    try:
        # s3_path here is the KEY to the object, not full s3:// path
        logger.info(f"Getting details for S3 object: s3://{BUCKET_NAME}/{s3_path}")
        response = s3.head_object(Bucket=BUCKET_NAME, Key=s3_path)
        
        file_schema = None # Schema might not always be associated this way
        # Example: if s3_path refers to a raw file, schema might be linked via naming convention
        if s3_path.startswith('raw/'):
            # This is a simplified assumption for schema path
            # upload_id_part = s3_path.split('/')[1] # e.g., raw/some_id/file.csv -> some_id
            # assumed_schema_key = f"schemas/{upload_id_part}/schema.json"
            # This logic is too specific and might be incorrect for generic file paths.
            # The new upload endpoint stores schema based on table_name: metadata/schema/{table_name}.json
            # This endpoint should probably just return S3 metadata. Schema linkage is complex.
            pass # For now, don't try to guess schema path.

        return {
            "file_name": os.path.basename(s3_path),
            "s3_key": s3_path,
            "size": response["ContentLength"],
            "last_modified": response["LastModified"].isoformat(),
            "content_type": response.get("ContentType", "application/octet-stream"),
            "e_tag": response.get("ETag"),
            "metadata": response.get("Metadata", {}), # Custom S3 metadata
            "schema": file_schema # Will be None unless logic above is implemented
        }
    except s3.exceptions.NoSuchKey:
        logger.error(f"File not found in S3 for details: s3://{BUCKET_NAME}/{s3_path}")
        raise HTTPException(status_code=404, detail=f"File not found at s3://{BUCKET_NAME}/{s3_path}")
    except Exception as e:
        logger.error(f"Error getting file details for s3://{BUCKET_NAME}/{s3_path}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/files/{s3_path:path}/preview")
async def get_file_preview(s3_path: str, limit: int = 10, s3: boto3.client = Depends(get_s3_client_dependency)): # RESTORED
    global BUCKET_NAME # From lifespan
    if not BUCKET_NAME:
        raise HTTPException(status_code=503, detail="S3 bucket not configured.")

    # s3_path is key. Only allow preview for certain prefixes if desired.
    # Example: if not s3_path.startswith(('raw/', 'data/')):
    # raise HTTPException(status_code=400, detail="Preview only for raw/ or data/ files.")
    
    temp_file_local_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file_obj:
            temp_file_local_path = temp_file_obj.name
        
        logger.info(f"Downloading s3://{BUCKET_NAME}/{s3_path} to {temp_file_local_path} for preview.")
        s3.download_file(BUCKET_NAME, s3_path, temp_file_local_path)
            
        file_ext = os.path.splitext(s3_path)[1].lower()
        df_preview = None
        if file_ext == '.xlsx':
            df_preview = pd.read_excel(temp_file_local_path, nrows=limit)
        elif file_ext == '.csv':
            df_preview = pd.read_csv(temp_file_local_path, nrows=limit)
        elif file_ext == '.parquet':
             # pyarrow might be needed: df.to_parquet(engine='pyarrow') / pd.read_parquet(engine='pyarrow')
            df_preview = pd.read_parquet(temp_file_local_path) # Parquet doesn't have nrows in read
            if len(df_preview) > limit: df_preview = df_preview.head(limit)

        else:
            raise HTTPException(status_code=400, detail=f"Preview unsupported for file type: {file_ext}")

        return df_preview.to_dict(orient='records')
    except s3.exceptions.NoSuchKey:
        logger.error(f"File not found for preview: s3://{BUCKET_NAME}/{s3_path}")
        raise HTTPException(status_code=404, detail="File not found for preview.")
    except Exception as e:
        logger.error(f"Error getting file preview for s3://{BUCKET_NAME}/{s3_path}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_local_path and os.path.exists(temp_file_local_path):
            os.unlink(temp_file_local_path)


@app.delete("/api/files/{s3_path:path}")
async def delete_file(s3_path: str, # RESTORED ORIGINAL NAME
                      s3: boto3.client = Depends(get_s3_client_dependency),
                      glue: boto3.client = Depends(get_glue_client_dependency)):
    global BUCKET_NAME, DATABASE_NAME # From lifespan
    if not BUCKET_NAME or not DATABASE_NAME: # DATABASE_NAME needed for Glue table deletion
        raise HTTPException(status_code=503, detail="S3/Glue not configured for delete.")

    # s3_path is key.
    # Add more safety checks if needed, e.g., prefix checks.
    # if not s3_path.startswith(('raw/', 'data/', 'metadata/', 'originals/')):
    #     raise HTTPException(status_code=400, detail="Invalid file path for deletion.")

    logger.info(f"Attempting to delete S3 object: s3://{BUCKET_NAME}/{s3_path}")
    try:
        s3.delete_object(Bucket=BUCKET_NAME, Key=s3_path)
        logger.info(f"S3 object s3://{BUCKET_NAME}/{s3_path} deleted.")
    except Exception as e:
        logger.error(f"Error deleting S3 object s3://{BUCKET_NAME}/{s3_path}: {str(e)}", exc_info=True)
        # Decide if this is fatal or if we continue to try Glue cleanup
        raise HTTPException(status_code=500, detail=f"Failed to delete S3 file: {e}")

    # If deleting a schema file, try to delete associated Glue table
    # metadata/schema/{table_name}.json
    if s3_path.startswith("metadata/schema/") and s3_path.endswith(".json"):
        table_name_from_schema_path = s3_path.split("/")[-1][:-5] # Remove .json
        logger.info(f"Schema file deleted, attempting to delete associated Glue table: {table_name_from_schema_path} in DB {DATABASE_NAME}")
        try:
            glue.delete_table(DatabaseName=DATABASE_NAME, Name=table_name_from_schema_path)
            logger.info(f"Glue table {table_name_from_schema_path} deleted from {DATABASE_NAME}.")
        except glue.exceptions.EntityNotFoundException:
            logger.warning(f"Glue table {table_name_from_schema_path} not found in {DATABASE_NAME}, no deletion needed.")
        except Exception as e:
            logger.warning(f"Error deleting Glue table {table_name_from_schema_path}: {str(e)}")
            # Non-fatal warning, S3 file was deleted.

    # If deleting a data file from a common data prefix, associated schema/table might exist
    # e.g., if s3_path is data/{table_name}/some.parquet
    # This logic is more complex and depends on conventions. For now, only direct schema link is handled.

    return {"message": f"File/resources related to {s3_path} processed for deletion."}


@app.post("/api/files/{s3_path:path}/convert")
async def convert_file( # RESTORED ORIGINAL NAME
    s3_path: str,
    request: FileConversionRequest,
    background_tasks: BackgroundTasks, # Fine as is
    s3: boto3.client = Depends(get_s3_client_dependency), # Use DI
    agent: CatalogAgent = Depends(get_catalog_agent_dependency) # Use DI
):
    global BUCKET_NAME # From lifespan
    logger.warning(f"Deprecated /api/files/{{s3_path}}/convert endpoint called for {s3_path}. This may not be fully functional.")
    if not BUCKET_NAME:
        raise HTTPException(status_code=503, detail="S3 bucket not configured.")
    
    # This endpoint's original logic was to get metadata, then call agent.convert_file
    # Getting metadata for a generic s3_path to construct CatalogState is non-trivial.
    # The CatalogAgent.convert_file likely assumes a well-formed CatalogState.
    # This endpoint is highly likely to be broken or incomplete.
    # For now, returning an error indicating it's not supported/deprecated.
    raise HTTPException(status_code=501, detail="This file conversion endpoint is deprecated/non-functional. Use specific upload process that handles conversion.")

@app.get("/api/catalog/audit/{table_name}")
async def get_audit_trail(table_name: str, s3: boto3.client = Depends(get_s3_client_dependency)): # RESTORED
    global BUCKET_NAME # From lifespan
    if not BUCKET_NAME:
        raise HTTPException(status_code=503, detail="S3 bucket not configured.")
        
    # Original logic assumed metadata at 'metadata/{table_name}.json'
    # This is a very specific convention.
    # Actual audit trails might be stored differently or need a dedicated service.
    assumed_metadata_key = f"metadata/{table_name}.json" # Example, might not be robust
    logger.info(f"Attempting to fetch audit metadata from s3://{BUCKET_NAME}/{assumed_metadata_key}")
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=assumed_metadata_key)
        metadata = json.loads(response['Body'].read().decode('utf-8'))
        # Extract audit-specific parts if they exist by convention
        audit_info = metadata.get("audit_metadata", {}) # e.g. a sub-dictionary
        conversion_hist = audit_info.get("conversion_history", [])
        return {
            "table_name": table_name,
            "audit_metadata_found": audit_info, # The content of "audit_metadata" key
            "conversion_history": conversion_hist # Specific part of it
        }
    except s3.exceptions.NoSuchKey:
        logger.warning(f"No audit metadata file found at {assumed_metadata_key} for table {table_name}")
        raise HTTPException(status_code=404, detail=f"Audit metadata not found for table {table_name}")
    except Exception as e:
        logger.error(f"Error fetching audit trail for {table_name}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/catalog/quality/{table_name}")
async def get_data_quality(
    table_name: str,
    user_id: str = Query("test_user"),
    s3: boto3.client = Depends(get_s3_client_dependency),
    catalog_service: CatalogService = Depends(get_catalog_service_dependency)
):
    """Get quality metrics for a table. If metrics don't exist, they will be calculated."""
    try:
        # First try to get stored quality metrics
        quality_metadata_key = f"metadata/quality/{user_id}/{table_name}/metrics.json"
        try:
            response = s3.get_object(
                Bucket=os.getenv('S3_BUCKET_NAME'),
                Key=quality_metadata_key
            )
            quality_metrics = json.loads(response['Body'].read())
            return quality_metrics
        except s3.exceptions.NoSuchKey:
            # If metrics don't exist, calculate them
            logger.info(f"No stored quality metrics found for {table_name}, calculating...")
            
            # Get table data and calculate metrics
            result = await catalog_service.run_quality_checks(
                table_name=table_name,
                user_id=user_id,
                force=True  # Force recalculation
            )
            
            if result["status"] == "success":
                return result
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to calculate quality metrics: {result.get('message', 'Unknown error')}"
                )
                
    except Exception as e:
        logger.error(f"Error getting quality metrics for {table_name}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting quality metrics: {str(e)}"
        )


# This is the NEW, preferred endpoint for uploads.
@app.post("/catalog/tables/{table_name}/files")
async def upload_file(
    table_name: str,
    file: UploadFile = File(...),
    create_new: bool = False,
    user_id: str = Header("test_user", alias="X-User-Id"),
    catalog_service: CatalogService = Depends(get_catalog_service_dependency)
):
    """
    Upload a file to a table in the catalog. Stores original under originals/{user_id}/{table_name}/{file_name}.
    """
    try:
        logger.info(f"Uploading file {file.filename} to table {table_name} for user {user_id}")
        file_upload_result = await catalog_service.upload_file(
            file=file,
            table_name=table_name,
            create_new=create_new,
            user_id=user_id
        )
        logger.info(f"Successfully uploaded file {file.filename} to table {table_name} for user {user_id}")
        return file_upload_result
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading file: {str(e)}"
        )


@app.post("/api/catalog/descriptive_query", response_model=DescriptiveQueryResponse)
async def descriptive_query(
    request: DescriptiveQueryRequest = Body(...),
    agent: CatalogAgent = Depends(get_catalog_agent_dependency)
):
    try:
        if request.table_name:
            result = await agent.process_descriptive_query(request.query, request.table_name)
        else:
            # If no table_name, try to infer or return error (for now, error)
            return DescriptiveQueryResponse(status="error", message="table_name is required for now.")
        if result["status"] == "success":
            return DescriptiveQueryResponse(
                status="success",
                query=result.get("query"),
                results=result.get("results")
            )
        else:
            return DescriptiveQueryResponse(status="error", message=result.get("message"))
    except Exception as e:
        return DescriptiveQueryResponse(status="error", message=str(e))

@app.get("/api/catalog/s3-list", response_model=List[str])
def list_s3_files():
    # Use environment variables for credentials and region
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
        region_name=os.environ.get("AWS_REGION", "us-east-1")
    )
    bucket = "lambda-code-q"
    prefix = "originals/"
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    # Only return non-directory, non-empty keys
    files = [obj['Key'] for obj in resp.get('Contents', []) if obj.get('Key') and not obj['Key'].endswith('/')]
    return files

@app.post("/api/v1/users/{user_id}/tables")
async def create_user_table(
    user_id: str,
    table_name: str,
    schema: Dict[str, Any],
    original_data_location: str,
    parquet_data_location: str,
    file_format: str = "parquet",
    partition_keys: Optional[List[Dict[str, str]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    compression: str = "none",
    encryption: str = "none",
    catalog_service: CatalogService = Depends(get_catalog_service_dependency)
) -> Dict[str, Any]:
    """
    Create a new table in the user's database with full metadata organization.
    - Database: user_{user_id}
    - Table: {table_name}
    - Supports Parquet and CSV
    - Partition keys optional
    - Metadata includes:
        - original_data_location: S3 path to raw/original data
        - parquet_data_location: S3 path to processed Parquet data
        - created_at, updated_at timestamps
        - file_format, compression, encryption
        - Any additional metadata (e.g., lineage, version, etc.)
    - StorageDescriptor contains primary data location, serialization, columns, etc.
    """
    try:
        result = await catalog_service.create_user_table(
            user_id=user_id,
            table_name=table_name,
            schema=schema,
            original_data_location=original_data_location,
            parquet_data_location=parquet_data_location,
            file_format=file_format,
            partition_keys=partition_keys,
            metadata=metadata,
            compression=compression,
            encryption=encryption
        )
        return result
    except Exception as e:
        logger.error(f"Error creating user table: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error creating table: {str(e)}"
        )

@app.post("/api/catalog/tables/{table_name}/process")
async def process_table(
    table_name: str,
    user_id: str = Header("test_user", alias="X-User-Id"),
    catalog_service: CatalogService = Depends(get_catalog_service_dependency)
):
    """Process a table in the catalog."""
    try:
        result = await catalog_service.process_descriptive_query(
            query="Show me all data",
            table_name=table_name,
            user_id=user_id
        )
        return result
    except Exception as e:
        logger.error(f"Error processing table {table_name}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing table: {str(e)}"
        )

@app.get("/api/catalog/tables/{table_name}/files", response_model=List[FileMetadata])
async def get_table_files(
    table_name: str,
    s3: boto3.client = Depends(get_s3_client_dependency)
):
    """
    Get all files associated with a specific table.
    Files are stored under the table's S3 prefix.
    """
    global BUCKET_NAME
    if not BUCKET_NAME:
        raise HTTPException(status_code=503, detail="S3 bucket not configured.")

    try:
        # List objects under the table's prefix
        prefix = f"raw/{table_name}/"
        logger.info(f"Listing files for table {table_name} under prefix: {prefix}")
        
        files = []
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj.get('Key', '')
                if not key or key.endswith('/'):
                    continue
                    
                files.append(FileMetadata(
                    file_name=os.path.basename(key),
                    file_type=os.path.splitext(key)[1][1:] or "unknown",
                    s3_path=key,
                    schema_path=f"schemas/{table_name}/schema.json",
                    created_at=obj['LastModified'].isoformat(),
                    size=obj['Size']
                ))
        
        return files
    except Exception as e:
        logger.error(f"Error listing files for table {table_name}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

LOG_BUCKET = os.getenv("AWS_S3_BUCKET", "your-bucket-name")
LOG_REGION = os.getenv("AWS_REGION", "us-east-1")
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_RETENTION_MINUTES = int(os.getenv("LOG_RETENTION_MINUTES", "10"))
log_manager = LogManager(
    bucket=LOG_BUCKET,
    region=LOG_REGION,
    log_dir=LOG_DIR,
    retention_minutes=LOG_RETENTION_MINUTES
)

@app.get("/api/logs/recent")
def get_recent_logs():
    """Return recent log entries (last 10 minutes) from S3 as JSON."""
    log_keys = log_manager.list_recent_logs()
    all_logs = []
    for key in log_keys:
        try:
            obj = log_manager.s3.get_object(Bucket=LOG_BUCKET, Key=key)
            content = obj['Body'].read().decode('utf-8')
            for line in content.splitlines():
                try:
                    entry = json.loads(line)
                    all_logs.append(entry)
                except Exception:
                    continue
        except Exception:
            continue
    # Sort logs by timestamp descending
    all_logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return JSONResponse(content=all_logs)

if __name__ == "__main__":
    import uvicorn
    # Make sure uvicorn uses the app instance with the lifespan manager attached.
    # The app instance is already global and has lifespan attached.
    logger.info("Starting Uvicorn server for FastAPI app...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 