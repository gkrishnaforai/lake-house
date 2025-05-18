"""FastAPI backend for the data lakehouse builder UI."""

from fastapi import (
    FastAPI, UploadFile, File, WebSocket, BackgroundTasks, HTTPException, Depends, Body, Header, Query
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
from io import BytesIO, StringIO
from botocore.exceptions import (
    ClientError, NoCredentialsError, PartialCredentialsError, BotoCoreError
)
import asyncio
import re
import traceback
import sys

from etl_architect_agent_v2.agents.classification.classification_agent import (
    ClassificationAgent,
    ClassificationState
)
# WebSocketManager seems unused, commenting out
# from etl_architect_agent_v2.api.websocket import (
# WebSocketManager, ClassificationProgress, ClassificationPreview
# )
# from etl_architect_agent_v2.api.websocket import (
#     ClassificationProgress, ClassificationPreview
# )
from etl_architect_agent_v2.core.llm.manager import LLMManager
# LLMError seems unused, commenting out
# from etl_architect_agent_v2.core.error_handler import LLMError 
# import time # Removed, asyncio.sleep is used
from etl_architect_agent_v2.agents.catalog_agent import (
    CatalogAgent, CatalogState
)
from etl_architect_agent_v2.backend.services.catalog_service import CatalogService
from etl_architect_agent_v2.backend.routes import catalog_routes
from etl_architect_agent_v2.backend.services.glue_service import GlueService
from etl_architect_agent_v2.backend.config import get_settings
from etl_architect_agent_v2.backend.models.catalog import (
    ColumnInfo,
    TableInfo,
    DatabaseCatalog,
    FileMetadata,
    FileInfo,
    FileConversionRequest,
    UploadProgress,
    UploadResponse,
    ChatRequest,
    ChatResponse,
    ClassificationRequest,
    DescriptiveQueryRequest,
    DescriptiveQueryResponse
)

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
LIFESPAN_DEBUG_LOG_FILE = "lifespan_debug.log" # Relative to workspace root

logger.info("AWS configuration will be initialized during application lifespan startup.")

# --- Lifespan Manager (defined BEFORE app instantiation) ---
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    # Attempt to open/create the debug log file immediately
    try:
        with open(LIFESPAN_DEBUG_LOG_FILE, "a") as f_debug_initial_check:
            f_debug_initial_check.write(f"{datetime.utcnow().isoformat()} - LIFESPAN FUNCTION ENTRY ATTEMPT (CONSTRUCTOR MODE)\n")
    except Exception as e_initial_log:
        # This print might not be visible in pytest if lifespan fails very early
        print(f"CRITICAL_LIFESPAN_LOG_FAIL: Could not open {LIFESPAN_DEBUG_LOG_FILE}: {str(e_initial_log)}")

    global s3_client, glue_client, catalog_service_instance, catalog_agent, llm_manager, classification_agent
    global BUCKET_NAME, DATABASE_NAME, aws_region

    # Open the debug log file in append mode
    with open(LIFESPAN_DEBUG_LOG_FILE, "a") as f_debug:
        f_debug.write(f"{datetime.utcnow().isoformat()} - LIFESPAN STARTING (CONSTRUCTOR MODE, after globals)\n")
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

        f_debug.write(f"{datetime.utcnow().isoformat()} - Read AWS Region: {aws_region}\n")
        logger.info(f"Read S3 Bucket: {BUCKET_NAME}")
        f_debug.write(f"{datetime.utcnow().isoformat()} - Read S3 Bucket: {BUCKET_NAME}\n")
        logger.info(f"Read Glue Database: {DATABASE_NAME}")
        f_debug.write(f"{datetime.utcnow().isoformat()} - Read Glue Database: {DATABASE_NAME}\n")
        key_id_to_log = f"{aws_access_key_local[:4]}...{aws_access_key_local[-4:]}" if aws_access_key_local and len(aws_access_key_local) > 8 else "Not Set or Too Short"
        logger.info(f"AWS Access Key ID: {key_id_to_log}")
        f_debug.write(f"{datetime.utcnow().isoformat()} - AWS Access Key ID: {key_id_to_log}\n")
        logger.info(f"AWS Secret Key: {'Set (masked)' if aws_secret_key_local else 'Not Set'}")
        f_debug.write(f"{datetime.utcnow().isoformat()} - AWS Secret Key: {'Set (masked)' if aws_secret_key_local else 'Not Set'}\n")
        logger.info(f"AWS Session Token: {'Set' if aws_session_token_local else 'Not Set'}")
        f_debug.write(f"{datetime.utcnow().isoformat()} - AWS Session Token: {'Set' if aws_session_token_local else 'Not Set'}\n")

        try:
            f_debug.write(f"{datetime.utcnow().isoformat()} - LIFESPAN_STAGE: Attempting to initialize S3 client...\n")
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_local,
                aws_secret_access_key=aws_secret_key_local,
                aws_session_token=aws_session_token_local,
                region_name=aws_region
            )
            f_debug.write(f"{datetime.utcnow().isoformat()} - LIFESPAN_STAGE: S3 client initialized successfully.\n")

            f_debug.write(f"{datetime.utcnow().isoformat()} - LIFESPAN_STAGE: Attempting to initialize Glue client...\n")
            glue_client = boto3.client(
                'glue',
                aws_access_key_id=aws_access_key_local,
                aws_secret_access_key=aws_secret_key_local,
                aws_session_token=aws_session_token_local,
                region_name=aws_region
            )
            f_debug.write(f"{datetime.utcnow().isoformat()} - LIFESPAN_STAGE: Glue client initialized successfully.\n")

            if not validate_aws_config_internal():
                f_debug.write(f"{datetime.utcnow().isoformat()} - LIFESPAN_VALIDATION_FAILURE: AWS configuration validation failed.\n")
            else:
                f_debug.write(f"{datetime.utcnow().isoformat()} - Lifespan: AWS configuration validated successfully.\n")

            f_debug.write(f"{datetime.utcnow().isoformat()} - LIFESPAN_STAGE: Attempting to initialize CatalogService...\n")
            catalog_service_instance = CatalogService(
                bucket=BUCKET_NAME,
                aws_region=aws_region
            )
            f_debug.write(f"{datetime.utcnow().isoformat()} - LIFESPAN_STAGE: CatalogService initialized successfully.\n")

            f_debug.write(f"{datetime.utcnow().isoformat()} - LIFESPAN_STAGE: Attempting to initialize CatalogAgent...\n")
            catalog_agent = CatalogAgent(catalog_service=catalog_service_instance)
            f_debug.write(f"{datetime.utcnow().isoformat()} - LIFESPAN_STAGE: CatalogAgent initialized successfully.\n")

            f_debug.write(f"{datetime.utcnow().isoformat()} - LIFESPAN_STAGE: Attempting to initialize LLMManager...\n")
            llm_manager = LLMManager()
            f_debug.write(f"{datetime.utcnow().isoformat()} - LIFESPAN_STAGE: LLMManager initialized successfully.\n")

            f_debug.write(f"{datetime.utcnow().isoformat()} - LIFESPAN_STAGE: Attempting to initialize ClassificationAgent...\n")
            classification_agent = ClassificationAgent()
            f_debug.write(f"{datetime.utcnow().isoformat()} - LIFESPAN_STAGE: ClassificationAgent initialized successfully.\n")

        except Exception as e:
            f_debug.write(f"{datetime.utcnow().isoformat()} - LIFESPAN_EXCEPTION_FILE (CONSTRUCTOR_MODE): Exception caught: {str(e)}\nTraceback:\n{traceback.format_exc()}\n")
            # Set to None so dependencies can check
            s3_client = None
            glue_client = None
            catalog_service_instance = None
            catalog_agent = None
            llm_manager = None
            classification_agent = None
        
        f_debug.write(f"{datetime.utcnow().isoformat()} - LIFESPAN YIELDING (CONSTRUCTOR MODE)\n")
    yield

    with open(LIFESPAN_DEBUG_LOG_FILE, "a") as f_debug:
        f_debug.write(f"{datetime.utcnow().isoformat()} - LIFESPAN SHUTDOWN (CONSTRUCTOR MODE)\n")

# Instantiate FastAPI app with lifespan manager
app = FastAPI(lifespan=lifespan)

from etl_architect_agent_v2.backend.routes import catalog_routes
app.include_router(catalog_routes.router, prefix="/api/catalog")

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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/api/health")
async def health_check():
    global s3_client, glue_client, BUCKET_NAME, DATABASE_NAME # Use globals set by lifespan
    aws_clients_were_initialized = s3_client is not None and glue_client is not None
    
    current_aws_validity = False
    if aws_clients_were_initialized:
        try:
            s3_client.list_buckets() # Light check
            current_aws_validity = True
        except Exception as e:
            logger.warning(f"Health check: Current AWS credentials seem to have an issue: {str(e)}")
            current_aws_validity = False # Explicitly false if check fails

    return {
        "status": "ok",
        "aws_clients_initialized_on_startup": aws_clients_were_initialized,
        "aws_credentials_currently_valid": current_aws_validity,
        "bucket_name": BUCKET_NAME if BUCKET_NAME else "Not Set",
        "database_name": DATABASE_NAME if DATABASE_NAME else "Not Set"
    }

def create_glue_table( # This is an old helper, ensure it uses global clients
    table_name: str, schema: Dict[str, Any], s3_path: str,
    s3: boto3.client = Depends(get_s3_client_dependency), # Attempt to inject if possible
    glue: boto3.client = Depends(get_glue_client_dependency) # Or rely on globals
) -> None:
    """Create a Glue table for the uploaded file. Uses global glue_client and DATABASE_NAME."""
    global glue_client, DATABASE_NAME # Fallback to globals if Depends not usable here
    current_glue_client = glue # Use injected if provided, else global
    if not current_glue_client: current_glue_client = get_glue_client_dependency() # explicit call
    
    if not DATABASE_NAME:
        logger.error("create_glue_table: DATABASE_NAME not configured.")
        raise RuntimeError("DATABASE_NAME not configured for Glue table creation.")

    try:
        columns = []
        for col in schema["columns"]:
            col_type = col["type"]
            # Simplified type mapping, consider centralizing this logic
            if "int" in col_type: glue_type = "bigint"
            elif "float" in col_type: glue_type = "double"
            elif "datetime" in col_type: glue_type = "timestamp"
            else: glue_type = "string"
            columns.append({"Name": col["name"], "Type": glue_type})

        logger.info(f"Creating Glue table '{table_name}' in DB '{DATABASE_NAME}' pointing to '{s3_path}'")
        current_glue_client.create_table(
            DatabaseName=DATABASE_NAME,
            TableInput={
                "Name": table_name,
                "StorageDescriptor": {
                    "Columns": columns,
                    "Location": s3_path, # This should be the S3 PREFIX (folder)
                    "InputFormat": "org.apache.hadoop.mapred.TextInputFormat", # For CSV, Parquet is different
                    "OutputFormat": "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
                    "SerdeInfo": {
                        "SerializationLibrary": "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe",
                        "Parameters": {"field.delim": ","} # For CSV
                    }
                },
                "TableType": "EXTERNAL_TABLE",
                "Parameters": {"classification": "csv"} # Adjust if not CSV
            }
        )
        logger.info(f"Glue table {table_name} created successfully.")
    except Exception as e:
        logger.error(f"Error creating Glue table {table_name}: {str(e)}", exc_info=True)
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
async def upload_file( # RESTORED ORIGINAL NAME
    file: UploadFile = File(...),
    s3: boto3.client = Depends(get_s3_client_dependency),
    # This old endpoint directly calls catalog_agent.process_file.
    # It would need the catalog_agent initialized by lifespan.
    agent: CatalogAgent = Depends(get_catalog_agent_dependency)
    ):
    """DEPRECATED by /catalog/tables/{table_name}/files.
    Upload a file to S3 and process it for catalog (old method).
    Uses injected s3_client, global BUCKET_NAME, and injected agent.
    """
    global BUCKET_NAME # From Lifespan
    logger.warning("Deprecated /api/upload endpoint called. Consider migrating to /catalog/tables/{table_name}/files.")
    if not BUCKET_NAME:
        return UploadResponse(status="error", message="S3 bucket not configured.", file_name=file.filename, s3_path="", error="Server config error")

    upload_id = str(uuid.uuid4())
    s3_target_key = f"raw/{upload_id}/{file.filename}" # Full key for S3
    
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
        
        logger.info(f"Uploading {file.filename} to s3://{BUCKET_NAME}/{s3_target_key} via old endpoint.")
        s3.upload_file(temp_file_path, BUCKET_NAME, s3_target_key)
        full_s3_path = f"s3://{BUCKET_NAME}/{s3_target_key}"
        logger.info(f"File {file.filename} uploaded to {full_s3_path} by old endpoint.")

        # This part is tricky because the old CatalogAgent.process_file might expect
        # different things or make assumptions that are no longer valid with CatalogService.
        # For now, let's assume this endpoint is mostly for raw upload and basic cataloging.
        # The CatalogState might need user_id, which isn't available here.
        state = CatalogState(
            file_path=full_s3_path, # process_file usually expects S3 path
            file_name=file.filename,
            file_type=file.content_type,
            s3_path=full_s3_path # Redundant but for consistency
        )
        logger.info(f"Calling agent.process_file for {file.filename} (old endpoint)")
        # agent (CatalogAgent) is injected. Its catalog_service should be the one from lifespan.
        processed_state = await agent.process_file(state, user_id="AgenticUser_OldUpload")

        if processed_state.error:
            logger.error(f"Old /api/upload: Error processing file with agent: {processed_state.error}")
            return UploadResponse(
                status="error", message=f"Error processing file: {processed_state.error}",
                file_name=file.filename, s3_path=full_s3_path, error=processed_state.error,
                details={"agent_status": processed_state.status}
            )

        return UploadResponse(
            status="success",
            message="File uploaded and processed via old endpoint.",
            file_name=file.filename,
            s3_path=full_s3_path, # This is the raw path for this endpoint
            table_name=processed_state.table_name,
            details={
                "agent_status": processed_state.status,
                "agent_table_name": processed_state.table_name,
                "agent_schema_path": processed_state.schema_path,
                "notes": "This endpoint is deprecated. Results might differ from new endpoint."
            }
        )

    except Exception as e:
        logger.error(f"Error in old /api/upload for {file.filename}: {str(e)}", exc_info=True)
        return UploadResponse(
            status="error", message=f"Unexpected error: {str(e)}",
            file_name=file.filename, s3_path="", error=str(e)
        )
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_data(request: ChatRequest): # RESTORED ORIGINAL NAME
    # This endpoint uses pandas but not direct AWS clients.
    # It relies on data passed in the request.
    message = request.message.lower()
    schema_data = request.schema # Renamed to avoid conflict with 'schema' module
    sample_data = request.sample_data
    sql_query_from_req = request.sql # Renamed

    if not schema_data or not sample_data:
        return JSONResponse(status_code=400, content={"error": "Schema and sample data required."})

    if sql_query_from_req:
        try:
            df = pd.DataFrame(sample_data)
            # This execute_sql_query is a local dummy.
            # Real SQL execution would use Athena via CatalogAgent/Service.
            results = execute_sql_query(sql_query_from_req, df) 
            return ChatResponse(response="Executed provided SQL.", sql=sql_query_from_req, data=results)
        except Exception as e:
            logger.error(f"Error executing provided SQL in /api/chat: {str(e)}", exc_info=True)
            return ChatResponse(response="Error executing SQL.", sql=sql_query_from_req, error=str(e))

    # Simplified LLM call for SQL generation (if LLM manager is available)
    # This endpoint is more of a mock without a full LLM chain for SQL gen.
    # For now, keeping the simple rule-based logic.
    # llm = get_llm_manager_dependency() # If we were to use LLM here

    response_text = "Could not generate SQL from your query."
    generated_sql = None
    if "show" in message and "top" in message:
        numbers = re.findall(r'\d+', message)
        limit = int(numbers[0]) if numbers else 5
        generated_sql = f"SELECT * FROM YourTable LIMIT {limit}" # Placeholder table name
        response_text = f"To show top {limit} rows:"
    # ... (other simple rules) ...
    else:
        response_text = "Please ask a more specific query for SQL generation."

    return ChatResponse(response=response_text, sql=generated_sql)


# Dummy execute_sql_query for the /api/chat endpoint's current implementation
def execute_sql_query(sql: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
    logger.info(f"Dummy execute_sql_query called with: {sql}")
    if "limit" in sql.lower():
        limit = int(sql.lower().split("limit")[1].strip())
        return df.head(limit).to_dict(orient='records')
    return df.head(5).to_dict(orient='records') # Default preview


@app.websocket("/ws/classification/{client_id}")
async def classification_websocket(websocket: WebSocket, client_id: str,
                                 agent: ClassificationAgent = Depends(get_classification_agent_dependency)):
    # WebSocketManager was removed, direct agent interaction might be complex here.
    # This endpoint needs a proper manager for websocket connections and broadcasting.
    # For now, it will connect but not do much beyond that.
    await websocket.accept()
    logger.info(f"WebSocket connection established for client_id: {client_id}")
    try:
        while True:
            data = await websocket.receive_text() # Or receive_json
            logger.info(f"WebSocket received from {client_id}: {data}")
            # Example: await websocket.send_text(f"Message received: {data}")
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {str(e)}", exc_info=True)
    finally:
        logger.info(f"WebSocket connection closed for client_id: {client_id}")


@app.post("/api/classify")
async def classify_data(
    request: ClassificationRequest,
    background_tasks: BackgroundTasks,
    agent: ClassificationAgent = Depends(get_classification_agent_dependency) # Use DI
):
    state = ClassificationState(
        classification_needed=True,
        user_instruction=request.user_instruction,
        original_data_path=request.data_path
    )
    # Pass the injected agent to the background task
    background_tasks.add_task(process_classification, request.client_id, state, agent)
    return {"status": "started", "message": "Classification process started"}

async def process_classification(client_id: str, state: ClassificationState, agent: ClassificationAgent):
    # Note: WebSocketManager is not used here as it was removed.
    # Sending updates back would require a different mechanism if websockets are live.
    try:
        logger.info(f"Background task: Starting classification for client {client_id}")
        processed_state = await agent.process(state) # Use passed agent
        logger.info(f"Background task: Classification for {client_id} status: {processed_state.status}")
        if processed_state.error:
            logger.error(f"Background task: Classification error for {client_id}: {processed_state.error}")
        # How to send updates back without WebSocketManager?
        # This would require a pub/sub or shared state accessible by the websocket handler.
    except Exception as e:
        logger.error(f"Background task: Error in classification process for {client_id}: {str(e)}", exc_info=True)

async def get_table_sample_data( # Uses global aws_region, DATABASE_NAME, BUCKET_NAME
    table_name: str, limit: int = 5
    # Cannot easily inject boto3 client here as it's not a FastAPI route
    # It will rely on global aws_region set by lifespan for its own client creation.
) -> List[Dict[str, Any]]:
    global aws_region, DATABASE_NAME, BUCKET_NAME # These are set in lifespan
    if not all([aws_region, DATABASE_NAME, BUCKET_NAME]):
        logger.error("get_table_sample_data: Essential AWS config not set globally.")
        return []
    try:
        athena_client_local = boto3.client('athena', region_name=aws_region)
        escaped_table_name = f'"{table_name}"' # Basic escaping
        
        logger.info(f"Querying Athena for sample data: {DATABASE_NAME}.{escaped_table_name}, limit {limit}")
        response = athena_client_local.start_query_execution(
            QueryString=f'SELECT * FROM "{DATABASE_NAME}".{escaped_table_name} LIMIT {limit}',
            QueryExecutionContext={'Database': DATABASE_NAME},
            ResultConfiguration={'OutputLocation': f's3://{BUCKET_NAME}/athena_query_results/sample_data/'}
        )
        query_execution_id = response['QueryExecutionId']
        
        while True: # Polling for query completion
            query_status_response = athena_client_local.get_query_execution(QueryExecutionId=query_execution_id)
            state = query_status_response['QueryExecution']['Status']['State']
            if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                break
            await asyncio.sleep(1) # Needs to be async if called from async context, or use time.sleep

        if state == 'SUCCEEDED':
            results_response = athena_client_local.get_query_results(QueryExecutionId=query_execution_id)
            if 'ResultSet' in results_response and 'Rows' in results_response['ResultSet']:
                rows = results_response['ResultSet']['Rows']
                if not rows: return []
                headers = [col_data.get('VarCharValue') for col_data in rows[0]['Data']]
                
                data_list = []
                for row_data in rows[1:]: # Skip header row
                    data_list.append(dict(zip(headers, [item.get('VarCharValue') for item in row_data['Data']])))
                return data_list
        else:
            logger.error(f"Athena query for sample data failed. State: {state}. Reason: {query_status_response['QueryExecution']['Status'].get('StateChangeReason')}")
            return []
        return [] # Default empty
    except Exception as e:
        logger.error(f"Error in get_table_sample_data for {table_name}: {str(e)}", exc_info=True)
        return []

async def analyze_table_schema( # Uses global llm_manager
    table_name: str, columns: List[Dict[str, Any]], sample_data: List[Dict[str, Any]],
    llm: LLMManager = Depends(get_llm_manager_dependency) # Use DI for LLMManager
) -> TableInfo:
    # sample_data is already List[Dict] here.
    column_infos_for_llm = []
    for col_dict in columns: # Assuming columns is List[Dict] like {'Name': 'id', 'Type': 'int'}
        col_name = col_dict.get("Name")
        if not col_name: continue
        sample_values_for_col = [row.get(col_name) for row in sample_data[:3] if row] # Max 3 samples
        column_infos_for_llm.append({
            "name": col_name,
            "type": col_dict.get("Type"),
            "sample_values": [str(v) for v in sample_values_for_col if v is not None] # Convert samples to string for prompt
        })

    context = {"table_name": table_name, "columns": column_infos_for_llm}
    
    table_prompt = (
        f"Analyze this table schema: {json.dumps(context, indent=2)}. "
        "Provide a concise description of what this table likely contains."
    )
    table_description = "LLM-generated description placeholder." # Default
    try:
        table_description = await llm.generate_response(table_prompt)
    except Exception as e:
        logger.error(f"LLM error generating table description for {table_name}: {e}")


    final_columns_info = []
    for col_dict in columns:
        col_name = col_dict.get("Name")
        if not col_name: continue
        col_type = col_dict.get("Type")
        sample_values_for_col = [row.get(col_name) for row in sample_data[:3] if row]

        col_prompt_context = {
            "column_name": col_name, "type": col_type, 
            "sample_values": [str(v) for v in sample_values_for_col if v is not None]
        }
        col_prompt = (
            f"Analyze this column: {json.dumps(col_prompt_context)}. "
            "Provide a brief description of what this column likely represents."
        )
        col_desc = "LLM-generated column description placeholder."
        try:
            col_desc = await llm.generate_response(col_prompt)
        except Exception as e:
            logger.error(f"LLM error for column {col_name} in {table_name}: {e}")
        
        final_columns_info.append(ColumnInfo(
            name=col_name, type=col_type, description=col_desc,
            sample_values=sample_values_for_col # Keep original sample values
        ))
        
    return TableInfo(
        name=table_name, description=table_description, columns=final_columns_info,
        s3_location=f"s3://{BUCKET_NAME}/raw/{table_name}" # Example S3 location
    )

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
async def get_data_quality(table_name: str, s3: boto3.client = Depends(get_s3_client_dependency)): # RESTORED
    global BUCKET_NAME # From lifespan
    if not BUCKET_NAME:
        raise HTTPException(status_code=503, detail="S3 bucket not configured.")
        
    # Similar to audit, assumes a specific metadata file convention.
    # Example: metadata/quality/{table_name}.json
    quality_metadata_key = f"metadata/quality/{table_name}.json"
    logger.info(f"Attempting to fetch quality metadata from s3://{BUCKET_NAME}/{quality_metadata_key}")
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=quality_metadata_key)
        quality_data = json.loads(response['Body'].read().decode('utf-8'))
        # The new CatalogService stores quality metrics directly.
        # This endpoint might be for a different format.
        # For now, return the content of the quality file.
        return quality_data # Return the whole JSON content of the quality file.
    except s3.exceptions.NoSuchKey:
        logger.warning(f"No quality metadata file found at {quality_metadata_key} for table {table_name}")
        raise HTTPException(status_code=404, detail=f"Quality metrics not found for table {table_name}")
    except Exception as e:
        logger.error(f"Error fetching data quality for {table_name}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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

if __name__ == "__main__":
    import uvicorn
    # Make sure uvicorn uses the app instance with the lifespan manager attached.
    # The app instance is already global and has lifespan attached.
    logger.info("Starting Uvicorn server for FastAPI app...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 