from fastapi import APIRouter, UploadFile, HTTPException
from src.core.agents.workflows.file_upload_workflow import FileUploadWorkflow
import os
import logging
from datetime import datetime
import uuid
import boto3

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/tables/{table_name}/files")
async def upload_table_file(
    table_name: str,
    file: UploadFile,
    database_name: str = "default",
    user_id: str = "default"
):
    """Upload a file and create an Iceberg table."""
    try:
        # Generate unique upload ID
        upload_id = str(uuid.uuid4())
        
        # Save uploaded file to S3
        bucket = os.getenv("AWS_S3_BUCKET", "your-bucket-name")
        s3_key = f"raw/uploads/{user_id}/{upload_id}/original.xlsx"
        
        # Save file to temp location first
        temp_file = (
            f"/tmp/{file.filename}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        try:
            with open(temp_file, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Upload to S3
            s3_client = boto3.client('s3')
            s3_client.upload_file(temp_file, bucket, s3_key)
            
            # Create and execute workflow
            workflow = FileUploadWorkflow()
            workflow_state = workflow.create_workflow(
                file_path=temp_file,
                table_name=table_name,
                database_name=database_name,
                user_id=user_id,
                upload_id=upload_id
            )
            
            result = await workflow.execute_workflow(workflow_state)
            
            if result["status"] == "error":
                raise HTTPException(
                    status_code=500,
                    detail=f"Workflow error: {result['message']}"
                )
            
            return {
                "status": "success",
                "message": f"File uploaded and table {table_name} created",
                "workflow_id": result["workflow_id"]
            }
            
        finally:
            # Cleanup temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.info(f"Cleaned up temporary file: {temp_file}")
                
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading file: {str(e)}"
        ) 