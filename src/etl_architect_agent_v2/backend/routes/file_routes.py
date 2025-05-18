# This file is deprecated. All file operations have been moved to catalog_routes.py
# Please use the catalog API endpoints instead. 

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from typing import List, Dict, Any
from etl_architect_agent_v2.backend.services.s3_service import S3Service
from etl_architect_agent_v2.backend.config import get_settings

router = APIRouter(tags=["files"])

def get_s3_service():
    settings = get_settings()
    return S3Service(
        bucket=settings.AWS_S3_BUCKET,
        aws_region=settings.AWS_REGION
    )

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    s3_service: S3Service = Depends(get_s3_service)
):
    """Upload a file to S3."""
    try:
        content = await file.read()
        key = f"uploads/{file.filename}"
        result = await s3_service.upload_file(content, key)
        return {
            "status": "success",
            "message": f"File {file.filename} uploaded successfully",
            "file_info": {
                "name": file.filename,
                "size": len(content),
                "location": f"s3://{s3_service.bucket}/{key}"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{key:path}")
async def get_file(
    key: str,
    s3_service: S3Service = Depends(get_s3_service)
):
    """Get a file from S3."""
    try:
        content = await s3_service.get_file(key)
        return {"content": content.decode()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{key:path}")
async def delete_file(
    key: str,
    s3_service: S3Service = Depends(get_s3_service)
):
    """Delete a file from S3."""
    try:
        result = await s3_service.delete_file(key)
        return {
            "status": "success",
            "message": f"File {key} deleted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 