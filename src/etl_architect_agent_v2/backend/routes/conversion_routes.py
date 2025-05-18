from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from pydantic import BaseModel
from ..services.conversion_service import ConversionService
from ..config import get_settings
import json


router = APIRouter(prefix="/api/convert", tags=["conversion"])


class ConversionRequest(BaseModel):
    file_path: str
    target_format: str = "avro"


# Dependency to get conversion service instance
def get_conversion_service():
    settings = get_settings()
    return ConversionService(aws_region=settings.AWS_REGION)


@router.post("")
async def convert_file(
    request: ConversionRequest,
    conversion_service: ConversionService = Depends(get_conversion_service)
):
    """
    Convert a file to the target format
    """
    try:
        result = await conversion_service.convert_file(
            file_path=request.file_path,
            target_format=request.target_format
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{file_name}")
async def get_conversion_status(
    file_name: str,
    conversion_service: ConversionService = Depends(get_conversion_service)
):
    """
    Get conversion status for a file
    """
    try:
        # Get status from S3 metadata
        status_key = f"metadata/conversion/{file_name}.json"
        try:
            response = conversion_service.s3_client.get_object(
                Bucket=conversion_service.bucket,
                Key=status_key
            )
            status = json.loads(response['Body'].read())
            return status
        except conversion_service.s3_client.exceptions.NoSuchKey:
            return {
                "status": "not_found",
                "message": "No conversion status found for this file"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 