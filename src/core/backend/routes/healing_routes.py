"""Healing agent routes for the AWS Architect Agent."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
import logging
from botocore.exceptions import ClientError

from src.core.backend.services.healing_service import HealingService

router = APIRouter()
logger = logging.getLogger(__name__)


async def get_healing_service() -> HealingService:
    """Get the HealingService instance."""
    return HealingService()


@router.get("/health")
async def health_check():
    """Check if the healing service is healthy."""
    try:
        return {
            'status': 'healthy',
            'message': 'Healing service is running'
        }
    except Exception as e:
        logger.error(f'Health check failed: {str(e)}')
        raise HTTPException(
            status_code=500,
            detail=f'Health check failed: {str(e)}'
        )


@router.get("/logs/recent")
async def get_recent_logs(
    healing_service: HealingService = Depends(get_healing_service)
):
    """Get recent error logs."""
    try:
        # For now, return a mock response since we don't have actual logs
        return [{
            'timestamp': '2024-05-24T15:19:03.167Z',
            'event_type': 'workflow_exception',
            'exception': 'An error occurred while accessing the catalog',
            'details': 'The server is currently unavailable'
        }]
    except Exception as e:
        logger.error(f'Error fetching recent logs: {str(e)}')
        raise HTTPException(
            status_code=500,
            detail=f'Failed to fetch recent logs: {str(e)}'
        )


@router.get("/aws-credentials/status")
async def check_aws_credentials(
    healing_service: HealingService = Depends(get_healing_service)
):
    """Check AWS credentials status and provide suggestions if expired."""
    try:
        # Try to make a simple AWS API call
        healing_service.glue_client.get_databases()
        return {
            'status': 'valid',
            'message': 'AWS credentials are valid'
        }
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        
        if error_code == 'ExpiredTokenException':
            return {
                'status': 'expired',
                'message': 'AWS credentials have expired',
                'suggestions': [
                    'Refresh your AWS credentials',
                    'Check your AWS credentials in the environment variables',
                    'Verify that your AWS session token is still valid',
                    'If using temporary credentials, request new ones'
                ],
                'error_details': {
                    'code': error_code,
                    'message': error_message
                }
            }
        else:
            return {
                'status': 'error',
                'message': 'AWS credentials error',
                'error_details': {
                    'code': error_code,
                    'message': error_message
                }
            }
    except Exception as e:
        logger.error(f'Error checking AWS credentials: {str(e)}')
        return {
            'status': 'error',
            'message': 'Failed to check AWS credentials',
            'error_details': {
                'message': str(e)
            }
        }


@router.post("/retry")
async def retry_operation(
    data: Dict[str, Any],
    healing_service: HealingService = Depends(get_healing_service)
):
    """Retry a failed operation."""
    try:
        workflow_id = data.get('workflow_id')
        node_id = data.get('node_id')
        event_type = data.get('event_type')

        if not workflow_id or not event_type:
            raise HTTPException(
                status_code=400,
                detail='Missing required parameters'
            )

        result = healing_service.retry_operation(
            workflow_id=workflow_id,
            node_id=node_id,
            event_type=event_type
        )
        return result

    except Exception as e:
        logger.error(f'Error in retry operation: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/support/report")
async def report_to_support(
    error_log: Dict[str, Any],
    healing_service: HealingService = Depends(get_healing_service)
):
    """Report an error to support team."""
    try:
        if not error_log:
            raise HTTPException(
                status_code=400,
                detail='Missing error log data'
            )

        result = healing_service.report_to_support(error_log)
        return result

    except Exception as e:
        logger.error(f'Error reporting to support: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auto-fix")
async def auto_fix(
    error_log: Dict[str, Any],
    healing_service: HealingService = Depends(get_healing_service)
):
    """Attempt to automatically fix an error."""
    try:
        if not error_log:
            raise HTTPException(
                status_code=400,
                detail='Missing error log data'
            )

        result = healing_service.auto_fix(error_log)
        return result

    except Exception as e:
        logger.error(f'Error in auto-fix: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e)) 