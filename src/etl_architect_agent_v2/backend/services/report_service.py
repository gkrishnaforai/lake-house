import json
import os
import boto3
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import uuid
from fastapi import HTTPException

class Report(BaseModel):
    id: str
    name: str
    description: str
    query: str
    schedule: Optional[str] = ""
    lastRun: Optional[datetime] = None
    createdBy: str
    createdAt: datetime
    updatedAt: datetime
    isFavorite: bool = False
    charts: List[Dict[str, Any]] = Field(default_factory=list)

class ReportService:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.bucket_name = os.getenv('AWS_S3_BUCKET', 'your-bucket-name')
        self.reports_prefix = 'reports/'
        self._ensure_reports_bucket()

    def _ensure_reports_bucket(self):
        """Ensure the reports bucket exists."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except Exception as e:
            print(f"Error checking bucket: {str(e)}")
            self.s3_client.create_bucket(Bucket=self.bucket_name)

    def _get_report_key(self, user_id: str, report_id: str) -> str:
        """Get the S3 key for a report."""
        return f"{self.reports_prefix}{user_id}/{report_id}.json"

    async def get_user_reports(self, user_id: str) -> List[Report]:
        """Get all reports for a user."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f"{self.reports_prefix}{user_id}/"
            )
            
            reports = []
            for obj in response.get('Contents', []):
                report_data = self.s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=obj['Key']
                )
                report_json = json.loads(report_data['Body'].read().decode('utf-8'))
                reports.append(Report(**report_json))
            
            return sorted(reports, key=lambda x: x.updatedAt, reverse=True)
        except Exception as e:
            print(f"Error getting user reports: {str(e)}")
            return []

    async def get_report(self, user_id: str, report_id: str) -> Optional[Report]:
        """Get a specific report."""
        try:
            report_data = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=self._get_report_key(user_id, report_id)
            )
            report_json = json.loads(report_data['Body'].read().decode('utf-8'))
            return Report(**report_json)
        except Exception as e:
            print(f"Error getting report: {str(e)}")
            return None

    async def save_report(self, user_id: str, report: Report) -> Report:
        """Save a new report for a user."""
        try:
            # Ensure the reports bucket exists
            self._ensure_reports_bucket()
            
            # Generate a unique ID for the report
            report_id = str(uuid.uuid4())
            
            # Add metadata
            now = datetime.now()
            report_json = {
                "id": report_id,
                "name": report.name,
                "description": report.description,
                "query": report.query,
                "schedule": report.schedule or "",
                "lastRun": None,
                "createdBy": user_id,
                "createdAt": now.isoformat(),
                "updatedAt": now.isoformat(),
                "isFavorite": report.isFavorite,
                "charts": report.charts or []
            }
            
            # Save to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self._get_report_key(user_id, report_id),
                Body=json.dumps(report_json)
            )
            
            # Convert ISO format strings back to datetime objects for the response
            report_json["createdAt"] = now
            report_json["updatedAt"] = now
            
            return Report(**report_json)
            
        except Exception as e:
            print(f"Error saving report: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save report: {str(e)}"
            )

    async def update_report(self, user_id: str, report_id: str, report: Report) -> Optional[Report]:
        """Update an existing report."""
        try:
            # Ensure the reports bucket exists
            self._ensure_reports_bucket()
            
            # Get existing report
            existing_report = await self.get_report(user_id, report_id)
            if not existing_report:
                return None
            
            # Update metadata
            now = datetime.now()
            report_json = {
                "id": report_id,
                "name": report.name,
                "description": report.description,
                "query": report.query,
                "schedule": report.schedule or "",
                "lastRun": existing_report.lastRun.isoformat() if existing_report.lastRun else None,
                "createdBy": existing_report.createdBy,
                "createdAt": existing_report.createdAt.isoformat(),
                "updatedAt": now.isoformat(),
                "isFavorite": report.isFavorite,
                "charts": report.charts or []
            }
            
            # Save to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self._get_report_key(user_id, report_id),
                Body=json.dumps(report_json)
            )
            
            # Convert ISO format strings back to datetime objects for the response
            report_json["createdAt"] = existing_report.createdAt
            report_json["updatedAt"] = now
            if existing_report.lastRun:
                report_json["lastRun"] = existing_report.lastRun
            
            return Report(**report_json)
            
        except Exception as e:
            print(f"Error updating report: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to update report: {str(e)}"
            )

    async def delete_report(self, user_id: str, report_id: str) -> bool:
        """Delete a report."""
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=self._get_report_key(user_id, report_id)
            )
            return True
        except Exception as e:
            print(f"Error deleting report: {str(e)}")
            return False

    async def run_report(self, user_id: str, report_id: str) -> Dict[str, Any]:
        """Run a report and get its results."""
        report = await self.get_report(user_id, report_id)
        if not report:
            return None
        
        # Update last run timestamp
        report.lastRun = datetime.now()
        await self.update_report(user_id, report_id, report)
        
        # Here you would typically execute the report query
        # For now, return a mock result
        return {
            "message": "Report executed successfully",
            "results": [],
            "lastRun": report.lastRun.isoformat()
        }

    async def schedule_report(self, user_id: str, report_id: str, schedule: str) -> bool:
        """Schedule a report to run periodically."""
        report = await self.get_report(user_id, report_id)
        if not report:
            return False
        
        report.schedule = schedule
        updated_report = await self.update_report(user_id, report_id, report)
        return updated_report is not None 