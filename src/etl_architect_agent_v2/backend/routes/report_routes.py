from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from ..services.report_service import ReportService
from ..auth import get_current_user, get_test_user
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()
report_service = ReportService()

class Report(BaseModel):
    id: Optional[str] = None
    name: str
    description: str
    query: str
    schedule: Optional[str] = ""
    lastRun: Optional[datetime] = None
    createdBy: Optional[str] = None
    createdAt: Optional[datetime] = None
    updatedAt: Optional[datetime] = None
    isFavorite: bool = False
    charts: List[Dict[str, Any]] = []

@router.get("/{user_id}")
async def get_user_reports(
    user_id: str,
    current_user: Any = Depends(get_test_user)  # Using test user for development
) -> List[Report]:
    """Get all reports for a user."""
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access these reports")
    return await report_service.get_user_reports(user_id)

@router.get("/{user_id}/{report_id}")
async def get_report(
    user_id: str,
    report_id: str,
    current_user: Any = Depends(get_test_user)  # Using test user for development
) -> Report:
    """Get a specific report."""
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this report")
    report = await report_service.get_report(user_id, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report

@router.post("/{user_id}")
async def create_report(
    user_id: str,
    report: Report,
    current_user: Any = Depends(get_test_user)  # Using test user for development
) -> Report:
    """Create a new report."""
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to create reports for this user")
    return await report_service.save_report(user_id, report)

@router.put("/{user_id}/{report_id}")
async def update_report(
    user_id: str,
    report_id: str,
    report: Report,
    current_user: Any = Depends(get_test_user)  # Using test user for development
) -> Report:
    """Update an existing report."""
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to update this report")
    updated_report = await report_service.update_report(user_id, report_id, report)
    if not updated_report:
        raise HTTPException(status_code=404, detail="Report not found")
    return updated_report

@router.delete("/{user_id}/{report_id}")
async def delete_report(
    user_id: str,
    report_id: str,
    current_user: Any = Depends(get_test_user)  # Using test user for development
) -> Dict[str, str]:
    """Delete a report."""
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this report")
    success = await report_service.delete_report(user_id, report_id)
    if not success:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"message": "Report deleted successfully"}

@router.post("/{user_id}/{report_id}/run")
async def run_report(
    user_id: str,
    report_id: str,
    current_user: Any = Depends(get_test_user)  # Using test user for development
) -> Dict[str, Any]:
    """Run a report and get its results."""
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to run this report")
    result = await report_service.run_report(user_id, report_id)
    if not result:
        raise HTTPException(status_code=404, detail="Report not found")
    return result

@router.post("/{user_id}/{report_id}/schedule")
async def schedule_report(
    user_id: str,
    report_id: str,
    schedule: str,
    current_user: Any = Depends(get_test_user)  # Using test user for development
) -> Dict[str, str]:
    """Schedule a report to run periodically."""
    if current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to schedule this report")
    success = await report_service.schedule_report(user_id, report_id, schedule)
    if not success:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"message": "Report scheduled successfully"} 