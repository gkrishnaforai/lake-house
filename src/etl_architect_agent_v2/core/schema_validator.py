"""Schema validator for JSON schemas."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class ValidationIssue:
    """Represents a validation issue found in the schema."""
    message: str
    path: str
    severity: str = "error"  # error, warning, info

@dataclass
class ValidationResult:
    """Result of schema validation."""
    is_valid: bool
    issues: List[ValidationIssue]
    fixed_content: Optional[Dict[str, Any]] = None

class SchemaValidator:
    """Validates JSON schemas for data catalog entries."""
    
    def __init__(self):
        """Initialize the schema validator."""
        self.required_fields = {
            "columns": list,
            "name": str,
            "type": str,
            "description": str,
            "sample_value": str,
            "quality_metrics": dict
        }
        
        self.quality_metrics = {
            "completeness": float,
            "uniqueness": float,
            "validity": float
        }
        
    def validate(self, content: Dict[str, Any]) -> ValidationResult:
        """Validate the schema content."""
        issues = []
        
        # Check if content is a dictionary
        if not isinstance(content, dict):
            issues.append(ValidationIssue(
                message="Schema must be a JSON object",
                path="",
                severity="error"
            ))
            return ValidationResult(is_valid=False, issues=issues)
            
        # Check for required top-level fields
        if "columns" not in content:
            issues.append(ValidationIssue(
                message="Missing required field: columns",
                path="",
                severity="error"
            ))
            return ValidationResult(is_valid=False, issues=issues)
            
        if not isinstance(content["columns"], list):
            issues.append(ValidationIssue(
                message="'columns' must be an array",
                path="columns",
                severity="error"
            ))
            return ValidationResult(is_valid=False, issues=issues)
            
        # Validate each column
        for i, column in enumerate(content["columns"]):
            if not isinstance(column, dict):
                issues.append(ValidationIssue(
                    message=f"Column at index {i} must be an object",
                    path=f"columns[{i}]",
                    severity="error"
                ))
                continue
                
            # Check required column fields
            for field, field_type in self.required_fields.items():
                if field not in column:
                    issues.append(ValidationIssue(
                        message=f"Missing required field: {field}",
                        path=f"columns[{i}]",
                        severity="error"
                    ))
                elif not isinstance(column[field], field_type):
                    issues.append(ValidationIssue(
                        message=f"Field '{field}' must be of type {field_type.__name__}",
                        path=f"columns[{i}].{field}",
                        severity="error"
                    ))
                    
            # Validate quality metrics
            if "quality_metrics" in column:
                metrics = column["quality_metrics"]
                if not isinstance(metrics, dict):
                    issues.append(ValidationIssue(
                        message="quality_metrics must be an object",
                        path=f"columns[{i}].quality_metrics",
                        severity="error"
                    ))
                else:
                    for metric, metric_type in self.quality_metrics.items():
                        if metric not in metrics:
                            issues.append(ValidationIssue(
                                message=f"Missing quality metric: {metric}",
                                path=f"columns[{i}].quality_metrics",
                                severity="error"
                            ))
                        elif not isinstance(metrics[metric], metric_type):
                            issues.append(ValidationIssue(
                                message=f"Quality metric '{metric}' must be a number",
                                path=f"columns[{i}].quality_metrics.{metric}",
                                severity="error"
                            ))
                        elif not 0 <= metrics[metric] <= 1:
                            issues.append(ValidationIssue(
                                message=f"Quality metric '{metric}' must be between 0 and 1",
                                path=f"columns[{i}].quality_metrics.{metric}",
                                severity="error"
                            ))
                            
        # Try to fix common issues if any were found
        fixed_content = None
        if issues:
            try:
                fixed_content = self._fix_issues(content, issues)
            except Exception as e:
                logger.error(f"Error fixing schema issues: {str(e)}")
                
        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            fixed_content=fixed_content
        )
        
    def _fix_issues(self, content: Dict[str, Any], issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Attempt to fix common schema issues."""
        fixed = json.loads(json.dumps(content))  # Deep copy
        
        for issue in issues:
            if issue.severity != "error":
                continue
                
            # Fix missing required fields
            if "Missing required field" in issue.message:
                field = issue.message.split(": ")[1]
                if field in self.required_fields:
                    field_type = self.required_fields[field]
                    if field_type == str:
                        fixed[field] = ""
                    elif field_type == list:
                        fixed[field] = []
                    elif field_type == dict:
                        fixed[field] = {}
                        
            # Fix quality metrics
            elif "Missing quality metric" in issue.message:
                metric = issue.message.split(": ")[1]
                if "quality_metrics" not in fixed:
                    fixed["quality_metrics"] = {}
                fixed["quality_metrics"][metric] = 1.0
                
            # Fix invalid quality metric values
            elif "must be between 0 and 1" in issue.message:
                metric = issue.message.split("'")[1]
                if "quality_metrics" in fixed:
                    fixed["quality_metrics"][metric] = 1.0
                    
        return fixed 