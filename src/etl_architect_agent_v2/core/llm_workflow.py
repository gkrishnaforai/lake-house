"""LLM workflow validation and regeneration using LangGraph."""

from typing import Dict, Any, List, Optional, Callable, Union, Type, TypeVar
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, ValidationError
import logging
import json
from datetime import datetime
import re
import traceback
from enum import Enum
import inspect
from functools import wraps
import unittest
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class SchemaExtractor:
    """Utility class to extract schema information from data."""
    
    @staticmethod
    def get_pandas_dtype_mapping() -> Dict[str, str]:
        """Map pandas dtypes to our schema types."""
        return {
            'object': 'string',
            'string': 'string',
            'int64': 'integer',
            'float64': 'float',
            'bool': 'boolean',
            'datetime64[ns]': 'datetime',
            'category': 'string'
        }
        
    @staticmethod
    def extract_schema(df: pd.DataFrame, sample_size: int = 5) -> Dict[str, Any]:
        """Extract schema information from a pandas DataFrame."""
        schema = {"columns": []}
        sample_data = df.head(sample_size)
        
        for column in df.columns:
            col_info = {
                "name": str(column),
                "type": SchemaExtractor.get_pandas_dtype_mapping().get(
                    str(df[column].dtype), 'string'
                ),
                "description": "",  # Will be filled by LLM
                "sample_value": str(sample_data[column].iloc[0]) if not sample_data[column].empty else "",
                "quality_metrics": {
                    "completeness": float(1 - df[column].isna().mean()),
                    "uniqueness": float(1 - (df[column].nunique() / len(df))),
                    "validity": 1.0  # Will be refined by LLM
                }
            }
            schema["columns"].append(col_info)
            
        return schema
        
    @staticmethod
    def create_optimized_prompt(df: pd.DataFrame, sample_size: int = 5) -> str:
        """Create an optimized prompt for schema generation."""
        schema = SchemaExtractor.extract_schema(df, sample_size)
        sample_data = df.head(sample_size).to_dict('records')
        
        prompt = f"""Please enhance the following schema with better descriptions and validation rules.
The schema was automatically extracted from the data.

Current Schema:
{json.dumps(schema, indent=2)}

Sample Data (first {sample_size} rows):
{json.dumps(sample_data, indent=2)}

Please provide:
1. Better column descriptions
2. Validation rules if any
3. Suggested data quality improvements
4. Any business rules you can infer

Return the enhanced schema in the same format.
"""
        return prompt


class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationIssue(BaseModel):
    """A validation issue with severity level."""
    message: str
    level: ValidationLevel
    field: Optional[str] = None
    fix_suggestion: Optional[str] = None


class ValidationResult(BaseModel):
    """Result of LLM output validation."""
    is_valid: bool = Field(..., description="Whether the validation passed")
    issues: List[ValidationIssue] = Field(default_factory=list, description="Validation issues")
    fixed_content: Optional[Dict[str, Any]] = Field(None, description="Fixed content if applicable")
    attempts: int = Field(default=1, description="Number of regeneration attempts")
    validation_time: datetime = Field(default_factory=datetime.utcnow)
    validation_type: str = Field(..., description="Type of validation performed")
    fixes_applied: List[str] = Field(default_factory=list, description="List of fixes applied")
    error_category: Optional[str] = Field(None, description="Category of error if any")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Detailed error information")


class LLMWorkflowState(BaseModel):
    """State for LLM workflow."""
    content: Dict[str, Any] = Field(..., description="LLM generated content")
    validation_result: Optional[ValidationResult] = Field(None, description="Validation result")
    max_attempts: int = Field(default=3, description="Maximum regeneration attempts")
    current_attempt: int = Field(default=1, description="Current attempt number")
    error: Optional[str] = Field(None, description="Error message if any")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    workflow_history: List[Dict[str, Any]] = Field(default_factory=list, description="Workflow history")
    attempt_history: List[Dict[str, Any]] = Field(default_factory=list, description="History of all attempts")
    fixes_applied: List[str] = Field(default_factory=list, description="List of fixes applied")
    llm_used: bool = Field(default=False, description="Whether LLM was used for fixes")
    final_status: Optional[str] = Field(None, description="Final status of the workflow")


class BaseValidator:
    """Base class for all validators."""
    
    def __init__(self, validation_type: str):
        self.validation_type = validation_type
        
    def validate(self, content: Dict[str, Any]) -> ValidationResult:
        """Validate content and return result."""
        raise NotImplementedError
        
    def fix(self, content: Dict[str, Any], issues: List[ValidationIssue]) -> Optional[Dict[str, Any]]:
        """Attempt to fix content based on validation issues."""
        raise NotImplementedError
        
    def _create_issue(
        self,
        message: str,
        level: ValidationLevel,
        field: Optional[str] = None,
        fix_suggestion: Optional[str] = None
    ) -> ValidationIssue:
        """Create a validation issue."""
        return ValidationIssue(
            message=message,
            level=level,
            field=field,
            fix_suggestion=fix_suggestion
        )


class SchemaValidator(BaseValidator):
    """Validator for schema generation."""
    
    def __init__(self):
        super().__init__("schema")
        
    def validate(self, content: Dict[str, Any]) -> ValidationResult:
        """Validate generated schema."""
        issues = []
        
        # Check basic structure
        if not isinstance(content, dict):
            issues.append(self._create_issue(
                "Schema must be a dictionary",
                ValidationLevel.ERROR
            ))
            return ValidationResult(
                is_valid=False,
                issues=issues,
                validation_type=self.validation_type
            )
            
        if "columns" not in content:
            issues.append(self._create_issue(
                "Schema must contain 'columns' key",
                ValidationLevel.ERROR
            ))
            return ValidationResult(
                is_valid=False,
                issues=issues,
                validation_type=self.validation_type
            )
            
        if not isinstance(content["columns"], list):
            issues.append(self._create_issue(
                "'columns' must be a list",
                ValidationLevel.ERROR
            ))
            return ValidationResult(
                is_valid=False,
                issues=issues,
                validation_type=self.validation_type
            )
            
        # Validate each column
        for i, col in enumerate(content["columns"]):
            if not isinstance(col, dict):
                issues.append(self._create_issue(
                    f"Column {i} must be a dictionary",
                    ValidationLevel.ERROR,
                    field=f"columns[{i}]"
                ))
                continue
                
            # Check required fields
            for field in ["name", "type", "description"]:
                if field not in col:
                    issues.append(self._create_issue(
                        f"Column {i} missing required field: {field}",
                        ValidationLevel.ERROR,
                        field=f"columns[{i}].{field}"
                    ))
                elif not isinstance(col[field], str):
                    issues.append(self._create_issue(
                        f"Column {i} field {field} must be a string",
                        ValidationLevel.ERROR,
                        field=f"columns[{i}].{field}",
                        fix_suggestion=f"Convert {field} to string"
                    ))
                    
            # Validate quality metrics
            if "quality_metrics" in col:
                metrics = col["quality_metrics"]
                if not isinstance(metrics, dict):
                    issues.append(self._create_issue(
                        f"Column {i} quality_metrics must be a dictionary",
                        ValidationLevel.ERROR,
                        field=f"columns[{i}].quality_metrics"
                    ))
                else:
                    for metric in ["completeness", "uniqueness", "validity"]:
                        if metric not in metrics:
                            issues.append(self._create_issue(
                                f"Column {i} missing quality metric: {metric}",
                                ValidationLevel.WARNING,
                                field=f"columns[{i}].quality_metrics.{metric}",
                                fix_suggestion=f"Add {metric} metric with default value 1.0"
                            ))
                        elif not isinstance(metrics[metric], (int, float)):
                            issues.append(self._create_issue(
                                f"Column {i} quality metric {metric} must be a number",
                                ValidationLevel.ERROR,
                                field=f"columns[{i}].quality_metrics.{metric}",
                                fix_suggestion=f"Convert {metric} to number"
                            ))
                        elif not 0 <= metrics[metric] <= 1:
                            issues.append(self._create_issue(
                                f"Column {i} quality metric {metric} must be between 0 and 1",
                                ValidationLevel.ERROR,
                                field=f"columns[{i}].quality_metrics.{metric}",
                                fix_suggestion=f"Clamp {metric} between 0 and 1"
                            ))
                            
        # Try to fix issues
        fixed_content = None
        if issues:
            try:
                fixed_content = self.fix(content, issues)
            except Exception as e:
                logger.error(f"Error fixing schema: {str(e)}")
                issues.append(self._create_issue(
                    f"Failed to fix schema: {str(e)}",
                    ValidationLevel.CRITICAL
                ))
                
        return ValidationResult(
            is_valid=len([i for i in issues if i.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]]) == 0,
            issues=issues,
            fixed_content=fixed_content,
            validation_type=self.validation_type
        )
        
    def fix(self, content: Dict[str, Any], issues: List[ValidationIssue]) -> Optional[Dict[str, Any]]:
        """Fix schema issues."""
        fixed = content.copy()
        
        # Ensure columns is a list
        if not isinstance(fixed.get("columns"), list):
            fixed["columns"] = []
            
        # Fix each column
        for i, col in enumerate(fixed["columns"]):
            # Ensure required fields are strings
            for field in ["name", "type", "description"]:
                if field not in col:
                    col[field] = ""
                elif not isinstance(col[field], str):
                    col[field] = str(col[field])
                    
            # Fix quality metrics
            if "quality_metrics" not in col:
                col["quality_metrics"] = {
                    "completeness": 1.0,
                    "uniqueness": 1.0,
                    "validity": 1.0
                }
            else:
                metrics = col["quality_metrics"]
                for metric in ["completeness", "uniqueness", "validity"]:
                    if metric not in metrics:
                        metrics[metric] = 1.0
                    elif not isinstance(metrics[metric], (int, float)):
                        metrics[metric] = 1.0
                    elif not 0 <= metrics[metric] <= 1:
                        metrics[metric] = max(0.0, min(1.0, float(metrics[metric])))
                        
        return fixed


class JSONValidator(BaseValidator):
    """Validator for JSON content."""
    
    def __init__(self):
        super().__init__("json")
        self.error_categories = {
            "syntax": [
                "unterminated string",
                "missing quotes",
                "invalid escape sequence",
                "unexpected character"
            ],
            "structure": [
                "missing comma",
                "invalid nesting",
                "unclosed bracket",
                "trailing comma"
            ],
            "format": [
                "invalid number",
                "invalid boolean",
                "invalid null",
                "invalid array",
                "invalid object"
            ],
            "type": [
                "must be a string",
                "must be a number",
                "must be a boolean",
                "must be an array",
                "must be an object"
            ],
            "complex": [
                "nested object inconsistency",
                "complex array formatting",
                "mixed type issues",
                "schema violation"
            ]
        }
        
        # Define required field types
        self.required_types = {
            "sample_value": str,
            "name": str,
            "type": str,
            "description": str
        }
        
        # Custom JSON encoder for datetime objects
        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return super().default(obj)
                
        self.json_encoder = DateTimeEncoder
        
    def _serialize_datetime(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Convert datetime objects to ISO format strings."""
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return obj
            
        return convert(content)
        
    def _categorize_error(self, error_msg: str) -> str:
        """Categorize JSON error into a specific type."""
        error_msg = error_msg.lower()
        for category, patterns in self.error_categories.items():
            if any(pattern in error_msg for pattern in patterns):
                return category
        return "complex"
        
    def _validate_field_types(self, content: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate field types in the content."""
        issues = []
        
        def check_field_types(obj: Dict[str, Any], path: str = "") -> None:
            for field, expected_type in self.required_types.items():
                if field in obj:
                    value = obj[field]
                    if not isinstance(value, expected_type):
                        issues.append(self._create_issue(
                            f"Field {field} must be a {expected_type.__name__}",
                            ValidationLevel.ERROR,
                            field=f"{path}.{field}" if path else field,
                            fix_suggestion=f"Convert {field} to {expected_type.__name__}"
                        ))
            
            # Check nested objects
            for key, value in obj.items():
                if isinstance(value, dict):
                    new_path = f"{path}.{key}" if path else key
                    check_field_types(value, new_path)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            new_path = f"{path}.{key}[{i}]" if path else f"{key}[{i}]"
                            check_field_types(item, new_path)
        
        check_field_types(content)
        return issues
        
    def validate(self, content: Union[str, Dict[str, Any]]) -> ValidationResult:
        """Validate JSON content."""
        issues = []
        fixes_applied = []
        
        # If content is already a dict, validate field types
        if isinstance(content, dict):
            # Convert datetime objects to strings
            content = self._serialize_datetime(content)
            type_issues = self._validate_field_types(content)
            issues.extend(type_issues)
            return ValidationResult(
                is_valid=len(issues) == 0,
                issues=issues,
                validation_type=self.validation_type,
                fixes_applied=fixes_applied
            )
            
        # If content is a string, try to parse it
        if isinstance(content, str):
            try:
                parsed_content = json.loads(content)
                # Convert datetime objects to strings
                parsed_content = self._serialize_datetime(parsed_content)
                type_issues = self._validate_field_types(parsed_content)
                issues.extend(type_issues)
                return ValidationResult(
                    is_valid=len(issues) == 0,
                    issues=issues,
                    validation_type=self.validation_type,
                    fixes_applied=fixes_applied
                )
            except json.JSONDecodeError as e:
                error_category = self._categorize_error(str(e))
                issues.append(self._create_issue(
                    f"Invalid JSON: {str(e)}",
                    ValidationLevel.ERROR
                ))
                
                # Try to fix common JSON issues
                try:
                    fixed_content = self.fix(content, issues)
                    if fixed_content:
                        # Convert datetime objects to strings
                        fixed_content = self._serialize_datetime(fixed_content)
                        # Validate field types in fixed content
                        type_issues = self._validate_field_types(fixed_content)
                        issues.extend(type_issues)
                        return ValidationResult(
                            is_valid=len(issues) == 0,
                            issues=issues,
                            fixed_content=fixed_content,
                            validation_type=self.validation_type,
                            fixes_applied=fixes_applied,
                            error_category=error_category,
                            error_details={
                                "error": str(e),
                                "line": e.lineno,
                                "column": e.colno,
                                "pos": e.pos
                            }
                        )
                except Exception as fix_error:
                    issues.append(self._create_issue(
                        f"Failed to fix JSON: {str(fix_error)}",
                        ValidationLevel.CRITICAL
                    ))
                    
        return ValidationResult(
            is_valid=False,
            issues=issues,
            validation_type=self.validation_type,
            error_category="complex",
            error_details={"error": "Failed to validate or fix JSON"}
        )
        
    def _fix_field_types(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Fix field types in the content."""
        fixed = content.copy()
        
        def fix_field_types(obj: Dict[str, Any]) -> None:
            for field, expected_type in self.required_types.items():
                if field in obj:
                    value = obj[field]
                    if not isinstance(value, expected_type):
                        try:
                            if expected_type == str:
                                obj[field] = str(value)
                            elif expected_type == int:
                                obj[field] = int(float(value))
                            elif expected_type == float:
                                obj[field] = float(value)
                            elif expected_type == bool:
                                obj[field] = bool(value)
                        except (ValueError, TypeError):
                            # If conversion fails, set a default value
                            if expected_type == str:
                                obj[field] = ""
                            elif expected_type == int:
                                obj[field] = 0
                            elif expected_type == float:
                                obj[field] = 0.0
                            elif expected_type == bool:
                                obj[field] = False
            
            # Fix nested objects
            for key, value in obj.items():
                if isinstance(value, dict):
                    fix_field_types(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            fix_field_types(item)
        
        fix_field_types(fixed)
        return fixed
        
    def fix(self, content: str, issues: List[ValidationIssue]) -> Optional[Dict[str, Any]]:
        """Fix common JSON issues."""
        if not isinstance(content, str):
            return None
            
        fixes_applied = []
        
        # Remove any markdown code block formatting
        content = self._remove_markdown_formatting(content)
        fixes_applied.append("removed_markdown_formatting")
        
        # Fix common JSON issues in priority order
        content = self._fix_unterminated_strings(content)
        fixes_applied.append("fixed_unterminated_strings")
        
        content = self._fix_trailing_commas(content)
        fixes_applied.append("fixed_trailing_commas")
        
        content = self._fix_missing_quotes(content)
        fixes_applied.append("fixed_missing_quotes")
        
        content = self._fix_boolean_values(content)
        fixes_applied.append("fixed_boolean_values")
        
        content = self._fix_numeric_values(content)
        fixes_applied.append("fixed_numeric_values")
        
        content = self._fix_null_values(content)
        fixes_applied.append("fixed_null_values")
        
        content = self._fix_array_formatting(content)
        fixes_applied.append("fixed_array_formatting")
        
        content = self._fix_object_formatting(content)
        fixes_applied.append("fixed_object_formatting")
        
        content = self._fix_incomplete_objects(content)
        fixes_applied.append("fixed_incomplete_objects")
        
        try:
            parsed_content = json.loads(content)
            # Fix field types
            fixed_content = self._fix_field_types(parsed_content)
            # Convert datetime objects to strings
            fixed_content = self._serialize_datetime(fixed_content)
            fixes_applied.append("fixed_field_types")
            return fixed_content
        except json.JSONDecodeError:
            return None
            
    def _remove_markdown_formatting(self, content: str) -> str:
        """Remove markdown code block formatting."""
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()
            
    def _fix_unterminated_strings(self, content: str) -> str:
        """Fix unterminated strings in JSON content."""
        # Find all string literals
        string_pattern = r'"([^"\\]*(?:\\.[^"\\]*)*)"'
        strings = re.finditer(string_pattern, content)
        
        # Check for unterminated strings
        fixed_content = content
        for match in strings:
            start, end = match.span()
            # If the string is unterminated, add a closing quote
            if end < len(content) and content[end] != '"':
                fixed_content = (
                    fixed_content[:end] + '"' + fixed_content[end:]
                )
                
        return fixed_content
        
    def _fix_trailing_commas(self, content: str) -> str:
        """Fix trailing commas in objects and arrays."""
        # Remove trailing commas in objects
        content = re.sub(r',\s*}', '}', content)
        # Remove trailing commas in arrays
        content = re.sub(r',\s*]', ']', content)
        return content
        
    def _fix_missing_quotes(self, content: str) -> str:
        """Fix missing quotes around property names and string values."""
        # Fix unquoted property names
        content = re.sub(
            r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:',
            r'\1"\2":',
            content
        )
        
        # Fix unquoted string values
        content = re.sub(
            r':\s*([a-zA-Z_][a-zA-Z0-9_]*)([,}])',
            r':"\1"\2',
            content
        )
        
        # Fix missing quotes around string values
        content = re.sub(
            r':\s*([^"{\[\d][^,}]*?)([,}])',
            r':"\1"\2',
            content
        )
        
        return content
        
    def _fix_boolean_values(self, content: str) -> str:
        """Fix boolean values (true/false)."""
        # Fix lowercase true/false
        content = re.sub(r':\s*true\b', ': true', content)
        content = re.sub(r':\s*false\b', ': false', content)
        # Fix uppercase TRUE/FALSE
        content = re.sub(r':\s*TRUE\b', ': true', content)
        content = re.sub(r':\s*FALSE\b', ': false', content)
        return content
        
    def _fix_numeric_values(self, content: str) -> str:
        """Fix numeric values."""
        # Fix leading zeros in numbers
        content = re.sub(r':\s*0+(\d+)', r':\1', content)
        # Fix decimal points without numbers
        content = re.sub(r':\s*\.', ': 0.', content)
        # Fix scientific notation
        content = re.sub(r':\s*(\d+)e(\d+)', r':\1e\2', content)
        return content
        
    def _fix_null_values(self, content: str) -> str:
        """Fix null values."""
        # Fix lowercase null
        content = re.sub(r':\s*null\b', ': null', content)
        # Fix uppercase NULL
        content = re.sub(r':\s*NULL\b', ': null', content)
        # Fix empty values as null
        content = re.sub(r':\s*""', ': null', content)
        return content
        
    def _fix_array_formatting(self, content: str) -> str:
        """Fix array formatting."""
        # Fix missing commas between array elements
        content = re.sub(r'}\s*{', '}, {', content)
        content = re.sub(r'\]\s*\[', '], [', content)
        # Fix empty arrays
        content = re.sub(r'\[\s*\]', '[]', content)
        return content
        
    def _fix_object_formatting(self, content: str) -> str:
        """Fix object formatting."""
        # Fix missing commas between object properties
        content = re.sub(r'"\s*"', '", "', content)
        # Fix empty objects
        content = re.sub(r'{\s*}', '{}', content)
        # Fix missing colons
        content = re.sub(r'"\s*"', '": "', content)
        return content
        
    def _fix_incomplete_objects(self, content: str) -> str:
        """Fix incomplete JSON objects with missing property values."""
        # Find property names followed by newline or end of object
        pattern = r'"([^"]+)"\s*:\s*(?=\n|}|$)'
        
        def replace_incomplete(match):
            prop_name = match.group(1)
            # Add null value for missing property
            return f'"{prop_name}": null'
            
        # Replace incomplete properties with null values
        content = re.sub(pattern, replace_incomplete, content)
        
        # Fix missing values in quality metrics
        quality_metrics_pattern = r'"quality_metrics"\s*:\s*{\s*"([^"]+)"\s*:(?=\n|}|$)'
        content = re.sub(
            quality_metrics_pattern,
            lambda m: f'"quality_metrics": {{"{m.group(1)}": 1.0',
            content
        )
        
        return content


class TestJSONValidator(unittest.TestCase):
    """Test cases for JSONValidator."""
    
    def setUp(self):
        self.validator = JSONValidator()
        
    def test_unterminated_string(self):
        """Test fixing unterminated strings."""
        content = '''{
            "name": "Total Funding Amount",
            "type": "float",
            "description": "Total funding amount
        }'''
        
        result = self.validator.validate(content)
        self.assertFalse(result.is_valid)
        
        fixed = self.validator.fix(content, result.issues)
        self.assertIsNotNone(fixed)
        self.assertEqual(fixed["description"], "Total funding amount")
        
    def test_missing_quotes(self):
        """Test fixing missing quotes."""
        content = '''{
            name: "Test",
            value: 123
        }'''
        
        result = self.validator.validate(content)
        self.assertFalse(result.is_valid)
        
        fixed = self.validator.fix(content, result.issues)
        self.assertIsNotNone(fixed)
        self.assertEqual(fixed["name"], "Test")
        
    def test_trailing_comma(self):
        """Test fixing trailing commas."""
        content = '''{
            "name": "Test",
            "value": 123,
        }'''
        
        result = self.validator.validate(content)
        self.assertFalse(result.is_valid)
        
        fixed = self.validator.fix(content, result.issues)
        self.assertIsNotNone(fixed)
        self.assertEqual(fixed["value"], 123)
        
    def test_complex_schema(self):
        """Test fixing complex schema with multiple issues."""
        content = '''{
            "columns": [
                {
                    name: "Last Funding Type",
                    type: "string",
                    description: "Type of the last funding",
                    sample_value: "Seed",
                    quality_metrics: {
                        completeness: 1.0,
                        uniqueness: 1.0,
                        validity: 1.0,
                    }
                },
                {
                    name: "Total Funding Amount",
                    type: "float",
                    description: "Total funding amount
                }
            ]
        }'''
        
        result = self.validator.validate(content)
        self.assertFalse(result.is_valid)
        
        fixed = self.validator.fix(content, result.issues)
        self.assertIsNotNone(fixed)
        self.assertEqual(len(fixed["columns"]), 2)
        self.assertEqual(fixed["columns"][0]["name"], "Last Funding Type")
        self.assertEqual(fixed["columns"][1]["type"], "float")


class PydanticValidator(BaseValidator):
    """Validator for Pydantic models."""
    
    def __init__(self, model_class: Type[T]):
        super().__init__("pydantic")
        self.model_class = model_class
        
    def validate(self, content: Dict[str, Any]) -> ValidationResult:
        """Validate content against Pydantic model."""
        issues = []
        
        try:
            self.model_class(**content)
            return ValidationResult(
                is_valid=True,
                issues=issues,
                validation_type=self.validation_type
            )
        except ValidationError as e:
            for error in e.errors():
                issues.append(self._create_issue(
                    error["msg"],
                    ValidationLevel.ERROR,
                    field=".".join(str(x) for x in error["loc"]),
                    fix_suggestion=error.get("ctx", {}).get("fix_suggestion")
                ))
                
        return ValidationResult(
            is_valid=False,
            issues=issues,
            validation_type=self.validation_type
        )
        
    def fix(self, content: Dict[str, Any], issues: List[ValidationIssue]) -> Optional[Dict[str, Any]]:
        """Attempt to fix Pydantic validation issues."""
        # This is a placeholder - Pydantic validation fixes would be model-specific
        return None


class LLMWorkflow:
    """Base workflow for LLM generation with validation and regeneration."""

    def __init__(
        self,
        validator: BaseValidator,
        generator: Callable[[Dict[str, Any]], Dict[str, Any]],
        max_attempts: int = 3
    ):
        """Initialize the workflow."""
        self.validator = validator
        self.generator = generator
        self.max_attempts = max_attempts
        self.graph = self._build_graph()

    def _build_graph(self) -> Graph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(LLMWorkflowState)

        # Add nodes
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("validate", self._validate_node)
        workflow.add_node("fix", self._fix_node)
        workflow.add_node("llm_fix", self._llm_fix_node)
        workflow.add_node("end", lambda x: x)  # End node that returns state as is

        # Add edges
        workflow.add_edge("generate", "validate")
        workflow.add_conditional_edges(
            "validate",
            self._should_regenerate,
            {
                "continue": "generate",
                "fix": "fix",
                "llm_fix": "llm_fix",
                "complete": "end"
            }
        )
        workflow.add_edge("fix", "validate")
        workflow.add_edge("llm_fix", "validate")

        # Set entry point
        workflow.set_entry_point("generate")

        return workflow.compile()

    async def _generate_node(self, state: LLMWorkflowState) -> LLMWorkflowState:
        """Generate content using LLM."""
        try:
            logger.info(f"Generating content (attempt {state.current_attempt})")
            state.content = await self.generator(state.metadata)
            state.workflow_history.append({
                "step": "generate",
                "attempt": state.current_attempt,
                "timestamp": datetime.utcnow().isoformat()
            })
            return state
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            logger.error(traceback.format_exc())
            state.error = str(e)
            state.workflow_history.append({
                "step": "generate",
                "attempt": state.current_attempt,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            return state

    async def _validate_node(self, state: LLMWorkflowState) -> LLMWorkflowState:
        """Validate generated content."""
        try:
            logger.info("Validating content")
            state.validation_result = self.validator.validate(state.content)
            state.workflow_history.append({
                "step": "validate",
                "attempt": state.current_attempt,
                "is_valid": state.validation_result.is_valid,
                "issues": [i.dict() for i in state.validation_result.issues],
                "timestamp": datetime.utcnow().isoformat()
            })
            return state
        except Exception as e:
            logger.error(f"Error validating content: {str(e)}")
            logger.error(traceback.format_exc())
            state.error = str(e)
            state.workflow_history.append({
                "step": "validate",
                "attempt": state.current_attempt,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            return state

    async def _fix_node(self, state: LLMWorkflowState) -> LLMWorkflowState:
        """Attempt to fix invalid content."""
        try:
            if state.validation_result and state.validation_result.fixed_content:
                logger.info("Applying fixes to content")
                state.content = state.validation_result.fixed_content
                state.fixes_applied.extend(state.validation_result.fixes_applied)
                state.workflow_history.append({
                    "step": "fix",
                    "attempt": state.current_attempt,
                    "fixes_applied": state.validation_result.fixes_applied,
                    "timestamp": datetime.utcnow().isoformat()
                })
            return state
        except Exception as e:
            logger.error(f"Error fixing content: {str(e)}")
            logger.error(traceback.format_exc())
            state.error = str(e)
            state.workflow_history.append({
                "step": "fix",
                "attempt": state.current_attempt,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            return state

    async def _llm_fix_node(self, state: LLMWorkflowState) -> LLMWorkflowState:
        """Use LLM to fix complex issues."""
        try:
            if state.validation_result:
                logger.info("Using LLM to fix complex issues")
                # Create detailed prompt for LLM
                prompt = self._create_llm_prompt(state)
                # Get LLM response
                state.content = await self.generator({
                    "prompt": prompt,
                    "original_content": state.content,
                    "validation_result": state.validation_result.dict()
                })
                state.llm_used = True
                state.workflow_history.append({
                    "step": "llm_fix",
                    "attempt": state.current_attempt,
                    "timestamp": datetime.utcnow().isoformat()
                })
            return state
        except Exception as e:
            logger.error(f"Error in LLM fix: {str(e)}")
            logger.error(traceback.format_exc())
            state.error = str(e)
            state.workflow_history.append({
                "step": "llm_fix",
                "attempt": state.current_attempt,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            return state

    def _create_llm_prompt(self, state: LLMWorkflowState) -> str:
        """Create detailed prompt for LLM fix."""
        prompt = f"""Please fix the following JSON issues:

Original JSON:
{json.dumps(state.content, indent=2)}

Validation Issues:
"""
        for issue in state.validation_result.issues:
            prompt += f"- {issue.message}\n"
            if issue.fix_suggestion:
                prompt += f"  Suggestion: {issue.fix_suggestion}\n"

        prompt += f"""
Error Category: {state.validation_result.error_category}
Error Details: {json.dumps(state.validation_result.error_details, indent=2)}

Fixes Already Attempted:
{json.dumps(state.fixes_applied, indent=2)}

Please provide a valid JSON that fixes all these issues while maintaining the original structure.
"""
        return prompt

    def _should_regenerate(self, state: LLMWorkflowState) -> str:
        """Determine if content should be regenerated."""
        if state.error:
            return "complete"  # Stop on error
        
        if not state.validation_result:
            return "complete"  # Stop if no validation result
            
        if state.validation_result.is_valid:
            state.final_status = "success"
            return "complete"  # Stop if valid
            
        if state.current_attempt >= state.max_attempts:
            state.final_status = "failed_after_max_attempts"
            return "complete"  # Stop if max attempts reached
            
        # Try automatic fixes first
        if state.validation_result.fixes_applied:
            return "fix"
            
        # If automatic fixes didn't work, try LLM
        if state.validation_result.error_category == "complex":
            return "llm_fix"
            
        # If we've tried LLM and it didn't work, try again
        if state.llm_used:
            state.current_attempt += 1
            return "generate"
            
        return "complete"  # Stop if no fix available

    async def run(self, metadata: Dict[str, Any]) -> LLMWorkflowState:
        """Run the workflow."""
        initial_state = LLMWorkflowState(
            content={},
            max_attempts=self.max_attempts,
            metadata=metadata
        )
        
        try:
            # Run the workflow and convert result to LLMWorkflowState
            result = await self.graph.ainvoke(initial_state)
            if isinstance(result, dict):
                result = LLMWorkflowState(**result)
                
            if not result.validation_result:
                result.validation_result = ValidationResult(
                    is_valid=False,
                    issues=[
                        ValidationIssue(
                            message="Workflow failed to produce validation result",
                            level=ValidationLevel.ERROR
                        )
                    ],
                    validation_type=self.validator.validation_type
                )
                
            # Record attempt history
            result.attempt_history = [
                {
                    "attempt": i + 1,
                    "content": step.get("content"),
                    "validation_result": step.get("validation_result"),
                    "fixes_applied": step.get("fixes_applied", []),
                    "llm_used": step.get("llm_used", False)
                }
                for i, step in enumerate(result.workflow_history)
            ]
            
            return result
        except Exception as e:
            logger.error(f"Error running workflow: {str(e)}")
            logger.error(traceback.format_exc())
            initial_state.error = str(e)
            initial_state.validation_result = ValidationResult(
                is_valid=False,
                issues=[
                    ValidationIssue(
                        message=f"Workflow error: {str(e)}",
                        level=ValidationLevel.ERROR
                    )
                ],
                validation_type=self.validator.validation_type
            )
            initial_state.workflow_history.append({
                "step": "workflow",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            return initial_state


def validate_llm_output(validator: BaseValidator):
    """Decorator to validate LLM output."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            workflow = LLMWorkflow(validator, func)
            result = await workflow.run(kwargs.get('metadata', {}))
            if not result.validation_result.is_valid:
                raise ValueError(
                    f"LLM output validation failed: {result.validation_result.issues}"
                )
            return result.content
        return wrapper
    return decorator 


if __name__ == "__main__":
    unittest.main() 