"""Models for data transformations."""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class TransformationConfig(BaseModel):
    """Configuration for a data transformation."""
    source_columns: List[str]
    transformation_type: str
    new_column_name: Optional[str] = None
    data_type: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    is_template: Optional[bool] = False
    template_name: Optional[str] = None


class TransformationTemplate(BaseModel):
    """Template for a data transformation."""
    name: str
    description: str
    transformation_type: str
    default_parameters: Dict[str, Any]
    prompt_template: Optional[str] = None
    example_input: Optional[Dict[str, Any]] = None
    example_output: Optional[Dict[str, Any]] = None


class TransformationResult(BaseModel):
    """Result of a data transformation."""
    status: str
    message: Optional[str] = None
    new_columns: List[Dict[str, str]] = []
    preview_data: Optional[List[Dict[str, Any]]] = None
    errors: Optional[List[str]] = None 