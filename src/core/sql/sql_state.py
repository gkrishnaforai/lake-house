from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from enum import Enum
import json
from pydantic import BaseModel, Field


class SQLGenerationStep(Enum):
    """Steps in the SQL generation process."""
    REQUIREMENTS_ANALYSIS = "requirements_analysis"
    SCHEMA_VALIDATION = "schema_validation"
    QUERY_GENERATION = "query_generation"
    QUERY_REVIEW = "query_review"
    QUERY_CORRECTION = "query_correction"


@dataclass
class SQLGenerationState:
    """State for SQL generation workflow."""
    workflow_id: str
    current_step: SQLGenerationStep = SQLGenerationStep.REQUIREMENTS_ANALYSIS
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    status: str = "pending"
    is_done: bool = False

    def update_step(self, step: SQLGenerationStep) -> None:
        """Update the current step."""
        self.current_step = step

    def set_error(self, error: str) -> None:
        """Set an error message."""
        self.error = error
        self.status = "error"

    def set_success(self) -> None:
        """Set success status."""
        self.status = "success"
        self.is_done = True

    def dict(self) -> Dict[str, Any]:
        """Convert the object to a dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "current_step": self.current_step.value,
            "metadata": self.metadata,
            "error": self.error,
            "status": self.status,
            "is_done": self.is_done
        }

    def to_json(self) -> str:
        """Convert the object to a JSON string."""
        return json.dumps(self.dict())

    @staticmethod
    def from_json(json_str: str) -> 'SQLGenerationState':
        """Create an instance from a JSON string."""
        data = json.loads(json_str)
        return SQLGenerationState(
            workflow_id=data["workflow_id"],
            current_step=SQLGenerationStep(data["current_step"]),
            metadata=data["metadata"],
            error=data.get("error"),
            status=data.get("status", "pending"),
            is_done=data.get("is_done", False)
        )


@dataclass
class SQLRequirements:
    """Requirements for SQL generation."""
    query: str
    schema: Dict[str, Any]
    constraints: Optional[Dict[str, Any]] = None
    missing_info: List[str] = field(default_factory=list)
    next_question: Optional[str] = None

    def to_json(self) -> str:
        """Convert the object to a JSON string."""
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(json_str: str) -> 'SQLRequirements':
        """Create an instance from a JSON string."""
        data = json.loads(json_str)
        return SQLRequirements(**data)

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return self.to_json()


class SQLGenerationOutput(BaseModel):
    """Output model for SQL generation."""
    sql_query: str
    explanation: str | None = None
    confidence: float = 1.0
    tables_used: list[str] = Field(default_factory=list)
    columns_used: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        return self.model_dump_json()

    @staticmethod
    def from_json(json_str: str) -> 'SQLGenerationOutput':
        return SQLGenerationOutput.model_validate_json(json_str)

    def __str__(self) -> str:
        return self.to_json()


class SQLGenerationError(Exception):
    """Custom exception for SQL generation errors."""
    pass


class TransformationOutput(BaseModel):
    """Output model for data transformation."""
    transformed_data: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    explanation: Optional[str] = None
    confidence: Optional[float] = None

    def to_json(self) -> str:
        return self.model_dump_json()

    @staticmethod
    def from_json(json_str: str) -> 'TransformationOutput':
        return TransformationOutput.model_validate_json(json_str)

    def __str__(self) -> str:
        return self.to_json()
