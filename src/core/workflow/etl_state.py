from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field
from .base_state import BaseWorkflowState
from pydantic import BaseModel, Field


class ETLStep(str, Enum):
    ANALYSIS = "analysis"
    ARCHITECTURE_GENERATION = "architecture_generation"
    ARCHITECTURE_REVIEW = "architecture_review"
    ARCHITECTURE_CORRECTION = "architecture_correction"
    QUESTION_GENERATION = "question_generation"
    TERRAFORM_GENERATION = "terraform_generation"
    TERRAFORM_REVIEW = "terraform_review"
    TERRAFORM_CORRECTION = "terraform_correction"
    COMPLETED = "completed"


class DataSourceType(str, Enum):
    """Types of data sources."""
    REAL_TIME = "real_time"
    BATCH = "batch"
    HYBRID = "hybrid"


class DataFormat(str, Enum):
    """Data formats."""
    JSON = "json"
    PROTOBUF = "protobuf"
    AVRO = "avro"
    CSV = "csv"


class ProcessingType(str, Enum):
    """Types of data processing."""
    STREAMING = "streaming"
    BATCH = "batch"
    MICRO_BATCH = "micro_batch"


class ETLRequirements(BaseModel):
    """ETL requirements model."""
    data_sources: Dict[str, Any] = Field(
        default_factory=dict,
        description="Data source configurations"
    )
    processing_type: ProcessingType = Field(
        default=ProcessingType.STREAMING,
        description="Type of data processing required"
    )
    data_format: DataFormat = Field(
        default=DataFormat.JSON,
        description="Format of the data"
    )
    volume: Dict[str, Any] = Field(
        default_factory=dict,
        description="Data volume specifications"
    )
    latency_requirements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Latency requirements"
    )
    analytics_needs: List[str] = Field(
        default_factory=list,
        description="Analytics requirements"
    )
    missing_info: List[str] = Field(
        default_factory=list,
        description="Missing information that needs to be gathered"
    )
    next_question: str = Field(
        default="",
        description="Next question to ask the user"
    )
    summary: str = Field(
        default="",
        description="Summary of current requirements"
    )


@dataclass
class ETLArchitecture:
    components: Dict[str, Any]
    data_flow: List[Dict[str, Any]]
    services: List[str]
    improvements_applied: bool = False


@dataclass
class ETLTerraform:
    code: str
    improvements_applied: bool = False


class ETLWorkflowState(BaseModel):
    """State for ETL workflow."""
    workflow_id: str
    current_step: ETLStep = ETLStep.ANALYSIS
    requirements: Optional[ETLRequirements] = None
    architecture: Optional[ETLArchitecture] = None
    terraform_code: Optional[ETLTerraform] = None
    questions: List[str] = field(default_factory=list)
    answers: Dict[str, str] = field(default_factory=dict)
    review_feedback: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "initialized"
    error: Optional[str] = None

    def update_step(self, new_step: ETLStep) -> None:
        self.current_step = new_step

    def add_question(self, question: str) -> None:
        self.questions.append(question)

    def add_answer(self, question: str, answer: str) -> None:
        self.answers[question] = answer

    def set_requirements(self, requirements: ETLRequirements) -> None:
        self.requirements = requirements

    def set_architecture(self, architecture: ETLArchitecture) -> None:
        self.architecture = architecture

    def set_terraform(self, terraform: ETLTerraform) -> None:
        self.terraform_code = terraform

    def set_review_feedback(self, feedback: Dict[str, Any]) -> None:
        self.review_feedback = feedback


class ETLState(BaseWorkflowState):
    """State class for ETL (Extract, Transform, Load) workflows."""
    
    def __init__(
        self,
        source: str,
        destination: str,
        execution_id: str,
        extracted_data: Optional[list] = None,
        transformed_data: Optional[list] = None,
        loaded_records: int = 0,
        extraction_status: str = "pending",
        transformation_status: str = "pending",
        loading_status: str = "pending",
        extraction_errors: Optional[list] = None,
        transformation_errors: Optional[list] = None,
        loading_errors: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None
    ):
        super().__init__(execution_id, metadata, timestamp)
        self.source = source
        self.destination = destination
        self.extracted_data = extracted_data or []
        self.transformed_data = transformed_data or []
        self.loaded_records = loaded_records
        self.extraction_status = extraction_status
        self.transformation_status = transformation_status
        self.loading_status = loading_status
        self.extraction_errors = extraction_errors or []
        self.transformation_errors = transformation_errors or []
        self.loading_errors = loading_errors or []


def create_etl_state(
    source: str,
    destination: str,
    execution_id: str,
    **kwargs
) -> ETLState:
    """Create a new ETL state with default values."""
    state = ETLState(
        source=source,
        destination=destination,
        execution_id=execution_id,
        **kwargs
    )
    return state 