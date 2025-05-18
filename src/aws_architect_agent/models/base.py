from enum import Enum
from typing import Dict, List, Optional, Union, Annotated
from typing_extensions import TypedDict
from operator import add
from pydantic import BaseModel, Field

class SolutionType(str, Enum):
    ETL = "etl"
    SERVERLESS = "serverless"
    CONTAINER = "container"
    ML = "ml"
    IOT = "iot"
    SECURITY = "security"

class ArchitectureState(str, Enum):
    INITIAL = "initial"
    REQUIREMENTS_GATHERING = "requirements_gathering"
    DESIGN = "design"
    REVIEW = "review"
    REFINEMENT = "refinement"
    FINALIZED = "finalized"

class ComponentType(str, Enum):
    DATA_SOURCE = "data_source"
    PROCESSING = "processing"
    STORAGE = "storage"
    NETWORKING = "networking"
    SECURITY = "security"
    MONITORING = "monitoring"

class Component(BaseModel):
    id: str = Field(..., description="Unique identifier for the component")
    type: ComponentType = Field(..., description="Type of the component")
    name: str = Field(..., description="Name of the component")
    description: str = Field(..., description="Description of the component")
    configuration: Dict[str, Union[str, int, float, bool, List[str]]] = Field(
        default_factory=dict,
        description="Component-specific configuration"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of component IDs this component depends on"
    )

class Architecture(BaseModel):
    id: str = Field(..., description="Unique identifier for the architecture")
    name: str = Field(..., description="Name of the architecture")
    description: str = Field(..., description="Description of the architecture")
    solution_type: SolutionType = Field(..., description="Type of solution")
    components: List[Component] = Field(
        default_factory=list,
        description="List of components in the architecture"
    )
    version: int = Field(default=1, description="Version number of the architecture")
    state: ArchitectureState = Field(
        default=ArchitectureState.INITIAL,
        description="Current state of the architecture"
    )
    requirements: Dict[str, str] = Field(
        default_factory=dict,
        description="Architecture requirements"
    )
    feedback: List[str] = Field(
        default_factory=list,
        description="List of feedback items"
    )

class ConversationState(TypedDict):
    current_architecture: Optional[Architecture]
    conversation_history: Annotated[List[Dict[str, str]], add]
    current_state: ArchitectureState
    requirements: Dict[str, str]
    feedback: List[str] 