"""State management for the ETL Architect Agent V2."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Annotated
from datetime import datetime
from enum import Enum
from langgraph.graph.message import add_messages

class AgentRole(Enum):
    """Enum representing different agent roles in the system."""
    REQUIREMENTS = "requirements"
    ARCHITECTURE = "architecture"
    VALIDATION = "validation"
    IMPLEMENTATION = "implementation"

class MessageRole(Enum):
    """Enum representing different message roles in conversations."""
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"

@dataclass
class Message:
    """Represents a message in the conversation."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConversationContext:
    """Manages the conversation state and context."""
    messages: Annotated[List[Message], add_messages] = field(default_factory=list)
    current_topic: str = ""
    missing_info: List[str] = field(default_factory=list)
    next_question: Optional[str] = None

@dataclass
class RequirementsState:
    """Stores the requirements gathered during the conversation."""
    data_sources: Dict[str, Any] = field(default_factory=dict)
    data_volume: Dict[str, Any] = field(default_factory=dict)
    processing_needs: List[str] = field(default_factory=list)
    latency_requirements: Dict[str, str] = field(default_factory=dict)
    security_requirements: List[str] = field(default_factory=list)
    analytics_needs: List[str] = field(default_factory=list)
    is_complete: bool = False

@dataclass
class ArchitectureState:
    """Stores the architecture design state."""
    components: Dict[str, Any] = field(default_factory=dict)
    data_flow: List[Dict[str, Any]] = field(default_factory=list)
    services: List[str] = field(default_factory=list)
    is_complete: bool = False

@dataclass
class ValidationState:
    """Stores validation results and recommendations."""
    issues: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    is_valid: bool = False

@dataclass
class ImplementationState:
    """Stores implementation details and progress."""
    code_components: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    is_complete: bool = False

@dataclass
class MemoryState:
    """Manages the agent's memory and context."""
    short_term: List[Message] = field(default_factory=list)
    long_term: List[Message] = field(default_factory=list)
    working_memory: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolState:
    """Tracks active tools and their usage history."""
    active_tools: List[str] = field(default_factory=list)
    tool_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ErrorState:
    """Represents error states and their context."""
    error_type: str
    error_message: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentState:
    """Main state class that combines all state components."""
    conversation: ConversationContext
    requirements: RequirementsState
    architecture: ArchitectureState
    validation: ValidationState
    implementation: ImplementationState
    memory: MemoryState
    tools: ToolState
    current_agent: AgentRole
    error: Optional[ErrorState] = None
    is_done: bool = False 