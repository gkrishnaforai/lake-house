"""ETL Architect Agent V2 - A multi-agent system for ETL pipeline design."""

__version__ = "0.1.0"

from .core.state_manager import (
    AgentState,
    AgentRole,
    Message,
    MessageRole,
    ConversationContext,
    RequirementsState,
    ArchitectureState,
    ValidationState,
    ImplementationState,
    MemoryState,
    ToolState,
    ErrorState
)

from .core.agent_orchestrator import AgentOrchestrator

__all__ = [
    "AgentState",
    "AgentRole",
    "Message",
    "MessageRole",
    "ConversationContext",
    "RequirementsState",
    "ArchitectureState",
    "ValidationState",
    "ImplementationState",
    "MemoryState",
    "ToolState",
    "ErrorState",
    "AgentOrchestrator"
] 