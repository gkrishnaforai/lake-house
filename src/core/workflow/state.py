from typing import TypedDict, Optional
from datetime import datetime


class BaseWorkflowState(TypedDict):
    """Base state for all workflows."""
    input: str
    output: str
    execution_id: str
    timestamp: str
    metadata: Optional[dict]


class DocumentationState(BaseWorkflowState):
    """State for documentation generation workflow."""
    topic: str
    documentation: str
    validation_feedback: str
    review_feedback: str
    is_approved: bool


def create_initial_state(
    input_data: str,
    execution_id: str,
    metadata: Optional[dict] = None
) -> BaseWorkflowState:
    """Create initial state for any workflow."""
    return {
        "input": input_data,
        "output": "",
        "execution_id": execution_id,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {}
    }


def create_documentation_state(
    topic: str,
    execution_id: str,
    metadata: Optional[dict] = None
) -> DocumentationState:
    """Create initial state for documentation workflow."""
    base_state = create_initial_state(topic, execution_id, metadata)
    return {
        **base_state,
        "topic": topic,
        "documentation": "",
        "validation_feedback": "",
        "review_feedback": "",
        "is_approved": False
    } 