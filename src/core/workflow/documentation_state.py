from typing import TypedDict
from .base_state import BaseWorkflowState, create_initial_state


class DocumentationState(BaseWorkflowState):
    """State for documentation generation workflow.
    
    Extends the base workflow state with fields specific to documentation
    generation and review workflows.
    """
    topic: str
    documentation: str
    validation_feedback: str
    review_feedback: str
    is_approved: bool


def create_documentation_state(
    topic: str,
    execution_id: str,
    metadata: dict = None
) -> DocumentationState:
    """Create initial state for documentation workflow.
    
    Args:
        topic: The topic to document
        execution_id: Unique identifier for this execution
        metadata: Optional additional metadata
        
    Returns:
        DocumentationState: Initialized documentation state
    """
    base_state = create_initial_state(topic, execution_id, metadata)
    return {
        **base_state,
        "topic": topic,
        "documentation": "",
        "validation_feedback": "",
        "review_feedback": "",
        "is_approved": False
    } 