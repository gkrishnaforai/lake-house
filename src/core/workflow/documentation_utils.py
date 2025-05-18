from typing import Optional
from .documentation_state import DocumentationState
from .utils import (
    update_state_timestamp,
    add_metadata,
    get_metadata,
    validate_state_required_fields
)


def validate_documentation_state(state: DocumentationState) -> bool:
    """Validate that a documentation state has all required fields.
    
    Args:
        state: The state to validate
        
    Returns:
        True if state is valid
    """
    required_fields = ["topic", "documentation"]
    return validate_state_required_fields(state, required_fields)


def is_documentation_approved(state: DocumentationState) -> bool:
    """Check if documentation has been approved.
    
    Args:
        state: The state to check
        
    Returns:
        True if documentation is approved
    """
    return state["is_approved"]


def add_review_metadata(
    state: DocumentationState,
    reviewer: str,
    review_type: str
) -> DocumentationState:
    """Add review metadata to documentation state.
    
    Args:
        state: The state to update
        reviewer: Name of the reviewer
        review_type: Type of review (technical/style)
        
    Returns:
        Updated state with review metadata
    """
    return add_metadata(
        state,
        "review_info",
        {
            "reviewer": reviewer,
            "type": review_type,
            "timestamp": update_state_timestamp(state)["timestamp"]
        }
    )


def get_review_history(state: DocumentationState) -> list[dict]:
    """Get the review history from metadata.
    
    Args:
        state: The state to query
        
    Returns:
        List of review history entries
    """
    return get_metadata(state, "review_history", [])


def is_documentation_complete(state: DocumentationState) -> bool:
    """Check if documentation workflow is complete.
    
    Args:
        state: The state to check
        
    Returns:
        True if documentation is complete
    """
    completion_fields = [
        "documentation",
        "validation_feedback",
        "review_feedback"
    ]
    return all(
        field in state and state[field] is not None and state[field] != ""
        for field in completion_fields
    )


def get_documentation_stats(state: DocumentationState) -> dict:
    """Get statistics about the documentation.
    
    Args:
        state: The state to analyze
        
    Returns:
        Dictionary containing documentation statistics
    """
    doc_length = len(state["documentation"]) if state["documentation"] else 0
    validation_length = len(state["validation_feedback"]) if state["validation_feedback"] else 0
    review_length = len(state["review_feedback"]) if state["review_feedback"] else 0
    
    return {
        "documentation_length": doc_length,
        "validation_feedback_length": validation_length,
        "review_feedback_length": review_length,
        "is_approved": state["is_approved"],
        "has_validation": bool(state["validation_feedback"]),
        "has_review": bool(state["review_feedback"])
    } 