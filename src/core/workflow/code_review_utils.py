from .code_review_state import CodeReviewState
from .utils import (
    update_state_timestamp,
    add_metadata,
    get_metadata,
    validate_state_required_fields
)


def validate_code_review_state(state: CodeReviewState) -> bool:
    """Validate that a code review state has all required fields.
    
    Args:
        state: The state to validate
        
    Returns:
        True if state is valid
    """
    required_fields = ["code", "language"]
    return validate_state_required_fields(state, required_fields)


def is_code_review_approved(state: CodeReviewState) -> bool:
    """Check if code review has been approved.
    
    Args:
        state: The state to check
        
    Returns:
        True if code review is approved
    """
    return state["is_approved"]


def add_code_review_metadata(
    state: CodeReviewState,
    reviewer: str,
    review_type: str
) -> CodeReviewState:
    """Add review metadata to code review state.
    
    Args:
        state: The state to update
        reviewer: Name of the reviewer
        review_type: Type of review (style/security/performance)
        
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


def get_code_review_history(state: CodeReviewState) -> list[dict]:
    """Get the review history from metadata.
    
    Args:
        state: The state to query
        
    Returns:
        List of review history entries
    """
    return get_metadata(state, "review_history", [])


def is_code_review_complete(state: CodeReviewState) -> bool:
    """Check if code review workflow is complete.
    
    Args:
        state: The state to check
        
    Returns:
        True if code review is complete
    """
    completion_fields = [
        "style_feedback",
        "security_feedback",
        "performance_feedback"
    ]
    return all(
        field in state and state[field] is not None and state[field] != ""
        for field in completion_fields
    )


def get_code_review_stats(state: CodeReviewState) -> dict:
    """Get statistics about the code review.
    
    Args:
        state: The state to analyze
        
    Returns:
        Dictionary containing code review statistics
    """
    code_length = len(state["code"]) if state["code"] else 0
    style_length = len(state["style_feedback"]) if state["style_feedback"] else 0
    security_length = len(state["security_feedback"]) if state["security_feedback"] else 0
    perf_length = len(state["performance_feedback"]) if state["performance_feedback"] else 0
    
    return {
        "code_length": code_length,
        "style_feedback_length": style_length,
        "security_feedback_length": security_length,
        "performance_feedback_length": perf_length,
        "is_approved": state["is_approved"],
        "has_style_review": bool(state["style_feedback"]),
        "has_security_review": bool(state["security_feedback"]),
        "has_performance_review": bool(state["performance_feedback"])
    } 