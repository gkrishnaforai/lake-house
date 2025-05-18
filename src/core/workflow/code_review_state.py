from .base_state import BaseWorkflowState, create_initial_state


class CodeReviewState(BaseWorkflowState):
    """State for code review workflow.
    
    Extends the base workflow state with fields specific to code review
    workflows.
    """
    code: str
    language: str
    style_feedback: str
    security_feedback: str
    performance_feedback: str
    is_approved: bool


def create_code_review_state(
    code: str,
    language: str,
    execution_id: str,
    metadata: dict = None
) -> CodeReviewState:
    """Create initial state for code review workflow.
    
    Args:
        code: The code to review
        language: Programming language of the code
        execution_id: Unique identifier for this execution
        metadata: Optional additional metadata
        
    Returns:
        CodeReviewState: Initialized code review state
    """
    base_state = create_initial_state(code, execution_id, metadata)
    return {
        **base_state,
        "code": code,
        "language": language,
        "style_feedback": "",
        "security_feedback": "",
        "performance_feedback": "",
        "is_approved": False
    } 