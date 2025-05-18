from datetime import datetime
import json

from langgraph.graph import StateGraph, START, END

from src.core.state_management.execution_manager import (
    ExecutionManager,
    ExecutionStatus
)
from src.core.workflow.documentation_state import (
    DocumentationState,
    create_documentation_state
)
from src.core.llm.manager import LLMManager


# Initialize components
llm_manager = LLMManager(
    model="gpt-3.5-turbo",
    api_key="YOUR_API_KEY"  # Replace with environment variable
)
execution_manager = ExecutionManager()


def track_state(state: DocumentationState) -> DocumentationState:
    """Track state changes using execution manager."""
    execution_manager.add_state(state["execution_id"], state)
    return state


def generate_documentation(state: DocumentationState) -> DocumentationState:
    """Generate documentation for the given topic."""
    template = "Generate comprehensive documentation about {topic}"
    state["documentation"] = llm_manager.invoke(
        template,
        {"topic": state["topic"]}
    )
    return track_state(state)


def validate_documentation(state: DocumentationState) -> DocumentationState:
    """Validate the generated documentation."""
    template = """
    Validate this documentation for {topic}:
    {doc}
    
    Provide technical feedback focusing on accuracy and completeness.
    """
    state["validation_feedback"] = llm_manager.invoke(
        template,
        {
            "topic": state["topic"],
            "doc": state["documentation"]
        }
    )
    return track_state(state)


def review_documentation(state: DocumentationState) -> DocumentationState:
    """Review the documentation and provide feedback."""
    template = """
    Review this documentation for {topic}:
    {doc}
    
    Provide feedback on style, clarity, and organization.
    """
    state["review_feedback"] = llm_manager.invoke(
        template,
        {
            "topic": state["topic"],
            "doc": state["documentation"]
        }
    )
    return track_state(state)


def create_workflow() -> StateGraph:
    """Create and configure the workflow."""
    workflow = StateGraph(DocumentationState)
    
    # Add nodes
    workflow.add_node("generate", generate_documentation)
    workflow.add_node("validate", validate_documentation)
    workflow.add_node("review", review_documentation)
    
    # Add edges
    workflow.add_edge("generate", "validate")
    workflow.add_edge("validate", "review")
    workflow.add_edge(START, "generate")
    workflow.add_edge("review", END)
    
    return workflow


def run_workflow(topic: str, user: str = "test_user") -> DocumentationState:
    """Run the workflow with execution tracking."""
    # Create new execution
    execution = execution_manager.create_execution(
        user=user,
        environment="test",
        trigger="manual"
    )
    
    # Initialize state
    state = create_documentation_state(
        topic=topic,
        execution_id=execution.execution_id
    )
    
    try:
        # Create and run workflow
        workflow = create_workflow()
        app = workflow.compile()
        result = app.invoke(state)
        
        # Update execution status
        execution_manager.update_execution_status(
            execution.execution_id,
            ExecutionStatus.COMPLETED,
            datetime.now().isoformat()
        )
        
        return result
    except Exception as e:
        # Update execution status on failure
        execution_manager.update_execution_status(
            execution.execution_id,
            ExecutionStatus.FAILED,
            datetime.now().isoformat()
        )
        raise e


if __name__ == "__main__":
    # Example usage
    topic = "AWS Lambda with Python"
    result = run_workflow(topic)
    
    # Print execution history
    execution = execution_manager.get_execution(result["execution_id"])
    print("\nExecution History:")
    for state in execution.states:
        print(f"\nTimestamp: {state['timestamp']}")
        print(f"State: {json.dumps(state['state'], indent=2)}") 