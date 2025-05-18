from typing import TypedDict
import json
from datetime import datetime
import pytest
from typing import Dict, Any, List
from src.core.workflow.base_state import BaseWorkflowState
from src.core.workflow.utils import (
    update_state_timestamp,
    add_metadata,
    get_metadata,
    validate_state_required_fields,
    merge_states,
    create_state_snapshot,
    is_state_complete,
    get_state_duration
)
from src.core.workflow.etl_state import ETLState, create_etl_state
from src.core.workflow.etl_utils import (
    validate_etl_state,
    is_etl_complete,
    get_etl_stats,
    add_etl_metadata
)

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from execution_manager import ExecutionManager, ExecutionStatus

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key="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"
)

# Initialize execution manager
execution_manager = ExecutionManager("/Users/krishnag/tools/llm/lanng-chain/web-app/ai_project/web-app/aws_architect_agent")

class WorkflowState(TypedDict):
    input: str
    output: str
    topic: str
    documentation: str
    validation_feedback: str
    review_feedback: str
    is_approved: bool
    execution_id: str

def track_state(state: WorkflowState) -> WorkflowState:
    """Track state changes using execution manager."""
    execution_manager.add_state(state["execution_id"], state)
    return state

def generate_documentation(state: WorkflowState) -> WorkflowState:
    """Generate documentation for the given topic."""
    prompt = ChatPromptTemplate.from_template(
        "answer the question {question}"
    )
    chain = prompt | llm
    state["documentation"] = chain.invoke(
        {"question": state["topic"]}
    ).content
    return track_state(state)

def validate_documentation(state: WorkflowState) -> WorkflowState:
    """Validate the generated documentation."""
    prompt = ChatPromptTemplate.from_template(
        "validate this documentation for {topic}:\n {doc}"
    )
    chain = prompt | llm
    state["validation_feedback"] = chain.invoke({
        "topic": state["topic"],
        "doc": state["documentation"]
    }).content
    return track_state(state)

def review_documentation(state: WorkflowState) -> WorkflowState:
    """Review the documentation and provide feedback."""
    prompt = ChatPromptTemplate.from_template(
        "review this documentation for {topic}: \n {doc} \n and provide feedback"
    )
    chain = prompt | llm
    state["review_feedback"] = chain.invoke({
        "topic": state["topic"],
        "doc": state["documentation"]
    }).content
    return track_state(state)

def process_input(state: WorkflowState) -> WorkflowState:
    """Process the input and update state."""
    state["output"] = f"Processed: {state['input']}"
    return track_state(state)

def human_interview(state: WorkflowState) -> WorkflowState:
    """Handle human input and update state."""
    print(f"\nCurrent state: {state['output']}")
    response = input("Enter your response: ")
    if response.lower() == "y":
        new_output = input("Enter new output: ")
        state["output"] = new_output
    return track_state(state)

def create_workflow() -> StateGraph:
    """Create and configure the workflow."""
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("generate", generate_documentation)
    workflow.add_node("validate", validate_documentation)
    workflow.add_node("review", review_documentation)
    # workflow.add_node("process", process_input)
    # workflow.add_node("human_interview", human_interview)
    
    # Add edges
    workflow.add_edge("generate", "validate")
    workflow.add_edge("validate", "review")
    workflow.add_edge(START, "generate")
    #workflow.add_edge("generate", "review")
    workflow.add_edge("review", END)
    
    return workflow

def run_workflow(topic: str, user: str = "test_user") -> WorkflowState:
    """Run the workflow with execution tracking."""
    # Create new execution
    execution = execution_manager.create_execution(
        user=user,
        environment="test",
        trigger="manual"
    )
    
    # Initialize state
    state = {
        "input": "",
        "output": "",
        "topic": topic,
        "documentation": "",
        "validation_feedback": "",
        "review_feedback": "",
        "is_approved": False,
        "execution_id": execution.execution_id
    }
    
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

def test_etl_workflow_with_utilities():
    """Test ETL workflow using utility functions."""
    
    # 1. Create initial ETL state
    initial_state = create_etl_state(
        source="test_data.csv",
        destination="processed_data.json",
        execution_id="test_123"
    )
    
    # 2. Validate state using utility
    assert validate_etl_state(initial_state)
    
    # 3. Add metadata using utility
    state_with_metadata = add_etl_metadata(
        initial_state,
        "test_metadata",
        {"version": "1.0", "environment": "test"}
    )
    
    # 4. Verify metadata was added
    metadata = get_metadata(state_with_metadata, "test_metadata")
    assert metadata["version"] == "1.0"
    assert metadata["environment"] == "test"
    
    # 5. Update state with extraction results
    extraction_state = merge_states(
        state_with_metadata,
        {
            "extracted_data": [{"id": 1, "name": "test"}],
            "extraction_status": "completed",
            "extraction_errors": []
        }
    )
    
    # 6. Update timestamp
    extraction_state = update_state_timestamp(extraction_state)
    
    # 7. Create snapshot
    snapshot = create_state_snapshot(extraction_state)
    assert "timestamp" in snapshot
    assert "state" in snapshot
    
    # 8. Add transformation results
    transformation_state = merge_states(
        extraction_state,
        {
            "transformed_data": [{"id": 1, "name": "TEST"}],
            "transformation_status": "completed",
            "transformation_errors": []
        }
    )
    
    # 9. Add loading results
    final_state = merge_states(
        transformation_state,
        {
            "loaded_records": 1,
            "loading_status": "completed",
            "loading_errors": []
        }
    )
    
    # 10. Verify completion using utility
    assert is_etl_complete(final_state)
    
    # 11. Get ETL statistics
    stats = get_etl_stats(final_state)
    assert stats["total_records"] == 1
    assert stats["status"] == "completed"
    assert stats["has_errors"] is False
    
    # 12. Calculate duration
    duration = get_state_duration(final_state)
    assert duration is not None
    assert duration >= 0

def test_etl_error_handling():
    """Test ETL error handling with utilities."""
    
    # 1. Create state with errors
    error_state = create_etl_state(
        source="invalid.csv",
        destination="error.json",
        execution_id="error_test"
    )
    
    # 2. Add error metadata
    error_state = add_etl_metadata(
        error_state,
        "error_info",
        {
            "type": "file_not_found",
            "message": "Source file not found",
            "timestamp": datetime.now().isoformat()
        }
    )
    
    # 3. Update state with error status
    error_state = merge_states(
        error_state,
        {
            "extraction_status": "failed",
            "extraction_errors": ["File not found"],
            "transformation_status": "skipped",
            "loading_status": "skipped"
        }
    )
    
    # 4. Verify error handling
    assert not is_etl_complete(error_state)
    stats = get_etl_stats(error_state)
    assert stats["status"] == "failed"
    assert stats["has_errors"] is True
    assert stats["error_count"] == 1

def test_etl_state_validation():
    """Test ETL state validation utilities."""
    
    # 1. Create valid state
    valid_state = create_etl_state(
        source="data.csv",
        destination="output.json",
        execution_id="valid_test"
    )
    
    # 2. Verify validation
    assert validate_etl_state(valid_state)
    
    # 3. Create invalid state
    invalid_state = create_etl_state(
        source="",  # Empty source
        destination="output.json",
        execution_id="invalid_test"
    )
    
    # 4. Verify validation fails
    assert not validate_etl_state(invalid_state)
    
    # 5. Test required fields validation
    required_fields = ["source", "destination", "execution_id"]
    assert validate_state_required_fields(valid_state, required_fields)
    assert not validate_state_required_fields(invalid_state, required_fields)

def test_etl_metadata_operations():
    """Test ETL metadata operations with utilities."""
    
    # 1. Create initial state
    state = create_etl_state(
        source="data.csv",
        destination="output.json",
        execution_id="metadata_test"
    )
    
    # 2. Add multiple metadata entries
    state = add_etl_metadata(state, "config", {"batch_size": 1000})
    state = add_etl_metadata(
        state,
        "performance",
        {"start_time": datetime.now().isoformat()}
    )
    
    # 3. Retrieve metadata
    config = get_metadata(state, "config")
    performance = get_metadata(state, "performance")
    
    assert config["batch_size"] == 1000
    assert "start_time" in performance
    
    # 4. Test non-existent metadata
    non_existent = get_metadata(state, "non_existent", default="default_value")
    assert non_existent == "default_value"

if __name__ == "__main__":
    # Example usage
    topic = "US population in 2020, 3 sentences"
    result = run_workflow(topic)
    
    # Print execution history
    execution = execution_manager.get_execution(result["execution_id"])
    print("\nExecution History:")
    for state in execution.states:
        print(f"\nTimestamp: {state['timestamp']}")
        print(f"State: {json.dumps(state['state'], indent=2)}")
