import asyncio
import logging
import os
import sys
import pytest

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.llm.manager import LLMManager
from src.core.state_management.state_manager import StateManager
from src.core.workflow.etl_agent_flow import ETLAgentFlow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Only show the message, not the log level
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_sales_dashboard_conversation():
    """Test the ETL agent conversation flow for a sales dashboard scenario."""
    # Initialize components
    llm_manager = LLMManager()
    state_manager = StateManager()
    etl_agent = ETLAgentFlow(llm_manager, state_manager)
    
    # Initial request
    user_request = "I need to create a dashboard for my sales data"
    workflow_id = "test_sales_dashboard"
    
    # Start the conversation
    print("\n=== Starting ETL Conversation ===")
    print(f"User: {user_request}")
    
    # Run the workflow
    state = await etl_agent.run(workflow_id, user_request)
    
    # Continue conversation until requirements are complete
    conversation_step = 1
    while True:
        # Check if we have requirements and next question
        if state.requirements and state.requirements.next_question:
            print(f"\n=== Step {conversation_step} ===")
            print(f"Agent: {state.requirements.next_question}")
            
            # Simulate user response based on the question
            if "volume" in state.requirements.next_question.lower():
                user_response = "We process about 10,000 sales records daily with peaks up to 50,000 during holidays"
            elif "format" in state.requirements.next_question.lower():
                user_response = "The data is in CSV format stored in S3"
            elif "access" in state.requirements.next_question.lower():
                user_response = "We need read-only access for the dashboard users"
            elif "refresh" in state.requirements.next_question.lower():
                user_response = "The dashboard should refresh every hour"
            else:
                user_response = "The data includes sales amount, region, date, and product category"
            
            print(f"User: {user_response}")
            
            # Continue the workflow with the response
            state = await etl_agent.run(workflow_id, user_response)
            conversation_step += 1
            
            # Check if we have all required information
            if not state.requirements.missing_info:
                print("\n=== All Requirements Gathered ===")
                break
        else:
            break
    
    # Log the final requirements
    print("\n=== Final Requirements ===")
    print(f"Data Sources: {state.requirements.data_sources}")
    print(f"Processing Type: {state.requirements.processing_type}")
    print(f"Data Format: {state.requirements.data_format}")
    print(f"Volume: {state.requirements.volume}")
    print(f"Latency Requirements: {state.requirements.latency_requirements}")
    print(f"Analytics Needs: {state.requirements.analytics_needs}")
    
    # Verify the requirements are complete
    assert state.requirements is not None
    assert not state.requirements.missing_info
    assert state.requirements.data_sources
    assert state.requirements.processing_type
    assert state.requirements.data_format
    assert state.requirements.volume
    assert state.requirements.latency_requirements
    assert state.requirements.analytics_needs

if __name__ == "__main__":
    asyncio.run(test_sales_dashboard_conversation()) 