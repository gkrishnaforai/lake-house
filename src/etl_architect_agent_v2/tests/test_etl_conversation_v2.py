"""Tests for the ETL Architect Agent V2 conversation flow."""

import pytest
from typing import List
from dataclasses import dataclass
from datetime import datetime
from etl_architect_agent_v2 import AgentOrchestrator
from etl_architect_agent_v2.core.state_manager import (
    AgentState,
    AgentRole,
    MessageRole,
    Message
)

@dataclass
class UserMessage:
    """Simulates a user message in the conversation."""
    role: str
    content: str
    timestamp: datetime = datetime.now()

class MockLLMUser:
    """Simulates a user providing ETL requirements through conversation."""
    
    def __init__(self):
        self.conversation_history: List[UserMessage] = []
        self.requirements_provided = False
    
    async def provide_initial_requirements(self) -> str:
        """Simulates user's initial request for ETL pipeline."""
        message = (
            "I need to create an ETL pipeline for our e-commerce platform. "
            "We have sales data from multiple sources including our website, "
            "mobile app, and third-party marketplaces. The data needs to be "
            "processed and loaded into our data warehouse for analytics."
        )
        self.conversation_history.append(UserMessage(role="user", content=message))
        return message
    
    async def respond_to_question(self, question: str) -> str:
        """Simulates user's response to agent's questions."""
        if "volume" in question.lower():
            response = (
                "We process about 100,000 transactions daily with peaks up to "
                "500,000 during holiday seasons. The data volume is expected to "
                "grow by 20% annually."
            )
        elif "format" in question.lower():
            response = (
                "The data comes in various formats: JSON from our website API, "
                "CSV from mobile app, and XML from third-party marketplaces. "
                "We also have some legacy data in Excel format."
            )
        elif "latency" in question.lower():
            response = (
                "We need near real-time processing for critical metrics like "
                "inventory levels and sales trends. Other analytics can be "
                "processed in daily batches."
            )
        elif "security" in question.lower():
            response = (
                "The data contains sensitive customer information and payment "
                "details. We need to ensure GDPR compliance and implement "
                "proper encryption and access controls."
            )
        else:
            response = (
                "We need to track customer behavior, product performance, "
                "inventory levels, and sales trends across all channels. "
                "The analytics should help us optimize inventory and "
                "personalize customer experiences."
            )
        
        self.conversation_history.append(UserMessage(role="user", content=response))
        return response

@pytest.mark.asyncio
async def test_etl_requirements_gathering():
    """Test the complete ETL requirements gathering process."""
    # Initialize mock user and orchestrator
    mock_user = MockLLMUser()
    orchestrator = AgentOrchestrator()
    
    # Get initial requirements
    initial_requirements = await mock_user.provide_initial_requirements()
    
    # Process initial requirements
    state = await orchestrator.run(initial_requirements)
    
    # Verify initial state
    assert state is not None
    assert state.current_agent == AgentRole.REQUIREMENTS
    assert len(state.conversation.messages) == 1
    assert state.conversation.messages[0].role == MessageRole.USER
    
    # Continue conversation until requirements are complete
    while not state.requirements.is_complete:
        # Get next question
        next_question = state.conversation.next_question
        assert next_question is not None
        
        # Get user response
        user_response = await mock_user.respond_to_question(next_question)
        
        # Process response
        state = await orchestrator.run(user_response)
    
    # Verify final state
    assert state.requirements.is_complete
    assert state.requirements.data_sources
    assert state.requirements.data_volume
    assert state.requirements.processing_needs
    assert state.requirements.latency_requirements
    assert state.requirements.security_requirements
    assert state.requirements.analytics_needs 