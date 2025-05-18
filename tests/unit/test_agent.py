from unittest.mock import MagicMock, patch

import pytest

from aws_architect_agent.core.agent import AWSArchitectAgent
from aws_architect_agent.models.base import (
    ArchitectureState,
    SolutionType,
)


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    with patch("langchain.chat_models.ChatOpenAI") as mock:
        mock.return_value = MagicMock()
        yield mock


@pytest.fixture
def agent(mock_llm):
    """Create an agent instance."""
    return AWSArchitectAgent()


def test_agent_initialization(agent):
    """Test agent initialization."""
    assert agent.conversation_state.current_state == ArchitectureState.INITIAL
    assert agent.conversation_state.current_architecture is None
    assert len(agent.conversation_state.conversation_history) == 0


def test_process_message(agent, mock_llm):
    """Test message processing."""
    mock_llm.return_value.return_value.content = "Test response"
    
    response = agent.process_message("Hello")
    
    assert response == "Test response"
    assert len(agent.conversation_state.conversation_history) == 2
    assert agent.conversation_state.conversation_history[0]["role"] == "user"
    assert (
        agent.conversation_state.conversation_history[0]["content"] == "Hello"
    )
    assert (
        agent.conversation_state.conversation_history[1]["role"] == "assistant"
    )
    assert (
        agent.conversation_state.conversation_history[1]["content"] 
        == "Test response"
    )


def test_requirements_gathering(agent, mock_llm):
    """Test requirements gathering state."""
    mock_llm.return_value.return_value.content = "requirements_complete"
    
    agent.process_message("I need an ETL pipeline")
    
    assert agent.conversation_state.current_state == ArchitectureState.DESIGN
    assert agent.conversation_state.current_architecture is not None
    assert (
        agent.conversation_state.current_architecture.solution_type
        == SolutionType.ETL
    )


def test_architecture_design(agent, mock_llm):
    """Test architecture design state."""
    # First, gather requirements
    mock_llm.return_value.return_value.content = "requirements_complete"
    agent.process_message("I need an ETL pipeline")
    
    # Then design architecture
    mock_llm.return_value.return_value.content = "Architecture designed"
    agent.process_message("Design the architecture")
    
    assert (
        agent.conversation_state.current_state == ArchitectureState.REVIEW
    )


def test_architecture_review(agent, mock_llm):
    """Test architecture review state."""
    # First, gather requirements and design
    mock_llm.return_value.return_value.content = "requirements_complete"
    agent.process_message("I need an ETL pipeline")
    mock_llm.return_value.return_value.content = "Architecture designed"
    agent.process_message("Design the architecture")
    
    # Then review
    mock_llm.return_value.return_value.content = "Review complete"
    agent.process_message("Review the architecture")
    
    assert (
        agent.conversation_state.current_state == ArchitectureState.REFINEMENT
    )


def test_architecture_refinement(agent, mock_llm):
    """Test architecture refinement state."""
    # First, go through all previous states
    mock_llm.return_value.return_value.content = "requirements_complete"
    agent.process_message("I need an ETL pipeline")
    mock_llm.return_value.return_value.content = "Architecture designed"
    agent.process_message("Design the architecture")
    mock_llm.return_value.return_value.content = "Review complete"
    agent.process_message("Review the architecture")
    
    # Then refine
    mock_llm.return_value.return_value.content = "refinement_complete"
    agent.process_message("Refine the architecture")
    
    assert agent.conversation_state.current_state == ArchitectureState.FINALIZED 