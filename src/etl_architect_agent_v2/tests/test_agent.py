"""Tests for the ETL Architect Agent V2."""

import pytest
from etl_architect_agent_v2 import AgentOrchestrator, AgentState

@pytest.mark.asyncio
async def test_agent_workflow():
    """Test the complete agent workflow."""
    # Initialize orchestrator
    orchestrator = AgentOrchestrator()
    
    # Test message
    message = (
        "Create an ETL pipeline for processing sales data. "
        "The data comes from multiple sources including our website, "
        "mobile app, and third-party marketplaces. We need to process "
        "about 100,000 transactions daily with peaks up to 500,000 "
        "during holiday seasons. The data needs to be cleaned, "
        "transformed, and loaded into our data warehouse for analytics."
    )
    
    # Run workflow
    state = await orchestrator.run(message)
    
    # Verify state
    assert isinstance(state, AgentState)
    assert state.requirements.is_complete
    assert state.architecture.is_complete
    assert state.validation.is_valid
    assert state.implementation.is_complete
    
    # Verify requirements
    assert state.requirements.data_sources
    assert state.requirements.data_volume
    assert state.requirements.processing_needs
    assert state.requirements.latency_requirements
    assert state.requirements.security_requirements
    assert state.requirements.analytics_needs
    
    # Verify architecture
    assert state.architecture.components
    assert state.architecture.data_flow
    assert state.architecture.services
    
    # Verify validation
    assert state.validation.issues
    assert state.validation.recommendations
    
    # Verify implementation
    assert state.implementation.code_components
    assert state.implementation.dependencies 