"""Pytest configuration file for ETL Architect Agent V2 tests."""

import pytest
import asyncio
from etl_architect_agent_v2 import AgentOrchestrator


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def agent_orchestrator():
    """Create an instance of AgentOrchestrator for testing."""
    return AgentOrchestrator() 