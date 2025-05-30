import asyncio
import logging
from .agent_factory import AgentFactory
from .agent_orchestrator import AgentOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_divider(title):
    logger.info("\n" + "=" * 30 + f" {title} " + "=" * 30)


async def main():
    orchestrator = AgentOrchestrator()
    await orchestrator.start()

    # Create Data Exploration Agent
    explorer_agent = await AgentFactory.create_agent(
        "data_exploration", "explorer_1"
    )
    if not explorer_agent:
        logger.error("Failed to create Data Exploration Agent")
        return
    await orchestrator.register_agent(
        "explorer_1",
        ["query_request", "schema_request", "data_quality_request"]
    )

    # Simulate a query event
    print_divider("QUERY REQUEST")
    await orchestrator.send_event_to_agent(
        event_type="query_request",
        payload={"query": "SELECT * FROM customers LIMIT 10"},
        source_agent="test_client",
        target_agent="explorer_1"
    )
    await asyncio.sleep(0.5)

    # Simulate a schema exploration event
    print_divider("SCHEMA REQUEST")
    await orchestrator.send_event_to_agent(
        event_type="schema_request",
        payload={"table_name": "customers"},
        source_agent="test_client",
        target_agent="explorer_1"
    )
    await asyncio.sleep(0.5)

    # Simulate a data quality event
    print_divider("DATA QUALITY REQUEST")
    await orchestrator.send_event_to_agent(
        event_type="data_quality_request",
        payload={"table_name": "customers"},
        source_agent="test_client",
        target_agent="explorer_1"
    )
    await asyncio.sleep(0.5)

    # Optionally, print agent state
    state = await explorer_agent.get_state()
    logger.info(f"Explorer Agent State: {state}")

    # Clean up
    await AgentFactory.destroy_agent("explorer_1")
    await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main()) 