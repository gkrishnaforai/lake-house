from typing import Dict, Optional, Type
import logging
from .base_agent import BaseAgent
from .data_exploration_agent import DataExplorationAgent

logger = logging.getLogger(__name__)


class AgentFactory:
    """Factory class for creating and managing agents."""

    _agents: Dict[str, Type[BaseAgent]] = {
        "data_exploration": DataExplorationAgent,
    }

    _instances: Dict[str, BaseAgent] = {}

    @classmethod
    async def create_agent(
        cls,
        agent_type: str,
        agent_id: str
    ) -> Optional[BaseAgent]:
        """Create a new agent instance."""
        try:
            if agent_type not in cls._agents:
                raise ValueError(f"Unknown agent type: {agent_type}")

            if agent_id in cls._instances:
                raise ValueError(f"Agent with ID {agent_id} already exists")

            agent_class = cls._agents[agent_type]
            agent = agent_class(agent_id)
            await agent.initialize()
            cls._instances[agent_id] = agent
            logger.info(f"Created {agent_type} agent with ID {agent_id}")
            return agent
        except Exception as e:
            logger.error(f"Error creating agent: {str(e)}")
            return None

    @classmethod
    def get_agent(cls, agent_id: str) -> Optional[BaseAgent]:
        """Get an existing agent instance."""
        return cls._instances.get(agent_id)

    @classmethod
    async def destroy_agent(cls, agent_id: str) -> bool:
        """Destroy an agent instance."""
        try:
            agent = cls._instances.get(agent_id)
            if not agent:
                return False

            await agent.cleanup()
            del cls._instances[agent_id]
            logger.info(f"Destroyed agent with ID {agent_id}")
            return True
        except Exception as e:
            logger.error(f"Error destroying agent: {str(e)}")
            return False

    @classmethod
    def list_agents(cls) -> Dict[str, str]:
        """List all active agents and their types."""
        return {
            agent_id: agent.__class__.__name__
            for agent_id, agent in cls._instances.items()
        }

    @classmethod
    def register_agent_type(
        cls,
        agent_type: str,
        agent_class: Type[BaseAgent]
    ) -> None:
        """Register a new agent type."""
        if not issubclass(agent_class, BaseAgent):
            raise ValueError(
                f"Agent class must inherit from BaseAgent: {agent_class}"
            )
        cls._agents[agent_type] = agent_class
        logger.info(f"Registered new agent type: {agent_type}") 