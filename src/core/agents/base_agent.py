from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel
from datetime import datetime


class AgentState(BaseModel):
    """Base state model for agents."""
    agent_id: str
    status: str
    last_updated: str
    metadata: Dict[str, Any] = {}


class AgentEvent(BaseModel):
    """Base event model for agent communication."""
    event_type: str
    source_agent: str
    target_agent: Optional[str]
    payload: Dict[str, Any]
    timestamp: str


class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.state = AgentState(
            agent_id=agent_id,
            status="initialized",
            last_updated=datetime.utcnow().isoformat(),
            metadata={}
        )
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent and its resources."""
        pass
    
    @abstractmethod
    async def process_event(self, event: AgentEvent) -> Optional[AgentEvent]:
        """Process an incoming event and optionally emit a response event."""
        pass
    
    @abstractmethod
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific task assigned to the agent."""
        pass
    
    @abstractmethod
    async def get_state(self) -> AgentState:
        """Get the current state of the agent."""
        pass
    
    @abstractmethod
    async def update_state(self, new_state: Dict[str, Any]) -> None:
        """Update the agent's state."""
        pass
    
    @abstractmethod
    async def handle_error(self, error: Exception) -> None:
        """Handle errors that occur during agent operation."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources when the agent is shutting down."""
        pass
    
    def emit_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        target_agent: Optional[str] = None
    ) -> AgentEvent:
        """Create and emit an event."""
        return AgentEvent(
            event_type=event_type,
            source_agent=self.agent_id,
            target_agent=target_agent,
            payload=payload,
            timestamp=datetime.utcnow().isoformat()
        ) 