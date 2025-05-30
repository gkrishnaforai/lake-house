from typing import Dict, List, Optional, Set
import asyncio
import logging
from datetime import datetime
from .base_agent import BaseAgent, AgentEvent
from .agent_factory import AgentFactory

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Orchestrates communication and coordination between agents."""

    def __init__(self):
        self.event_queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
        self.running = False
        self.event_handlers: Dict[str, Set[str]] = {}
        self.logger = logging.getLogger(f"{__name__}.orchestrator")

    async def start(self) -> None:
        """Start the orchestrator."""
        if self.running:
            return

        self.running = True
        self.logger.info("Starting agent orchestrator")
        asyncio.create_task(self._process_events())

    async def stop(self) -> None:
        """Stop the orchestrator."""
        if not self.running:
            return

        self.running = False
        self.logger.info("Stopping agent orchestrator")
        # Wait for event queue to be processed
        await self.event_queue.join()

    async def register_agent(
        self,
        agent_id: str,
        event_types: List[str]
    ) -> None:
        """Register an agent to handle specific event types."""
        if agent_id not in self.event_handlers:
            self.event_handlers[agent_id] = set()
        self.event_handlers[agent_id].update(event_types)
        self.logger.info(
            f"Registered agent {agent_id} for events: {event_types}"
        )

    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from handling events."""
        if agent_id in self.event_handlers:
            del self.event_handlers[agent_id]
            self.logger.info(f"Unregistered agent {agent_id}")

    async def send_event(self, event: AgentEvent) -> None:
        """Send an event to be processed by the orchestrator."""
        await self.event_queue.put(event)
        self.logger.debug(f"Queued event: {event.event_type}")

    async def _process_events(self) -> None:
        """Process events from the queue."""
        while self.running:
            try:
                event = await self.event_queue.get()
                await self._handle_event(event)
                self.event_queue.task_done()
            except Exception as e:
                self.logger.error(f"Error processing event: {str(e)}")

    async def _handle_event(self, event: AgentEvent) -> None:
        """Handle a single event."""
        try:
            # If event has a specific target, only send to that agent
            if event.target_agent:
                agent = AgentFactory.get_agent(event.target_agent)
                if agent:
                    response = await agent.process_event(event)
                    if response:
                        await self.send_event(response)
                return

            # Otherwise, broadcast to all agents registered for this event type
            for agent_id, event_types in self.event_handlers.items():
                if event.event_type in event_types:
                    agent = AgentFactory.get_agent(agent_id)
                    if agent:
                        response = await agent.process_event(event)
                        if response:
                            await self.send_event(response)

        except Exception as e:
            self.logger.error(
                f"Error handling event {event.event_type}: {str(e)}"
            )

    async def broadcast_event(
        self,
        event_type: str,
        payload: Dict,
        source_agent: str
    ) -> None:
        """Broadcast an event to all agents."""
        event = AgentEvent(
            event_type=event_type,
            source_agent=source_agent,
            target_agent=None,
            payload=payload,
            timestamp=datetime.utcnow().isoformat()
        )
        await self.send_event(event)

    async def send_event_to_agent(
        self,
        event_type: str,
        payload: Dict,
        source_agent: str,
        target_agent: str
    ) -> None:
        """Send an event to a specific agent."""
        event = AgentEvent(
            event_type=event_type,
            source_agent=source_agent,
            target_agent=target_agent,
            payload=payload,
            timestamp=datetime.utcnow().isoformat()
        )
        await self.send_event(event) 