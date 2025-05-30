from typing import Dict, Any, Optional
from datetime import datetime
import logging
from .base_agent import BaseAgent, AgentEvent, AgentState
from ..aws.aws_service_manager import AWSServiceManager


logger = logging.getLogger(__name__)


class DataExplorationAgent(BaseAgent):
    """Agent responsible for data exploration tasks in the data lake."""

    def __init__(
        self,
        agent_id: str,
        aws_manager: Optional[AWSServiceManager] = None,
        region_name: str = "us-east-1"
    ):
        """Initialize the data exploration agent.
        
        Args:
            agent_id: Unique identifier for the agent
            aws_manager: Optional AWS service manager instance
            region_name: AWS region name
        """
        super().__init__(agent_id)
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
        self.aws_manager = aws_manager or AWSServiceManager(
            region_name=region_name
        )

    async def initialize(self) -> None:
        """Initialize the data exploration agent."""
        try:
            await self.update_state({
                "status": "ready",
                "last_updated": datetime.utcnow().isoformat(),
                "metadata": {
                    "capabilities": [
                        "query_execution",
                        "schema_exploration",
                        "data_quality_check"
                    ]
                }
            })
            self.logger.info(
                f"Data Exploration Agent {self.agent_id} initialized"
            )
        except Exception as e:
            await self.handle_error(e)
            raise

    async def process_event(self, event: AgentEvent) -> Optional[AgentEvent]:
        """Process incoming events for data exploration tasks."""
        try:
            if event.event_type == "query_request":
                return await self._handle_query_request(event)
            elif event.event_type == "schema_request":
                return await self._handle_schema_request(event)
            elif event.event_type == "data_quality_request":
                return await self._handle_data_quality_request(event)
            else:
                self.logger.warning(f"Unknown event type: {event.event_type}")
                return None
        except Exception as e:
            await self.handle_error(e)
            return None

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific task assigned to the agent."""
        try:
            task_type = task.get("type")
            if task_type == "query":
                return await self._execute_query(task)
            elif task_type == "schema":
                return await self._explore_schema(task)
            elif task_type == "quality":
                return await self._check_data_quality(task)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
        except Exception as e:
            await self.handle_error(e)
            raise

    async def get_state(self) -> AgentState:
        """Get the current state of the agent."""
        return self.state

    async def update_state(self, new_state: Dict[str, Any]) -> None:
        """Update the agent's state."""
        self.state = AgentState(
            agent_id=self.agent_id,
            status=new_state.get("status", self.state.status),
            last_updated=new_state.get(
                "last_updated", datetime.utcnow().isoformat()
            ),
            metadata=new_state.get("metadata", self.state.metadata)
        )

    async def handle_error(self, error: Exception) -> None:
        """Handle errors that occur during agent operation."""
        self.logger.error(f"Error in agent {self.agent_id}: {str(error)}")
        await self.update_state({
            "status": "error",
            "last_updated": datetime.utcnow().isoformat(),
            "metadata": {
                "error": str(error),
                "error_type": type(error).__name__
            }
        })

    async def cleanup(self) -> None:
        """Clean up resources when the agent is shutting down."""
        try:
            await self.update_state({
                "status": "shutting_down",
                "last_updated": datetime.utcnow().isoformat()
            })
            self.logger.info(f"Agent {self.agent_id} cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            raise

    async def _handle_query_request(
        self,
        event: AgentEvent
    ) -> Optional[AgentEvent]:
        """Handle a query request event."""
        try:
            query = event.payload.get("query")
            database = event.payload.get("database")
            output_location = event.payload.get("output_location")
            
            if not all([query, database, output_location]):
                raise ValueError("Missing required query parameters")
            
            results = await self.aws_manager.execute_query(
                query=query,
                database=database,
                output_location=output_location
            )
            
            return self.emit_event(
                event_type="query_response",
                payload=results,
                target_agent=event.source_agent
            )
        except Exception as e:
            await self.handle_error(e)
            return None

    async def _handle_schema_request(
        self,
        event: AgentEvent
    ) -> Optional[AgentEvent]:
        """Handle a schema request event."""
        try:
            database = event.payload.get("database")
            table = event.payload.get("table")
            
            if not all([database, table]):
                raise ValueError("Missing required schema parameters")
            
            schema = await self.aws_manager.get_table_schema(
                database=database,
                table=table
            )
            
            return self.emit_event(
                event_type="schema_response",
                payload=schema,
                target_agent=event.source_agent
            )
        except Exception as e:
            await self.handle_error(e)
            return None

    async def _handle_data_quality_request(
        self,
        event: AgentEvent
    ) -> Optional[AgentEvent]:
        """Handle a data quality request event."""
        try:
            database = event.payload.get("database")
            table = event.payload.get("table")
            metrics = event.payload.get("metrics", [
                "completeness",
                "accuracy",
                "consistency"
            ])
            
            if not all([database, table]):
                raise ValueError(
                    "Missing required quality check parameters"
                )
            
            quality = await self.aws_manager.check_data_quality(
                database=database,
                table=table,
                metrics=metrics
            )
            
            return self.emit_event(
                event_type="quality_response",
                payload=quality,
                target_agent=event.source_agent
            )
        except Exception as e:
            await self.handle_error(e)
            return None

    async def _execute_query(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a query on the data lake."""
        query = task.get("query")
        database = task.get("database")
        output_location = task.get("output_location")
        
        if not all([query, database, output_location]):
            raise ValueError("Missing required query parameters")
        
        return await self.aws_manager.execute_query(
            query=query,
            database=database,
            output_location=output_location
        )

    async def _explore_schema(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Explore the schema of a table."""
        database = task.get("database")
        table = task.get("table")
        
        if not all([database, table]):
            raise ValueError("Missing required schema parameters")
        
        return await self.aws_manager.get_table_schema(
            database=database,
            table=table
        )

    async def _check_data_quality(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Check data quality metrics for a table."""
        database = task.get("database")
        table = task.get("table")
        metrics = task.get("metrics", [
            "completeness",
            "accuracy",
            "consistency"
        ])
        
        if not all([database, table]):
            raise ValueError("Missing required quality check parameters")
        
        return await self.aws_manager.check_data_quality(
            database=database,
            table=table,
            metrics=metrics
        ) 