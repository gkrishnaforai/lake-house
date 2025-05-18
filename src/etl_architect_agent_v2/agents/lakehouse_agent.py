"""Lakehouse automation agent for data ingestion and management."""

from typing import Dict, Any, Optional
import logging
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from ..core.state import AgentState
from ..core.schema_registry import SchemaRegistry
from ..nodes.lakehouse_nodes import LakehouseNodes

logger = logging.getLogger(__name__)


class LakehouseAgent:
    """Agent for automating lakehouse data ingestion and management."""
    
    def __init__(
        self,
        region_name: str = "us-east-1",
        schema_registry: Optional[SchemaRegistry] = None
    ):
        """Initialize the lakehouse agent.
        
        Args:
            region_name: AWS region name
            schema_registry: Optional schema registry instance
        """
        self.nodes = LakehouseNodes(
            region_name=region_name,
            schema_registry=schema_registry
        )
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for lakehouse operations."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("discover_files", self._discover_files)
        workflow.add_node("infer_schema", self._infer_schema)
        workflow.add_node("create_tables", self._create_tables)
        workflow.add_node("load_data", self._load_data)
        workflow.add_node("generate_report", self._generate_report)
        workflow.add_node("handle_error", self._handle_error)
        
        # Set entry point
        workflow.set_entry_point("discover_files")
        
        # Add edges
        workflow.add_edge("discover_files", "infer_schema")
        workflow.add_edge("infer_schema", "create_tables")
        workflow.add_edge("create_tables", "load_data")
        workflow.add_edge("load_data", "generate_report")
        workflow.add_edge("generate_report", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    async def _discover_files(self, state: AgentState) -> AgentState:
        """Discover new files in S3 bucket."""
        try:
            return await self.nodes.discover_files(
                state=state,
                bucket_name=state["bucket_name"],
                prefix=state.get("prefix", ""),
                file_types=state.get("file_types")
            )
        except Exception as e:
            state["error"] = f"Error discovering files: {str(e)}"
            return state
    
    async def _infer_schema(self, state: AgentState) -> AgentState:
        """Infer schema from discovered files."""
        try:
            return await self.nodes.infer_schema(
                state=state,
                sample_size=state.get("sample_size", 1000)
            )
        except Exception as e:
            state["error"] = f"Error inferring schema: {str(e)}"
            return state
    
    async def _create_tables(self, state: AgentState) -> AgentState:
        """Create or update Iceberg tables."""
        try:
            return await self.nodes.create_iceberg_tables(
                state=state,
                database_name=state["database_name"]
            )
        except Exception as e:
            state["error"] = f"Error creating tables: {str(e)}"
            return state
    
    async def _load_data(self, state: AgentState) -> AgentState:
        """Load data into Iceberg tables."""
        try:
            return await self.nodes.load_data(
                state=state,
                database_name=state["database_name"]
            )
        except Exception as e:
            state["error"] = f"Error loading data: {str(e)}"
            return state
    
    async def _generate_report(self, state: AgentState) -> AgentState:
        """Generate summary report."""
        try:
            return await self.nodes.generate_report(
                state=state,
                database_name=state["database_name"]
            )
        except Exception as e:
            state["error"] = f"Error generating report: {str(e)}"
            return state
    
    async def _handle_error(self, state: AgentState) -> AgentState:
        """Handle errors in the workflow."""
        logger.error(f"Workflow error: {state.get('error', 'Unknown error')}")
        return state
    
    async def run_workflow(
        self,
        bucket_name: str,
        database_name: str,
        prefix: str = "",
        file_types: Optional[list] = None,
        sample_size: int = 1000
    ) -> Dict[str, Any]:
        """Run the lakehouse automation workflow.
        
        Args:
            bucket_name: S3 bucket name
            database_name: Glue database name
            prefix: S3 prefix to scan
            file_types: List of file extensions to include
            sample_size: Number of rows to sample for schema inference
            
        Returns:
            Workflow results
        """
        initial_state = AgentState(
            bucket_name=bucket_name,
            database_name=database_name,
            prefix=prefix,
            file_types=file_types or ['.csv', '.json'],
            sample_size=sample_size,
            error=None,
            conversation_history=[]
        )
        
        try:
            final_state = await self.workflow.ainvoke(initial_state)
            
            if final_state.get("error"):
                raise RuntimeError(final_state["error"])
            
            return {
                "status": "success",
                "discovered_files": final_state.get("discovered_files", []),
                "created_tables": final_state.get("created_tables", []),
                "loaded_tables": final_state.get("loaded_tables", []),
                "report": final_state.get("report", {})
            }
        except Exception as e:
            logger.error(f"Error running workflow: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            } 