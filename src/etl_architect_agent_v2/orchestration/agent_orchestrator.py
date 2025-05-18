"""Agent orchestrator for coordinating agent interactions and managing workflows."""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from pydantic import BaseModel, ConfigDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from etl_architect_agent_v2.agents.catalog_agent import CatalogAgent
from etl_architect_agent_v2.agents.schema_extractor_agent import SchemaExtractorAgent
from etl_architect_agent_v2.agents.data_quality_agent import DataQualityAgent
from etl_architect_agent_v2.core.llm_manager import LLMManager
from etl_architect_agent_v2.core.error_handler import ErrorHandler
from etl_architect_agent_v2.backend.services.s3_service import S3Service

logger = logging.getLogger(__name__)


class WorkflowState(BaseModel):
    """State management for workflow execution."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    workflow_id: str
    workflow_type: str
    status: str = "pending"  # pending, in_progress, completed, failed
    current_step: str = "initialization"
    progress: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}
    created_at: datetime = datetime.utcnow()
    updated_at: datetime = datetime.utcnow()
    steps_completed: List[str] = []
    next_steps: List[str] = []
    agent_states: Dict[str, Any] = {}


class AgentOrchestrator:
    """Coordinates agent interactions and manages workflows."""
    
    def __init__(self, bucket: str, aws_region: str = "us-east-1"):
        """Initialize the orchestrator with required components."""
        self.llm_manager = LLMManager()
        self.error_handler = ErrorHandler()
        self.s3_service = S3Service(bucket, aws_region)
        self.catalog_agent = CatalogAgent()
        self.schema_agent = SchemaExtractorAgent()
        self.quality_agent = DataQualityAgent()
        self._setup_workflow_templates()
        
    def _setup_workflow_templates(self):
        """Set up workflow templates for different operations."""
        self.workflow_templates = {
            "file_upload": {
                "steps": [
                    "validate_input",
                    "extract_schema",
                    "check_quality",
                    "update_catalog",
                    "create_table"
                ],
                "agents": ["catalog", "schema", "quality"],
                "dependencies": {
                    "extract_schema": ["validate_input"],
                    "check_quality": ["extract_schema"],
                    "update_catalog": ["check_quality"],
                    "create_table": ["update_catalog"]
                }
            },
            "schema_evolution": {
                "steps": [
                    "validate_schema",
                    "check_compatibility",
                    "plan_migration",
                    "execute_migration",
                    "update_catalog"
                ],
                "agents": ["catalog", "schema"],
                "dependencies": {
                    "check_compatibility": ["validate_schema"],
                    "plan_migration": ["check_compatibility"],
                    "execute_migration": ["plan_migration"],
                    "update_catalog": ["execute_migration"]
                }
            }
        }
        
    async def execute_workflow(
        self,
        workflow_type: str,
        input_data: Dict[str, Any]
    ) -> WorkflowState:
        """Execute a workflow with the given type and input data."""
        try:
            # Initialize workflow state
            state = WorkflowState(
                workflow_id=f"{workflow_type}_{datetime.utcnow().isoformat()}",
                workflow_type=workflow_type,
                metadata=input_data
            )
            
            # Get workflow template
            template = self.workflow_templates.get(workflow_type)
            if not template:
                raise ValueError(f"Unknown workflow type: {workflow_type}")
                
            # Set initial steps
            state.next_steps = template["steps"]
            
            # Execute workflow steps
            while state.next_steps and not state.error:
                current_step = state.next_steps[0]
                state.current_step = current_step
                state.status = "in_progress"
                
                # Check dependencies
                dependencies = template["dependencies"].get(current_step, [])
                if not all(dep in state.steps_completed for dep in dependencies):
                    state.error = f"Dependencies not met for step {current_step}"
                    break
                
                # Execute step
                try:
                    await self._execute_step(state, current_step, input_data)
                    state.steps_completed.append(current_step)
                    state.next_steps.pop(0)
                    state.progress = len(state.steps_completed) / len(template["steps"])
                except Exception as e:
                    state.error = f"Error in step {current_step}: {str(e)}"
                    break
                
            # Update final state
            if not state.error:
                state.status = "completed"
            else:
                state.status = "failed"
                
            state.updated_at = datetime.utcnow()
            return state
            
        except Exception as e:
            logger.error(f"Error executing workflow: {str(e)}", exc_info=True)
            self.error_handler.handle_error(e, {"workflow_type": workflow_type})
            raise
            
    async def _execute_step(
        self,
        state: WorkflowState,
        step: str,
        input_data: Dict[str, Any]
    ) -> None:
        """Execute a single workflow step."""
        try:
            # Create step execution prompt
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=f"""You are a workflow execution expert. Execute the following step:
                Step: {step}
                Workflow Type: {state.workflow_type}
                Current State: {state.dict()}
                Input Data: {input_data}
                
                Provide clear instructions for executing this step."""),
                HumanMessage(content="Execute the step and provide the result.")
            ])
            
            # Get step execution plan
            chain = prompt | self.llm_manager.llm | StrOutputParser()
            execution_plan = await chain.ainvoke({})
            
            # Execute step based on type
            if step == "validate_input":
                await self._validate_input(state, input_data)
            elif step == "extract_schema":
                await self._extract_schema(state, input_data)
            elif step == "check_quality":
                await self._check_quality(state, input_data)
            elif step == "update_catalog":
                await self._update_catalog(state, input_data)
            elif step == "create_table":
                await self._create_table(state, input_data)
            else:
                raise ValueError(f"Unknown step: {step}")
                
        except Exception as e:
            logger.error(f"Error executing step {step}: {str(e)}", exc_info=True)
            self.error_handler.handle_error(e, {
                "workflow_id": state.workflow_id,
                "step": step
            })
            raise
            
    async def _validate_input(self, state: WorkflowState, input_data: Dict[str, Any]) -> None:
        """Validate input data for the workflow."""
        try:
            # Validate file presence and format
            if "file" not in input_data:
                raise ValueError("No file provided in input data")
                
            file = input_data["file"]
            if not hasattr(file, "filename"):
                raise ValueError("Invalid file object")
                
            # Validate file format
            file_format = file.filename.split(".")[-1].lower()
            if file_format not in ["csv", "json", "parquet", "xlsx", "xls"]:
                raise ValueError(f"Unsupported file format: {file_format}")
                
            # Validate table name if provided
            if "table_name" in input_data:
                table_name = input_data["table_name"]
                if not isinstance(table_name, str) or not table_name:
                    raise ValueError("Invalid table name")
                    
            # Update state with validation results
            state.agent_states["validation"] = {
                "file_format": file_format,
                "file_size": len(await file.read()),
                "validated_at": datetime.utcnow().isoformat()
            }
            
            # Reset file pointer
            await file.seek(0)
            
        except Exception as e:
            logger.error(f"Input validation failed: {str(e)}")
            raise
            
    async def _extract_schema(self, state: WorkflowState, input_data: Dict[str, Any]) -> None:
        """Extract schema from input data."""
        try:
            file = input_data["file"]
            file_format = file.filename.split(".")[-1].lower()
            
            # Extract schema using SchemaExtractorAgent
            schema_result = await self.schema_agent.extract_schema(
                file=file,
                file_format=file_format
            )
            
            # Store schema in state
            state.agent_states["schema"] = {
                "schema": schema_result["schema"],
                "version": schema_result.get("version"),
                "extracted_at": datetime.utcnow().isoformat()
            }
            
            # Reset file pointer
            await file.seek(0)
            
        except Exception as e:
            logger.error(f"Schema extraction failed: {str(e)}")
            raise
            
    async def _check_quality(self, state: WorkflowState, input_data: Dict[str, Any]) -> None:
        """Check data quality metrics."""
        try:
            file = input_data["file"]
            schema = state.agent_states["schema"]["schema"]
            
            # Check quality using DataQualityAgent
            quality_result = await self.quality_agent.check_quality(
                file=file,
                schema=schema
            )
            
            # Store quality metrics in state
            state.agent_states["quality"] = {
                "metrics": quality_result["metrics"],
                "checked_at": datetime.utcnow().isoformat()
            }
            
            # Reset file pointer
            await file.seek(0)
            
        except Exception as e:
            logger.error(f"Quality check failed: {str(e)}")
            raise
            
    async def _update_catalog(self, state: WorkflowState, input_data: Dict[str, Any]) -> None:
        """Update catalog with new information."""
        try:
            file = input_data["file"]
            table_name = input_data.get("table_name", file.filename.split(".")[0])
            schema = state.agent_states["schema"]["schema"]
            quality_metrics = state.agent_states["quality"]["metrics"]
            
            # Update catalog using CatalogAgent
            catalog_result = await self.catalog_agent.update_catalog(
                file_info={
                    "file_name": file.filename,
                    "table_name": table_name,
                    "format": file.filename.split(".")[-1].lower(),
                    "size": len(await file.read()),
                    "created_at": datetime.utcnow().isoformat()
                },
                schema=schema,
                quality_metrics=quality_metrics
            )
            
            # Store catalog update in state
            state.agent_states["catalog"] = {
                "update_result": catalog_result,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Reset file pointer
            await file.seek(0)
            
        except Exception as e:
            logger.error(f"Catalog update failed: {str(e)}")
            raise
            
    async def _create_table(self, state: WorkflowState, input_data: Dict[str, Any]) -> None:
        """Create database table."""
        try:
            file = input_data["file"]
            table_name = input_data.get("table_name", file.filename.split(".")[0])
            schema = state.agent_states["schema"]["schema"]
            
            # Create table using CatalogAgent
            table_result = await self.catalog_agent.create_table(
                table_name=table_name,
                schema=schema,
                file=file
            )
            
            # Store table creation in state
            state.agent_states["table"] = {
                "creation_result": table_result,
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Reset file pointer
            await file.seek(0)
            
        except Exception as e:
            logger.error(f"Table creation failed: {str(e)}")
            raise 