from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
import boto3
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
import logging
import os
from src.core.agents.tools.log_manager import LogManager
import traceback

logger = logging.getLogger(__name__)

LOG_BUCKET = os.getenv("AWS_S3_BUCKET", "your-bucket-name")
LOG_REGION = os.getenv("AWS_REGION", "us-east-1")
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_RETENTION_MINUTES = int(os.getenv("LOG_RETENTION_MINUTES", "10"))
log_manager = LogManager(
    bucket=LOG_BUCKET,
    region=LOG_REGION,
    log_dir=LOG_DIR,
    retention_minutes=LOG_RETENTION_MINUTES
)

def log_structured(event_type, data):
    """Create a log entry and write it to the log file."""
    # Create a copy of the data to avoid modifying the original
    log_data = data.copy()
    
    # Handle context if it exists
    if "context" in log_data:
        context_copy = log_data["context"].copy()
        # Remove DataFrame from context if it exists
        if "dataframe" in context_copy:
            del context_copy["dataframe"]
        log_data["context"] = context_copy
    
    # Create log entry
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        **log_data
    }
    
    # Write log entry
    log_manager.write_log(log_entry)

class WorkflowStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PAUSED = "PAUSED"

class NodeType(str, Enum):
    SOURCE = "SOURCE"
    TRANSFORMATION = "TRANSFORMATION"
    DESTINATION = "DESTINATION"
    QUALITY = "QUALITY"
    METADATA = "METADATA"

class NodeStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class WorkflowState(BaseModel):
    workflow_id: str
    nodes: Dict[str, 'ETLNode']
    node_order: List[str]
    context: Dict[str, Any]
    status: str = WorkflowStatus.PENDING
    current_node: Optional[str] = None
    results: Dict[str, Any] = Field(default_factory=dict)

class ETLNode(BaseModel):
    node_id: str
    node_type: str
    tools: List[BaseTool]
    description: str
    status: str = NodeStatus.PENDING
    result: Optional[Dict[str, Any]] = None

class ETLOrchestrationAgent:
    def __init__(self, dynamodb_client=None):
        self.dynamodb_client = dynamodb_client or boto3.client('dynamodb')
        self.workflow_states = {}
    
    async def execute_workflow(self, workflow_state: WorkflowState) -> Dict[str, Any]:
        """Execute a workflow."""
        workflow_id = workflow_state.workflow_id
        log_structured("workflow_start", {
            "workflow_id": workflow_id,
            "node_order": workflow_state.node_order,
            "context": workflow_state.context
        })
        try:
            workflow_state.status = WorkflowStatus.RUNNING
            self.workflow_states[workflow_id] = workflow_state
            await self.save_workflow_state(workflow_state)
            for node_id in workflow_state.node_order:
                node = workflow_state.nodes[node_id]
                workflow_state.current_node = node_id
                node.status = NodeStatus.RUNNING
                await self.save_workflow_state(workflow_state)
                try:
                    log_structured("node_start", {
                        "workflow_id": workflow_id,
                        "node_id": node_id,
                        "node_type": node.node_type,
                        "context": workflow_state.context
                    })
                    result = await self.execute_node(node.node_id, workflow_state.context)
                    node.result = result
                    node.status = NodeStatus.COMPLETED
                    workflow_state.results[node_id] = result
                    log_structured("node_success", {
                        "workflow_id": workflow_id,
                        "node_id": node_id,
                        "node_type": node.node_type,
                        "result": result
                    })
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.error(f"Error executing node {node_id}: {str(e)}\nTraceback:\n{tb}", exc_info=True)
                    log_structured("node_failure", {
                        "workflow_id": workflow_id,
                        "node_id": node_id,
                        "node_type": node.node_type,
                        "exception": str(e),
                        "traceback": tb
                    })
                    node.status = NodeStatus.FAILED
                    workflow_state.status = WorkflowStatus.FAILED
                    await self.save_workflow_state(workflow_state)
                    log_structured("workflow_failure", {
                        "workflow_id": workflow_id,
                        "failed_node": node_id,
                        "exception": str(e),
                        "traceback": tb
                    })
                    return {
                        "status": "error",
                        "message": f"Error executing node {node_id}: {str(e)}\nTraceback:\n{tb}"
                    }
                await self.save_workflow_state(workflow_state)
            workflow_state.status = WorkflowStatus.COMPLETED
            await self.save_workflow_state(workflow_state)
            log_structured("workflow_success", {
                "workflow_id": workflow_id,
                "results": workflow_state.results
            })
            return {
                "status": "success",
                "workflow_id": workflow_id,
                "results": workflow_state.results
            }
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Error executing workflow: {str(e)}\nTraceback:\n{tb}", exc_info=True)
            log_structured("workflow_exception", {
                "workflow_id": workflow_id,
                "exception": str(e),
                "traceback": tb
            })
            return {
                "status": "error",
                "message": f"Error executing workflow: {str(e)}\nTraceback:\n{tb}"
            }
    
    async def execute_node(self, node_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a node in the workflow."""
        # This method should be implemented by specific workflow classes
        raise NotImplementedError
    
    async def save_workflow_state(self, workflow_state: WorkflowState):
        """Save workflow state to DynamoDB."""
        try:
            item = {
                "workflow_id": {"S": workflow_state.workflow_id},
                "status": {"S": workflow_state.status},
                "current_node": {"S": workflow_state.current_node or ""},
                "context": {"M": self._dict_to_dynamodb(workflow_state.context)},
                "results": {"M": self._dict_to_dynamodb(workflow_state.results)}
            }
            
            self.dynamodb_client.put_item(
                TableName="etl_workflow_states",
                Item=item
            )
            
        except Exception as e:
            logger.error(f"Error saving workflow state: {str(e)}")
            log_structured("workflow_state_save_error", {
                "workflow_id": workflow_state.workflow_id,
                "exception": str(e)
            })
            raise
    
    async def load_workflow_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """Load workflow state from DynamoDB."""
        try:
            response = self.dynamodb_client.get_item(
                TableName="etl_workflow_states",
                Key={"workflow_id": {"S": workflow_id}}
            )
            
            if "Item" not in response:
                return None
            
            item = response["Item"]
            
            # Convert DynamoDB format to Python dict
            context = self._dynamodb_to_dict(item["context"]["M"])
            results = self._dynamodb_to_dict(item["results"]["M"])
            
            # Recreate workflow state
            workflow_state = WorkflowState(
                workflow_id=item["workflow_id"]["S"],
                nodes={},  # TODO: Recreate nodes from saved state
                node_order=[],  # TODO: Recreate node order from saved state
                context=context,
                status=item["status"]["S"],
                current_node=item["current_node"]["S"] or None,
                results=results
            )
            
            return workflow_state
            
        except Exception as e:
            logger.error(f"Error loading workflow state: {str(e)}")
            log_structured("workflow_state_load_error", {
                "workflow_id": workflow_id,
                "exception": str(e)
            })
            return None
    
    def _dict_to_dynamodb(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Python dict to DynamoDB format."""
        result = {}
        for k, v in d.items():
            if isinstance(v, str):
                result[k] = {"S": v}
            elif isinstance(v, int):
                result[k] = {"N": str(v)}
            elif isinstance(v, float):
                result[k] = {"N": str(v)}
            elif isinstance(v, bool):
                result[k] = {"BOOL": v}
            elif isinstance(v, dict):
                result[k] = {"M": self._dict_to_dynamodb(v)}
            elif isinstance(v, list):
                result[k] = {"L": [self._dict_to_dynamodb({"item": x})["item"] for x in v]}
            elif v is None:
                result[k] = {"NULL": True}
            else:
                result[k] = {"S": str(v)}
        return result
    
    def _dynamodb_to_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Convert DynamoDB format to Python dict."""
        result = {}
        for k, v in d.items():
            if "S" in v:
                result[k] = v["S"]
            elif "N" in v:
                result[k] = float(v["N"])
            elif "BOOL" in v:
                result[k] = v["BOOL"]
            elif "M" in v:
                result[k] = self._dynamodb_to_dict(v["M"])
            elif "L" in v:
                result[k] = [self._dynamodb_to_dict({"item": x})["item"] for x in v["L"]]
            elif "NULL" in v:
                result[k] = None
        return result

    @staticmethod
    def periodic_s3_log_upload():
        log_manager.upload_logs_to_s3() 