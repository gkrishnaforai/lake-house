from typing import Dict, Any
from src.core.agents.etl_orchestration_agent import (
    ETLNode, NodeType, WorkflowState, ETLOrchestrationAgent
)
from src.core.agents.tools.file_loading_tools import (
    FileLoadingTool,
    SchemaInferenceTool,
    TableValidationTool,
    FileInfo,
    IcebergTableInfo
)
import os
import boto3
import logging

logger = logging.getLogger(__name__)

class FileLoadingWorkflow(ETLOrchestrationAgent):
    """Workflow for loading files into Iceberg tables."""
    
    def __init__(self, s3_client=None, glue_client=None, dynamodb_client=None):
        super().__init__(dynamodb_client)
        
        # Initialize AWS clients with credentials
        self.s3_client = s3_client or boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        
        self.glue_client = glue_client or boto3.client(
            'glue',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        
        # Initialize tools with AWS clients
        self.file_loading_tool = FileLoadingTool(
            self.s3_client, self.glue_client
        )
        self.schema_inference_tool = SchemaInferenceTool()
        self.table_validation_tool = TableValidationTool(self.glue_client)

        # Initialize nodes with type and required fields
        self.nodes = {
            "schema_inference": {
                "type": "schema_inference",
                "required_fields": ["file_path"]
            },
            "file_loading": {
                "type": "file_loading",
                "required_fields": [
                    "file_path", "table_name", "database_name",
                    "s3_location", "schema"
                ]
            },
            "table_validation": {
                "type": "table_validation",
                "required_fields": ["database_name", "table_name"]
            }
        }
        
    def create_workflow(self, file_path: str, table_name: str, database_name: str,
                       s3_location: str) -> WorkflowState:
        """Create a workflow for loading a file into an Iceberg table."""
        
        # Create nodes
        schema_node = ETLNode(
            node_id="schema_inference",
            node_type=NodeType.SOURCE,
            tools=[self.schema_inference_tool],
            description="Infer schema from file"
        )
        
        loading_node = ETLNode(
            node_id="file_loading",
            node_type=NodeType.TRANSFORMATION,
            tools=[self.file_loading_tool],
            description="Load file into S3 and create Iceberg table"
        )
        
        validation_node = ETLNode(
            node_id="table_validation",
            node_type=NodeType.QUALITY,
            tools=[self.table_validation_tool],
            description="Validate Iceberg table creation"
        )
        
        # Create workflow state
        workflow_state = WorkflowState(
            workflow_id=f"file_loading_{table_name}",
            nodes={
                "schema_inference": schema_node,
                "file_loading": loading_node,
                "table_validation": validation_node
            },
            node_order=["schema_inference", "file_loading", "table_validation"],
            context={
                "file_path": file_path,
                "table_name": table_name,
                "database_name": database_name,
                "s3_location": s3_location
            }
        )
        
        return workflow_state
    
    async def execute_node(self, node_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific node in the workflow."""
        node = self.nodes.get(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found in workflow")
        
        # Validate required fields
        required_fields = node.get("required_fields", [])
        missing_fields = [field for field in required_fields if field not in context]
        if missing_fields:
            raise ValueError(f"Missing required fields for node {node_id}: {missing_fields}")
        
        # Execute node based on type
        if node["type"] == "schema_inference":
            # Infer schema from file
            schema = self.schema_inference_tool._run(context["file_path"])
            context["schema"] = schema
            return {"status": "success", "schema": schema}
        
        elif node["type"] == "file_loading":
            # Create table with inferred schema if it doesn't exist
            file_name = os.path.basename(context["file_path"])
            file_type = os.path.splitext(context["file_path"])[1].lstrip('.')
            
            # Construct S3 path using the s3_location from context
            s3_path = f"{context['s3_location']}/{file_name}"
            
            file_info = FileInfo(
                file_name=file_name,
                file_type=file_type,
                s3_path=s3_path,
                size=os.path.getsize(context["file_path"]),
                table_schema=context["schema"]
            )
            table_info = IcebergTableInfo(
                table_name=context["table_name"],
                database_name=context["database_name"],
                s3_location=context["s3_location"],
                table_schema=context["schema"]
            )
            result = self.file_loading_tool._run(file_info, table_info)
            if result["status"] == "error":
                raise ValueError(f"Error loading file: {result['message']}")
            return result
        
        elif node["type"] == "table_validation":
            # Validate table creation
            result = self.table_validation_tool._run(
                context["database_name"],
                context["table_name"]
            )
            if result["status"] == "error":
                raise ValueError(f"Error validating table: {result['message']}")
            return result
        
        else:
            raise ValueError(f"Unknown node type: {node['type']}") 