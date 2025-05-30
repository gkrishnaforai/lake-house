from typing import Dict, Any, Optional
from datetime import datetime
import logging
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor

from .base_agent import BaseAgent
from ..aws.aws_service_manager import AWSServiceManager

logger = logging.getLogger(__name__)

class InfrastructureAgent(BaseAgent):
    """Agent responsible for generating Terraform infrastructure code for ETL pipelines."""

    def __init__(
        self,
        agent_id: str,
        aws_manager: Optional[AWSServiceManager] = None,
        region_name: str = "us-east-1",
        llm_model: str = "gpt-4"
    ):
        """Initialize the infrastructure agent.
        
        Args:
            agent_id: Unique identifier for the agent
            aws_manager: Optional AWS service manager instance
            region_name: AWS region name
            llm_model: LLM model to use for infrastructure generation
        """
        super().__init__(agent_id)
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
        self.aws_manager = (
            aws_manager or AWSServiceManager(region_name=region_name)
        )
        self.llm_model = llm_model
        self.workflow_graph = None
        self.tool_executor = None

    async def initialize(self) -> None:
        """Initialize the infrastructure agent and setup LangGraph workflow."""
        try:
            # Initialize tools for the workflow
            tools = [
                {
                    "name": "analyze_requirements",
                    "description": "Analyze ETL pipeline requirements",
                    "function": self._analyze_requirements
                },
                {
                    "name": "generate_terraform",
                    "description": "Generate Terraform configuration",
                    "function": self._generate_terraform
                },
                {
                    "name": "validate_terraform",
                    "description": "Validate Terraform configuration",
                    "function": self._validate_terraform
                },
                {
                    "name": "optimize_terraform",
                    "description": "Optimize Terraform configuration",
                    "function": self._optimize_terraform
                }
            ]
            
            # Create tool executor
            self.tool_executor = ToolExecutor(tools)
            
            # Define workflow nodes
            def create_workflow_nodes():
                return {
                    "analyze": self._analyze_requirements,
                    "generate": self._generate_terraform,
                    "validate": self._validate_terraform,
                    "optimize": self._optimize_terraform
                }
            
            # Create workflow graph
            self.workflow_graph = StateGraph(nodes=create_workflow_nodes())
            
            # Add edges
            self.workflow_graph.add_edge("analyze", "generate")
            self.workflow_graph.add_edge("generate", "validate")
            self.workflow_graph.add_edge("validate", "optimize")
            self.workflow_graph.add_edge("optimize", END)
            
            # Set entry point
            self.workflow_graph.set_entry_point("analyze")
            
            # Compile workflow
            self.workflow = self.workflow_graph.compile()
            
            await self.update_state({
                "status": "ready",
                "last_updated": datetime.utcnow().isoformat(),
                "metadata": {
                    "capabilities": [
                        "terraform_generation",
                        "infrastructure_validation",
                        "cost_optimization"
                    ]
                }
            })
            self.logger.info(f"Infrastructure Agent {self.agent_id} initialized")
            
        except Exception as e:
            await self.handle_error(e)
            raise

    async def _analyze_requirements(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ETL pipeline requirements and determine infrastructure needs.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with infrastructure requirements
        """
        try:
            requirements = state.get("requirements", {})
            
            # Analyze data sources
            data_sources = requirements.get("data_sources", [])
            source_infra = []
            for source in data_sources:
                source_type = source.get("type")
                if source_type == "s3":
                    source_infra.append({
                        "type": "aws_s3_bucket",
                        "config": {
                            "bucket": source.get("name"),
                            "versioning": True
                        }
                    })
                elif source_type == "rds":
                    source_infra.append({
                        "type": "aws_db_instance",
                        "config": {
                            "identifier": source.get("name"),
                            "engine": source.get("engine"),
                            "instance_class": source.get("instance_class")
                        }
                    })
            
            # Analyze processing requirements
            processing = requirements.get("processing", {})
            processing_infra = []
            if processing.get("type") == "glue":
                processing_infra.append({
                    "type": "aws_glue_job",
                    "config": {
                        "name": processing.get("name"),
                        "role_arn": processing.get("role_arn"),
                        "command": {
                            "name": "glueetl",
                            "script_location": processing.get("script_location")
                        }
                    }
                })
            elif processing.get("type") == "lambda":
                processing_infra.append({
                    "type": "aws_lambda_function",
                    "config": {
                        "function_name": processing.get("name"),
                        "runtime": "python3.9",
                        "handler": "index.handler",
                        "filename": "lambda.zip"
                    }
                })
            
            # Analyze storage requirements
            storage = requirements.get("storage", {})
            storage_infra = []
            if storage.get("type") == "s3":
                storage_infra.append({
                    "type": "aws_s3_bucket",
                    "config": {
                        "bucket": storage.get("name"),
                        "versioning": True
                    }
                })
            elif storage.get("type") == "redshift":
                storage_infra.append({
                    "type": "aws_redshift_cluster",
                    "config": {
                        "cluster_identifier": storage.get("name"),
                        "node_type": storage.get("node_type"),
                        "master_username": storage.get("master_username"),
                        "master_password": storage.get("master_password"),
                        "database_name": storage.get("database_name")
                    }
                })
            
            # Update state with infrastructure requirements
            state["infrastructure"] = {
                "data_sources": source_infra,
                "processing": processing_infra,
                "storage": storage_infra
            }
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error analyzing requirements: {str(e)}")
            raise

    async def _generate_terraform(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Terraform configuration based on infrastructure requirements.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with Terraform configuration
        """
        try:
            infrastructure = state.get("infrastructure", {})
            
            # Generate provider configuration
            provider_config = {
                "terraform": {
                    "required_providers": {
                        "aws": {
                            "source": "hashicorp/aws",
                            "version": "~> 5.0"
                        }
                    }
                },
                "provider": {
                    "aws": {
                        "region": self.aws_manager.region_name
                    }
                }
            }
            
            # Generate resource configurations
            resources = {}
            
            # Add data source resources
            for source in infrastructure.get("data_sources", []):
                resource_type = source["type"]
                if resource_type not in resources:
                    resources[resource_type] = []
                resources[resource_type].append(source["config"])
            
            # Add processing resources
            for proc in infrastructure.get("processing", []):
                resource_type = proc["type"]
                if resource_type not in resources:
                    resources[resource_type] = []
                resources[resource_type].append(proc["config"])
            
            # Add storage resources
            for storage in infrastructure.get("storage", []):
                resource_type = storage["type"]
                if resource_type not in resources:
                    resources[resource_type] = []
                resources[resource_type].append(storage["config"])
            
            # Combine configurations
            terraform_config = {
                **provider_config,
                "resource": resources
            }
            
            # Update state with Terraform configuration
            state["terraform"] = terraform_config
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error generating Terraform: {str(e)}")
            raise

    async def _validate_terraform(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Terraform configuration.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with validation results
        """
        try:
            terraform_config = state.get("terraform", {})
            
            # Validate provider configuration
            if "provider" not in terraform_config:
                raise ValueError("Missing provider configuration")
            
            # Validate resource configurations
            if "resource" not in terraform_config:
                raise ValueError("Missing resource configurations")
            
            resources = terraform_config["resource"]
            validation_results = []
            
            for resource_type, resource_configs in resources.items():
                for config in resource_configs:
                    # Validate required fields
                    if resource_type == "aws_s3_bucket":
                        if "bucket" not in config:
                            validation_results.append(
                                f"Missing bucket name in {resource_type}"
                            )
                    elif resource_type == "aws_glue_job":
                        if "name" not in config or "role_arn" not in config:
                            validation_results.append(
                                "Missing required fields in {resource_type}"
                            )
                    elif resource_type == "aws_lambda_function":
                        if "function_name" not in config:
                            validation_results.append(
                                f"Missing function name in {resource_type}"
                            )
                    elif resource_type == "aws_redshift_cluster":
                        required_fields = [
                            "cluster_identifier",
                            "node_type",
                            "master_username",
                            "master_password",
                            "database_name"
                        ]
                        missing_fields = [
                            field for field in required_fields
                            if field not in config
                        ]
                        if missing_fields:
                            validation_results.append(
                                f"Missing fields {missing_fields} in "
                                f"{resource_type}"
                            )
            
            # Update state with validation results
            state["validation"] = {
                "is_valid": len(validation_results) == 0,
                "results": validation_results
            }
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error validating Terraform: {str(e)}")
            raise

    async def _optimize_terraform(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize Terraform configuration for cost and performance.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with optimized Terraform configuration
        """
        try:
            terraform_config = state.get("terraform", {})
            resources = terraform_config.get("resource", {})
            optimizations = []
            
            # Optimize S3 buckets
            if "aws_s3_bucket" in resources:
                for bucket_config in resources["aws_s3_bucket"]:
                    # Add lifecycle rules for cost optimization
                    if "lifecycle_rule" not in bucket_config:
                        bucket_config["lifecycle_rule"] = [{
                            "enabled": True,
                            "transition": [{
                                "days": 90,
                                "storage_class": "STANDARD_IA"
                            }]
                        }]
                        optimizations.append(
                            "Added lifecycle rules to S3 bucket"
                        )
            
            # Optimize Lambda functions
            if "aws_lambda_function" in resources:
                for lambda_config in resources["aws_lambda_function"]:
                    # Add memory and timeout configurations
                    if "memory_size" not in lambda_config:
                        lambda_config["memory_size"] = 128
                        optimizations.append(
                            "Set default memory size for Lambda function"
                        )
                    if "timeout" not in lambda_config:
                        lambda_config["timeout"] = 30
                        optimizations.append(
                            "Set default timeout for Lambda function"
                        )
            
            # Optimize Redshift clusters
            if "aws_redshift_cluster" in resources:
                for cluster_config in resources["aws_redshift_cluster"]:
                    # Add automated snapshot retention
                    if "automated_snapshot_retention_period" not in cluster_config:
                        cluster_config["automated_snapshot_retention_period"] = 7
                        optimizations.append(
                            "Added automated snapshot retention for Redshift cluster"
                        )
            
            # Update state with optimizations
            state["optimizations"] = optimizations
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error optimizing Terraform: {str(e)}")
            raise

    async def generate_infrastructure(
        self,
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate infrastructure code for an ETL pipeline.
        
        Args:
            requirements: ETL pipeline requirements
            
        Returns:
            Generated infrastructure code and metadata
        """
        try:
            # Initialize workflow state
            state = {
                "requirements": requirements,
                "status": "in_progress",
                "start_time": datetime.utcnow().isoformat()
            }
            
            # Execute workflow
            final_state = await self.workflow.ainvoke(state)
            
            # Prepare response
            response = {
                "terraform": final_state.get("terraform", {}),
                "validation": final_state.get("validation", {}),
                "optimizations": final_state.get("optimizations", []),
                "metadata": {
                    "status": "completed",
                    "start_time": state["start_time"],
                    "end_time": datetime.utcnow().isoformat()
                }
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating infrastructure: {str(e)}")
            raise 