from typing import Dict, List

from aws_architect_agent.models.base import (
    Architecture,
    Component,
    ComponentType,
)
from aws_architect_agent.utils.logging import get_logger

logger = get_logger(__name__)


class TerraformGenerator:
    """Service for generating Terraform infrastructure code."""

    def __init__(self, architecture: Architecture) -> None:
        """Initialize the Terraform generator.

        Args:
            architecture: Architecture to generate
        """
        self.architecture = architecture
        self.resources: Dict[str, List[Dict]] = {
            "aws_s3_bucket": [],
            "aws_lambda_function": [],
            "aws_glue_job": [],
            "aws_redshift_cluster": [],
            "aws_dynamodb_table": [],
            "aws_cloudwatch_metric_alarm": [],
            "aws_xray_sampling_rule": [],
        }

    def generate(self) -> Dict:
        """Generate Terraform configuration.

        Returns:
            Terraform configuration as a dictionary
        """
        for component in self.architecture.components:
            if component.type == ComponentType.DATA_SOURCE:
                self._generate_data_source(component)
            elif component.type == ComponentType.PROCESSING:
                self._generate_processing(component)
            elif component.type == ComponentType.STORAGE:
                self._generate_storage(component)
            elif component.type == ComponentType.MONITORING:
                self._generate_monitoring(component)

        return {
            "terraform": {
                "required_providers": {
                    "aws": {
                        "source": "hashicorp/aws",
                        "version": "~> 5.0",
                    }
                }
            },
            "provider": {
                "aws": {
                    "region": "us-east-1",
                }
            },
            "resource": self.resources,
        }

    def _generate_data_source(self, component: Component) -> None:
        """Generate data source resources.

        Args:
            component: Data source component
        """
        source_type = component.configuration.get("source_type", "")
        if source_type == "s3":
            self._generate_s3_bucket(component)
        elif source_type == "rds":
            self._generate_rds_instance(component)
        elif source_type == "dynamodb":
            self._generate_dynamodb_table(component)

    def _generate_processing(self, component: Component) -> None:
        """Generate processing resources.

        Args:
            component: Processing component
        """
        processing_type = component.configuration.get("processing_type", "")
        if processing_type == "glue":
            self._generate_glue_job(component)
        elif processing_type == "lambda":
            self._generate_lambda_function(component)
        elif processing_type == "emr":
            self._generate_emr_cluster(component)

    def _generate_storage(self, component: Component) -> None:
        """Generate storage resources.

        Args:
            component: Storage component
        """
        storage_type = component.configuration.get("storage_type", "")
        if storage_type == "s3":
            self._generate_s3_bucket(component)
        elif storage_type == "redshift":
            self._generate_redshift_cluster(component)
        elif storage_type == "dynamodb":
            self._generate_dynamodb_table(component)

    def _generate_monitoring(self, component: Component) -> None:
        """Generate monitoring resources.

        Args:
            component: Monitoring component
        """
        monitoring_type = component.configuration.get("monitoring_type", "")
        if monitoring_type == "cloudwatch":
            self._generate_cloudwatch_alarms(component)
        elif monitoring_type == "xray":
            self._generate_xray_tracing(component)

    def _generate_s3_bucket(self, component: Component) -> None:
        """Generate S3 bucket resource.

        Args:
            component: Component configuration
        """
        self.resources["aws_s3_bucket"].append({
            component.id: {
                "bucket": component.name.lower(),
                "versioning": {
                    "enabled": True,
                },
            }
        })
        logger.info(f"Generated S3 bucket resource: {component.id}")

    def _generate_lambda_function(self, component: Component) -> None:
        """Generate Lambda function resource.

        Args:
            component: Component configuration
        """
        self.resources["aws_lambda_function"].append({
            component.id: {
                "function_name": component.name,
                "runtime": "python3.9",
                "handler": "index.handler",
                "filename": "lambda.zip",
            }
        })
        logger.info(f"Generated Lambda function resource: {component.id}")

    def _generate_glue_job(self, component: Component) -> None:
        """Generate Glue job resource.

        Args:
            component: Component configuration
        """
        self.resources["aws_glue_job"].append({
            component.id: {
                "name": component.name,
                "role_arn": "arn:aws:iam::123456789012:role/GlueServiceRole",
                "command": {
                    "name": "glueetl",
                    "script_location": "s3://path/to/script.py",
                },
            }
        })
        logger.info(f"Generated Glue job resource: {component.id}")

    def _generate_redshift_cluster(self, component: Component) -> None:
        """Generate Redshift cluster resource.

        Args:
            component: Component configuration
        """
        self.resources["aws_redshift_cluster"].append({
            component.id: {
                "cluster_identifier": component.name,
                "node_type": "dc2.large",
                "master_username": "admin",
                "master_password": "Redshift123",
                "database_name": "mydb",
            }
        })
        logger.info(f"Generated Redshift cluster resource: {component.id}")

    def _generate_dynamodb_table(self, component: Component) -> None:
        """Generate DynamoDB table resource.

        Args:
            component: Component configuration
        """
        self.resources["aws_dynamodb_table"].append({
            component.id: {
                "name": component.name,
                "hash_key": "id",
                "attribute": [
                    {
                        "name": "id",
                        "type": "S",
                    }
                ],
                "billing_mode": "PAY_PER_REQUEST",
            }
        })
        logger.info(f"Generated DynamoDB table resource: {component.id}")

    def _generate_cloudwatch_alarms(self, component: Component) -> None:
        """Generate CloudWatch alarm resources.

        Args:
            component: Component configuration
        """
        self.resources["aws_cloudwatch_metric_alarm"].append({
            component.id: {
                "alarm_name": f"{component.name}-alarm",
                "comparison_operator": "GreaterThanThreshold",
                "evaluation_periods": 2,
                "metric_name": "Errors",
                "namespace": "AWS/Lambda",
                "period": 300,
                "statistic": "Sum",
                "threshold": 1,
            }
        })
        logger.info(f"Generated CloudWatch alarm resource: {component.id}")

    def _generate_xray_tracing(self, component: Component) -> None:
        """Generate X-Ray sampling rule resource.

        Args:
            component: Component configuration
        """
        self.resources["aws_xray_sampling_rule"].append({
            component.id: {
                "rule_name": f"{component.name}-rule",
                "resource_arn": "*",
                "priority": 100,
                "fixed_rate": 0.1,
                "reservoir_size": 1,
                "service_name": "*",
                "service_type": "*",
                "host": "*",
                "http_method": "*",
                "url_path": "*",
                "version": 1,
            }
        })
        logger.info(f"Generated X-Ray sampling rule resource: {component.id}") 