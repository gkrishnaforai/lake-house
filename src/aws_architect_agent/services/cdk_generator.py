from typing import Dict

from aws_cdk import (
    App,
    Stack,
    aws_glue as glue,
    aws_lambda as lambda_,
    aws_s3 as s3,
    aws_iam as iam,
)
from constructs import Construct

from aws_architect_agent.models.base import (
    Architecture,
    Component,
    ComponentType,
)
from aws_architect_agent.utils.logging import get_logger

logger = get_logger(__name__)


class AWSArchitectureStack(Stack):
    """AWS CDK stack for the architecture."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        architecture: Architecture,
        **kwargs,
    ) -> None:
        """Initialize the stack.

        Args:
            scope: Parent construct
            construct_id: Unique identifier for the stack
            architecture: Architecture to generate
            **kwargs: Additional stack properties
        """
        super().__init__(scope, construct_id, **kwargs)
        self.architecture = architecture
        self._create_resources()

    def _create_resources(self) -> None:
        """Create AWS resources based on the architecture."""
        for component in self.architecture.components:
            if component.type == ComponentType.DATA_SOURCE:
                self._create_data_source(component)
            elif component.type == ComponentType.PROCESSING:
                self._create_processing(component)
            elif component.type == ComponentType.STORAGE:
                self._create_storage(component)
            elif component.type == ComponentType.MONITORING:
                self._create_monitoring(component)

    def _create_data_source(self, component: Component) -> None:
        """Create a data source resource.

        Args:
            component: Data source component
        """
        source_type = component.configuration.get("source_type", "")
        if source_type == "s3":
            self._create_s3_bucket(component)
        elif source_type == "rds":
            self._create_rds_instance(component)
        elif source_type == "dynamodb":
            self._create_dynamodb_table(component)

    def _create_processing(self, component: Component) -> None:
        """Create a processing resource.

        Args:
            component: Processing component
        """
        processing_type = component.configuration.get("processing_type", "")
        if processing_type == "glue":
            self._create_glue_job(component)
        elif processing_type == "lambda":
            self._create_lambda_function(component)
        elif processing_type == "emr":
            self._create_emr_cluster(component)

    def _create_storage(self, component: Component) -> None:
        """Create a storage resource.

        Args:
            component: Storage component
        """
        storage_type = component.configuration.get("storage_type", "")
        if storage_type == "s3":
            self._create_s3_bucket(component)
        elif storage_type == "redshift":
            self._create_redshift_cluster(component)
        elif storage_type == "dynamodb":
            self._create_dynamodb_table(component)

    def _create_monitoring(self, component: Component) -> None:
        """Create a monitoring resource.

        Args:
            component: Monitoring component
        """
        monitoring_type = component.configuration.get("monitoring_type", "")
        if monitoring_type == "cloudwatch":
            self._create_cloudwatch_alarms(component)
        elif monitoring_type == "xray":
            self._create_xray_tracing(component)

    def _create_s3_bucket(self, component: Component) -> None:
        """Create an S3 bucket.

        Args:
            component: Component configuration
        """
        bucket = s3.Bucket(
            self,
            f"{component.id}-bucket",
            bucket_name=component.name.lower(),
            versioned=True,
        )
        logger.info(f"Created S3 bucket: {bucket.bucket_name}")

    def _create_lambda_function(self, component: Component) -> None:
        """Create a Lambda function.

        Args:
            component: Component configuration
        """
        function = lambda_.Function(
            self,
            f"{component.id}-function",
            function_name=component.name,
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="index.handler",
            code=lambda_.Code.from_asset("lambda"),
        )
        logger.info(f"Created Lambda function: {function.function_name}")

    def _create_glue_job(self, component: Component) -> None:
        """Create a Glue job.

        Args:
            component: Component configuration
        """
        job = glue.CfnJob(
            self,
            f"{component.id}-job",
            name=component.name,
            role=iam.Role.from_role_arn(
                self,
                f"{component.id}-role",
                role_arn="arn:aws:iam::123456789012:role/GlueServiceRole",
            ).role_arn,
            command=glue.CfnJob.JobCommandProperty(
                name="glueetl",
                script_location="s3://path/to/script.py",
            ),
        )
        logger.info(f"Created Glue job: {job.name}")


class CDKGenerator:
    """Service for generating AWS CDK infrastructure code."""

    def __init__(self, architecture: Architecture) -> None:
        """Initialize the CDK generator.

        Args:
            architecture: Architecture to generate
        """
        self.architecture = architecture
        self.app = App()
        self.stack = AWSArchitectureStack(
            self.app,
            f"{architecture.name}-stack",
            architecture=architecture,
        )

    def synthesize(self) -> None:
        """Synthesize the CDK app."""
        self.app.synth()

    def generate_template(self) -> Dict:
        """Generate the CloudFormation template.

        Returns:
            CloudFormation template as a dictionary
        """
        return self.app.synth().get_stack_by_name(
            self.stack.stack_name
        ).template 