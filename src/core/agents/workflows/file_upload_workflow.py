from typing import Dict, Any
from datetime import datetime
import os
import sys
import boto3
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.functions import current_timestamp, col, year, month, dayofmonth, lit
from src.core.agents.etl_orchestration_agent import (
    ETLNode, NodeType, WorkflowState, ETLOrchestrationAgent
)
import logging

logger = logging.getLogger(__name__)

# Constants for configuration values
DEFAULT_TIMEOUT_MS = "60000"
DEFAULT_RETRY_INTERVAL = "500"
DEFAULT_RETRY_LIMIT = "10"
DEFAULT_THROTTLE_INTERVAL = "100"
DEFAULT_THROTTLE_LIMIT = "20"

# S3A timeout configurations
S3A_TIMEOUT_CONFIGS = {
    "fs.s3a.connection.timeout": DEFAULT_TIMEOUT_MS,
    "fs.s3a.socket.timeout": DEFAULT_TIMEOUT_MS,
    "fs.s3a.connection.establish.timeout": DEFAULT_TIMEOUT_MS,
    "fs.s3a.connection.request.timeout": DEFAULT_TIMEOUT_MS,
    "fs.s3a.request.timeout": "0",
    "fs.s3a.connection.ttl": DEFAULT_TIMEOUT_MS,
    "fs.s3a.retry.interval": DEFAULT_RETRY_INTERVAL,
    "fs.s3a.retry.limit": DEFAULT_RETRY_LIMIT,
    "fs.s3a.retry.throttle.interval": DEFAULT_THROTTLE_INTERVAL,
    "fs.s3a.retry.throttle.limit": DEFAULT_THROTTLE_LIMIT,
    "fs.s3a.aws.client.request.timeout": DEFAULT_TIMEOUT_MS,
    "fs.s3a.aws.client.socket.timeout": DEFAULT_TIMEOUT_MS
}

# Critical S3A configurations
S3A_CRITICAL_CONFIGS = {
    "fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
    "fs.s3a.endpoint": "s3.amazonaws.com",
    "fs.s3a.aws.credentials.provider": 
        "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
    "fs.s3a.select.errors.include.sql": "false",
    "fs.s3a.connection.ssl.enabled": "true",
    "fs.s3a.impl.disable.cache": "true",
    "fs.impl.disable.cache": "true",
    "fs.s3a.impl.disable.cache.impl": "true",
    "fs.s3a.impl.disable.cache.impl.threads": "true"
}

class FileUploadWorkflow(ETLOrchestrationAgent):
    """Workflow for uploading files to S3 and creating Iceberg tables."""
    
    def __init__(self, s3_client=None, glue_client=None, dynamodb_client=None):
        super().__init__(dynamodb_client)
        self._init_clients(s3_client, glue_client)
        self._setup_env()
        self._create_policy_file()
        self._configure_spark()
        self._verify_s3a_config()
    
    def _init_clients(self, s3_client, glue_client):
        """Initialize AWS clients."""
        self.s3_client = s3_client or boto3.client('s3')
        self.glue_client = glue_client or boto3.client('glue')
    
    def _setup_env(self):
        """Set up environment variables."""
        os.environ['PYSPARK_PYTHON'] = sys.executable
        os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
        
        # Set S3A configurations in environment
        s3a_opts = " ".join([f"-D{k}={v}" for k, v in S3A_TIMEOUT_CONFIGS.items()])
        os.environ['HADOOP_OPTS'] = s3a_opts
    
    def _create_policy_file(self):
        """Create security policy file."""
        policy_content = """
grant {
    permission java.security.AllPermission;
};
"""
        with open("/tmp/spark.policy", "w") as f:
            f.write(policy_content)
    
    def _configure_spark(self):
        """Configure Spark session with S3A settings."""
        spark_conf = SparkConf() \
            .setAppName("FileUploadWorkflow") \
            .setMaster("local[*]") \
            .set("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \
            .set("spark.local.dir", "/tmp/spark-temp") \
            .set("spark.jars.packages", 
                 "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
            .set("spark.sql.debug.maxToStringFields", "100")
        
        # Set S3A configurations
        for key, value in S3A_TIMEOUT_CONFIGS.items():
            spark_conf.set(f"spark.hadoop.{key}", value)
        
        for key, value in S3A_CRITICAL_CONFIGS.items():
            spark_conf.set(f"spark.hadoop.{key}", value)
        
        # Set security options
        security_opts = "-Djava.security.manager=allow -Djava.security.policy=file:/tmp/spark.policy"
        spark_conf.set("spark.driver.extraJavaOptions", security_opts)
        spark_conf.set("spark.executor.extraJavaOptions", security_opts)
        
        # Create Spark session
        self.spark = SparkSession.builder.config(conf=spark_conf).getOrCreate()
        self.spark.sparkContext.setLogLevel("INFO")
    
    def _verify_s3a_config(self):
        """Verify S3A configuration is properly set."""
        hadoop_conf = self.spark._jsc.hadoopConfiguration()
        
        # Check critical configurations
        for key, expected_value in S3A_CRITICAL_CONFIGS.items():
            actual_value = hadoop_conf.get(key)
            if not actual_value:
                logger.warning(f"Missing critical configuration: {key}")
                hadoop_conf.set(key, expected_value)
        
        # Check timeout configurations
        for key, expected_value in S3A_TIMEOUT_CONFIGS.items():
            actual_value = hadoop_conf.get(key)
            if not actual_value or not actual_value.isdigit():
                logger.warning(f"Invalid timeout configuration: {key}={actual_value}")
                hadoop_conf.setInt(key, int(expected_value))
    
    def _handle_timeout_values(self, hadoop_conf):
        """Handle timeout values in Hadoop configuration."""
        props = hadoop_conf.getProps()
        
        for key in props.keySet():
            value = props.get(key)
            if (key.startswith('fs.s3a.') and 
                any(unit in str(value).lower() for unit in ['s', 'ms', 'm']) and 
                not str(value).isdigit() and
                'org.apache.hadoop' not in str(value)):
                
                if key in S3A_TIMEOUT_CONFIGS:
                    logger.warning(f"Replacing invalid timeout value - {key}: {value} with {S3A_TIMEOUT_CONFIGS[key]}")
                    hadoop_conf.setInt(key, int(S3A_TIMEOUT_CONFIGS[key]))
                elif key not in S3A_CRITICAL_CONFIGS:
                    logger.warning(f"Invalid Hadoop config value detected - {key}: {value}. Removing it.")
                    hadoop_conf.unset(key)
    
    def _verify_final_config(self, hadoop_conf):
        """Verify final configuration values."""
        props = hadoop_conf.getProps()
        for key in props.keySet():
            value = props.get(key)
            logger.info(f"Final config - {key}: {value}")
            
            if (key.startswith('fs.s3a.') and 
                any(unit in str(value).lower() for unit in ['s', 'ms', 'm']) and 
                not str(value).isdigit() and
                'org.apache.hadoop' not in str(value)):
                logger.error(f"ERROR: Found invalid duration value in final config - {key}: {value}")
                raise ValueError(f"Invalid duration value in final config: {key}={value}")
    
    def create_workflow(
        self,
        file_path: str,
        table_name: str,
        database_name: str,
        user_id: str,
        upload_id: str
    ) -> WorkflowState:
        """Create a workflow for uploading a file and creating an Iceberg table."""
        
        # Create nodes
        read_node = ETLNode(
            node_id="read_file",
            node_type=NodeType.SOURCE,
            tools=[],  # No tools needed, handled in execute_node
            description="Read input file from S3"
        )
        
        transform_node = ETLNode(
            node_id="transform_data",
            node_type=NodeType.TRANSFORMATION,
            tools=[],  # No tools needed, handled in execute_node
            description="Add audit fields and prepare for partitioning"
        )
        
        partition_node = ETLNode(
            node_id="partition_data",
            node_type=NodeType.TRANSFORMATION,
            tools=[],  # No tools needed, handled in execute_node
            description="Partition data by user_id and date"
        )
        
        save_node = ETLNode(
            node_id="save_parquet",
            node_type=NodeType.DESTINATION,
            tools=[],  # No tools needed, handled in execute_node
            description="Save data as Parquet with partitioning"
        )
        
        catalog_node = ETLNode(
            node_id="register_table",
            node_type=NodeType.METADATA,
            tools=[],  # No tools needed, handled in execute_node
            description="Register table in Glue catalog"
        )
        
        # Create workflow state
        workflow_state = WorkflowState(
            workflow_id=f"file_upload_{upload_id}",
            nodes={
                "read_file": read_node,
                "transform_data": transform_node,
                "partition_data": partition_node,
                "save_parquet": save_node,
                "register_table": catalog_node
            },
            node_order=[
                "read_file",
                "transform_data",
                "partition_data",
                "save_parquet",
                "register_table"
            ],
            context={
                "file_path": file_path,
                "table_name": table_name,
                "database_name": database_name,
                "user_id": user_id,
                "upload_id": upload_id,
                "bucket": os.getenv("AWS_S3_BUCKET", "your-bucket-name")
            }
        )
        
        return workflow_state
    
    async def execute_node(self, node_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific node in the workflow."""
        try:
            if node_id == "read_file":
                # Read XLSX file from S3
                bucket = context["bucket"]
                key = f"raw/uploads/{context['user_id']}/{context['upload_id']}/original.xlsx"
                
                # Download file to temp location
                temp_file = f"/tmp/{context['upload_id']}.xlsx"
                self.s3_client.download_file(bucket, key, temp_file)
                
                # Read with pandas
                df = pd.read_excel(temp_file)
                
                # Convert to Spark DataFrame
                spark_df = self.spark.createDataFrame(df)
                
                # Cleanup
                os.remove(temp_file)
                
                # Create result with DataFrame info
                result = {
                    "status": "success",
                    "dataframe_info": {
                        "num_rows": spark_df.count(),
                        "columns": spark_df.columns,
                        "schema": [str(field) for field in spark_df.schema.fields]
                    }
                }
                
                # Add DataFrame to context for next nodes
                context["dataframe"] = spark_df
                
                return result
            
            elif node_id == "transform_data":
                # Get DataFrame from previous node
                spark_df = context["dataframe"]
                
                # Add audit fields
                spark_df = spark_df \
                    .withColumn("updated_at", current_timestamp()) \
                    .withColumn("user_id", lit(context["user_id"]))
                
                # Create result with DataFrame info
                result = {
                    "status": "success",
                    "dataframe_info": {
                        "num_rows": spark_df.count(),
                        "columns": spark_df.columns,
                        "schema": [str(field) for field in spark_df.schema.fields]
                    }
                }
                
                # Update DataFrame in context
                context["dataframe"] = spark_df
                
                return result
            
            elif node_id == "partition_data":
                # Get DataFrame from previous node
                spark_df = context["dataframe"]
                
                # Extract date components
                spark_df = spark_df \
                    .withColumn("year", year(col("updated_at"))) \
                    .withColumn("month", month(col("updated_at"))) \
                    .withColumn("day", dayofmonth(col("updated_at")))
                
                # Create result with DataFrame info
                result = {
                    "status": "success",
                    "dataframe_info": {
                        "num_rows": spark_df.count(),
                        "columns": spark_df.columns,
                        "schema": [str(field) for field in spark_df.schema.fields]
                    }
                }
                
                # Update DataFrame in context
                context["dataframe"] = spark_df
                
                return result
            
            elif node_id == "save_parquet":
                # Get DataFrame from previous node
                spark_df = context["dataframe"]
                
                # Save as Parquet with partitioning
                output_path = f"s3a://{context['bucket']}/processed/parquet/{context['table_name']}"
                logger.info(f"Attempting to save Parquet files to: {output_path}")
                
                # Get current Hadoop configuration
                hadoop_conf = self.spark.sparkContext._jsc.hadoopConfiguration()
                
                # Handle timeout values
                self._handle_timeout_values(hadoop_conf)
                
                # Verify final configuration
                self._verify_final_config(hadoop_conf)
                
                # Save with explicit configuration
                writer = spark_df.write \
                    .partitionBy("user_id", "year", "month", "day") \
                    .mode("overwrite")
                
                # Execute write
                writer.parquet(output_path)
                
                return {
                    "status": "success",
                    "output_path": output_path,
                    "dataframe_info": {
                        "num_rows": spark_df.count(),
                        "columns": spark_df.columns,
                        "schema": [str(field) for field in spark_df.schema.fields]
                    }
                }
            
            elif node_id == "register_table":
                # Create table in Glue catalog
                table_location = f"s3a://{context['bucket']}/processed/parquet/{context['table_name']}"
                
                # Get schema from DataFrame
                schema = context["dataframe"].schema
                
                # Create table input
                table_input = {
                    'Name': context["table_name"],
                    'TableType': 'EXTERNAL_TABLE',
                    'Parameters': {
                        'classification': 'parquet',
                        'typeOfData': 'file'
                    },
                    'StorageDescriptor': {
                        'Location': table_location,
                        'InputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat',
                        'OutputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat',
                        'SerdeInfo': {
                            'SerializationLibrary': 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'
                        },
                        'Columns': [
                            {'Name': field.name, 'Type': field.dataType.simpleString()}
                            for field in schema.fields
                        ]
                    },
                    'PartitionKeys': [
                        {'Name': 'user_id', 'Type': 'string'},
                        {'Name': 'year', 'Type': 'int'},
                        {'Name': 'month', 'Type': 'int'},
                        {'Name': 'day', 'Type': 'int'}
                    ]
                }
                
                try:
                    # Create table
                    self.glue_client.create_table(
                        DatabaseName=context["database_name"],
                        TableInput=table_input
                    )
                except self.glue_client.exceptions.AlreadyExistsException:
                    # Update table if it exists
                    self.glue_client.update_table(
                        DatabaseName=context["database_name"],
                        TableInput=table_input
                    )
                
                return {
                    "status": "success",
                    "table_name": context["table_name"],
                    "database_name": context["database_name"],
                    "location": table_location,
                    "schema": [str(field) for field in schema.fields]
                }
            
            else:
                raise ValueError(f"Unknown node type: {node_id}")
                
        except Exception as e:
            logger.error(f"Error executing node {node_id}: {str(e)}")
            return {"status": "error", "message": str(e)} 