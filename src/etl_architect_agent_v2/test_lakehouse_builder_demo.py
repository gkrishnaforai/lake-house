"""Lakehouse Builder Demo.

This module demonstrates how to build a data lakehouse using AWS S3 and Glue Catalog.
"""

import asyncio
import logging
import os
import tempfile
from typing import Dict, List, Any
import pandas as pd
import boto3
from botocore.exceptions import ClientError
from core.llm.manager import LLMManager
from etl_architect_agent_v2.agents.schema_extractor.excel_schema_agent import (
    ExcelSchemaExtractorAgent
)
from etl_architect_agent_v2.agents.schema_extractor.sql_generator_agent import (
    SQLGeneratorAgent
)
from etl_architect_agent_v2.agents.database_config_agent import (
    DatabaseConfigAgent
)


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LakehouseBuilderDemo:
    """Demo class for building a data lakehouse."""
    
    def __init__(self):
        """Initialize the demo."""
        self.llm_manager = LLMManager()
        self.excel_agent = ExcelSchemaExtractorAgent(self.llm_manager)
        self.sql_agent = SQLGeneratorAgent(self.llm_manager)
        self.db_config_agent = DatabaseConfigAgent(self.llm_manager)
        
        # Initialize AWS clients
        self.s3_client = boto3.client('s3')
        self.glue_client = boto3.client('glue')
        self.athena_client = boto3.client('athena')
        
        # Load configuration from environment
        self.bucket_name = os.getenv('AWS_S3_BUCKET', 'my-data-lakehouse')
        self.database_name = os.getenv('AWS_GLUE_DATABASE', 'lakehouse_db')
        self.region = os.getenv('AWS_REGION', 'us-east-1')
        
        # Load AWS credentials
        self.aws_credentials = self.db_config_agent.load_credentials_from_env()
    
    async def setup_aws_resources(self):
        """Set up required AWS resources."""
        try:
            # Configure lakehouse using DatabaseConfigAgent
            config_result = await self.db_config_agent.configure_database(
                db_type='aws_s3_lakehouse',
                database=self.database_name,
                s3_bucket=self.bucket_name,
                region=self.region,
                additional_config={
                    'glue_catalog': True,
                    'athena_workgroup': 'primary',
                    'encryption': True,
                    'lifecycle_rules': True,
                    'versioning': True,
                    'aws_credentials': self.aws_credentials
                }
            )
            
            if not config_result.success:
                raise ValueError(
                    f"Failed to configure lakehouse: {config_result.message}"
                )
            
            # Execute setup commands
            for cmd in config_result.setup_commands:
                logger.info(f"Executing command: {cmd}")
                # Here you would execute the command using boto3
                # For example, creating S3 bucket, setting up Glue catalog, etc.
            
            # Create S3 bucket if it doesn't exist
            try:
                self.s3_client.head_bucket(Bucket=self.bucket_name)
                logger.info(f"Bucket {self.bucket_name} already exists")
            except ClientError:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={
                        'LocationConstraint': self.region
                    }
                )
                logger.info(f"Created bucket {self.bucket_name}")
            
            # Create Glue database if it doesn't exist
            try:
                self.glue_client.get_database(Name=self.database_name)
                logger.info(f"Database {self.database_name} already exists")
            except ClientError:
                self.glue_client.create_database(
                    DatabaseInput={
                        'Name': self.database_name,
                        'Description': 'Data Lakehouse Database'
                    }
                )
                logger.info(f"Created database {self.database_name}")
            
            # Apply security recommendations
            for rec in config_result.security_recommendations:
                logger.info(f"Security recommendation: {rec}")
                # Here you would implement the security recommendations
                # For example, setting up bucket policies, encryption, etc.
            
            # Apply performance tips
            for tip in config_result.performance_tips:
                logger.info(f"Performance tip: {tip}")
                # Here you would implement the performance optimizations
                # For example, setting up partitioning, compression, etc.
            
        except Exception as e:
            logger.error(f"Error setting up AWS resources: {str(e)}", exc_info=True)
            raise
    
    async def create_sample_excel(self) -> str:
        """Create a sample Excel file with test data."""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
                # Create sample data
                employees_df = pd.DataFrame({
                    'id': range(1, 6),
                    'name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
                    'age': [30, 25, 35, 28, 32],
                    'email': [
                        'john@example.com',
                        'jane@example.com',
                        'bob@example.com',
                        'alice@example.com',
                        'charlie@example.com'
                    ]
                })
                
                orders_df = pd.DataFrame({
                    'order_id': range(1, 6),
                    'customer_id': [1, 2, 1, 3, 4],
                    'product': ['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard'],
                    'amount': [1000.00, 800.00, 500.00, 300.00, 100.00],
                    'date': pd.date_range(start='2024-01-01', periods=5)
                })
                
                # Write to Excel
                with pd.ExcelWriter(tmp.name) as writer:
                    employees_df.to_excel(writer, sheet_name='employees', index=False)
                    orders_df.to_excel(writer, sheet_name='orders', index=False)
                
                return tmp.name
            
        except Exception as e:
            logger.error(f"Error creating sample Excel: {str(e)}", exc_info=True)
            raise
    
    async def upload_to_s3(self, file_path: str, s3_key: str):
        """Upload a file to S3."""
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, s3_key)
            logger.info(f"Uploaded {file_path} to s3://{self.bucket_name}/{s3_key}")
        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}", exc_info=True)
            raise
    
    async def create_glue_table(
        self,
        table_name: str,
        s3_location: str,
        schema: List[Dict[str, Any]]
    ):
        """Create or update a Glue table."""
        try:
            # Convert schema to Glue format
            columns = []
            for col in schema:
                columns.append({
                    'Name': col['name'],
                    'Type': col['type']
                })
            
            table_input = {
                'Name': table_name,
                'StorageDescriptor': {
                    'Columns': columns,
                    'Location': f's3://{self.bucket_name}/{s3_location}',
                    'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
                    'OutputFormat': (
                        'org.apache.hadoop.hive.ql.io.'
                        'HiveIgnoreKeyTextOutputFormat'
                    ),
                    'SerdeInfo': {
                        'SerializationLibrary': (
                            'org.openx.data.jsonserde.JsonSerDe'
                        )
                    }
                },
                'TableType': 'EXTERNAL_TABLE'
            }
            
            try:
                # Check if table exists
                self.glue_client.get_table(
                    DatabaseName=self.database_name,
                    Name=table_name
                )
                # Table exists, try to update it
                try:
                    self.glue_client.update_table(
                        DatabaseName=self.database_name,
                        TableInput=table_input
                    )
                    logger.info(f"Updated Glue table {table_name}")
                except self.glue_client.exceptions.AccessDeniedException:
                    logger.warning(
                        f"Permission denied to update table {table_name}. "
                        "Continuing with existing table."
                    )
            except self.glue_client.exceptions.EntityNotFoundException:
                # Table doesn't exist, try to create it
                try:
                    self.glue_client.create_table(
                        DatabaseName=self.database_name,
                        TableInput=table_input
                    )
                    logger.info(f"Created Glue table {table_name}")
                except self.glue_client.exceptions.AccessDeniedException:
                    logger.error(
                        f"Permission denied to create table {table_name}. "
                        "Required permissions: glue:CreateTable"
                    )
                    raise
                except self.glue_client.exceptions.AlreadyExistsException:
                    logger.info(f"Table {table_name} already exists")
            
        except Exception as e:
            if isinstance(e, self.glue_client.exceptions.AccessDeniedException):
                logger.error(
                    "AWS Glue permissions error. Required permissions:\n"
                    "- glue:GetTable\n"
                    "- glue:CreateTable\n"
                    "- glue:UpdateTable\n"
                    f"Error: {str(e)}"
                )
            else:
                logger.error(f"Error managing Glue table: {str(e)}", exc_info=True)
            raise
    
    async def run_query(self, query: str) -> List[Dict[str, Any]]:
        """Run a query using Athena."""
        try:
            # Start query execution
            response = self.athena_client.start_query_execution(
                QueryString=query,
                QueryExecutionContext={
                    'Database': self.database_name
                },
                ResultConfiguration={
                    'OutputLocation': f's3://{self.bucket_name}/query-results/'
                }
            )
            
            query_execution_id = response['QueryExecutionId']
            
            # Wait for query to complete
            while True:
                query_status = self.athena_client.get_query_execution(
                    QueryExecutionId=query_execution_id
                )['QueryExecution']['Status']['State']
                
                if query_status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                    break
                
                await asyncio.sleep(1)
            
            if query_status == 'SUCCEEDED':
                # Get results
                results = self.athena_client.get_query_results(
                    QueryExecutionId=query_execution_id
                )
                
                # Parse results
                if not results['ResultSet']['Rows']:
                    return []
                
                # Get column names from the first row
                header_row = results['ResultSet']['Rows'][0]['Data']
                columns = [
                    col.get('VarCharValue', '') 
                    for col in header_row
                ]
                
                # Parse data rows
                rows = []
                for row in results['ResultSet']['Rows'][1:]:
                    row_data = {}
                    for col, val in zip(columns, row['Data']):
                        row_data[col] = val.get('VarCharValue', '')
                    rows.append(row_data)
                
                return rows
            else:
                error_message = self.athena_client.get_query_execution(
                    QueryExecutionId=query_execution_id
                )['QueryExecution']['Status'].get('StateChangeReason', '')
                raise Exception(f"Query failed with status: {query_status}. {error_message}")
            
        except Exception as e:
            logger.error(f"Error running query: {str(e)}", exc_info=True)
            raise
    
    async def test_lakehouse_builder(self):
        """Test the lakehouse builder functionality."""
        try:
            # Set up AWS resources
            await self.setup_aws_resources()
            
            # Create sample Excel file
            excel_file = await self.create_sample_excel()
            
            # Extract schema
            schema = await self.excel_agent.extract_schema(excel_file)
            logger.info(f"Extracted schema: {schema.dict()}")
            
            # Convert schema to list format for SQL generation
            schema_list = []
            for sheet_name, sheet_data in schema.dict()['sheets'].items():
                schema_list.append({
                    'table_name': sheet_name.lower(),
                    'columns': [
                        {
                            'name': col_name,
                            'type': col_type
                        }
                        for col_name, col_type in sheet_data['data_types'].items()
                    ]
                })
            
            # Generate SQL
            sql_statements = await self.sql_agent.generate_sql(
                db_schema=schema_list,
                db_type='postgresql'
            )
            logger.info(f"Generated SQL: {sql_statements.dict()}")
            
            # Upload Excel to S3
            s3_key = 'raw/excel/sample_data.xlsx'
            await self.upload_to_s3(excel_file, s3_key)
            
            # Create Glue tables
            for sheet_name, sheet_data in schema.dict()['sheets'].items():
                table_name = sheet_name.lower()
                s3_location = f'raw/excel/{table_name}'
                
                # Convert schema to Glue format
                columns = []
                for col_name, col_type in sheet_data['data_types'].items():
                    columns.append({
                        'name': col_name,
                        'type': self._convert_to_glue_type(col_type)
                    })
                
                await self.create_glue_table(table_name, s3_location, columns)
            
            # Test query
            query = """
            SELECT e.name, COUNT(o.order_id) as order_count, 
                   SUM(o.amount) as total_amount
            FROM employees e
            LEFT JOIN orders o ON e.id = o.customer_id
            GROUP BY e.name
            ORDER BY total_amount DESC
            """
            
            results = await self.run_query(query)
            logger.info(f"Query results: {results}")
            
            # Clean up
            os.unlink(excel_file)
            
        except Exception as e:
            logger.error(f"Error in lakehouse builder: {str(e)}", exc_info=True)
            raise
    
    def _convert_to_glue_type(self, pandas_type: str) -> str:
        """Convert pandas data type to Glue data type."""
        type_mapping = {
            'int64': 'bigint',
            'float64': 'double',
            'object': 'string',
            'datetime64[ns]': 'timestamp',
            'bool': 'boolean'
        }
        return type_mapping.get(pandas_type, 'string')


async def main():
    """Main function."""
    try:
        demo = LakehouseBuilderDemo()
        await demo.test_lakehouse_builder()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main()) 