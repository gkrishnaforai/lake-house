from typing import Dict, Any, Optional, List
import boto3
from botocore.exceptions import ClientError
import logging
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class AWSServiceManager:
    """Manages AWS service integrations for the data lake system."""
    
    def __init__(
        self,
        region_name: str = "us-east-1",
        profile_name: Optional[str] = None
    ):
        """Initialize AWS service manager.
        
        Args:
            region_name: AWS region name
            profile_name: Optional AWS profile name
        """
        self.region_name = region_name
        self.profile_name = profile_name
        self.session = boto3.Session(
            region_name=region_name,
            profile_name=profile_name
        )
        
        # Initialize AWS service clients
        self.s3 = self.session.client('s3')
        self.glue = self.session.client('glue')
        self.athena = self.session.client('athena')
        
    async def create_data_lake(
        self,
        bucket_name: str,
        database_name: str,
        table_name: str,
        schema: List[Dict[str, str]],
        location: str = "s3://{bucket_name}/data/{table_name}"
    ) -> Dict[str, Any]:
        """Create a new data lake with S3, Glue, and Athena.
        
        Args:
            bucket_name: S3 bucket name
            database_name: Glue database name
            table_name: Glue table name
            schema: List of column definitions
            location: S3 location for the table
            
        Returns:
            Dict containing creation results
        """
        try:
            # Create S3 bucket if it doesn't exist
            try:
                self.s3.head_bucket(Bucket=bucket_name)
            except ClientError:
                self.s3.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={
                        'LocationConstraint': self.region_name
                    }
                )
            
            # Create Glue database
            try:
                self.glue.create_database(
                    DatabaseInput={
                        'Name': database_name,
                        'Description': f'Database for {table_name}'
                    }
                )
            except ClientError as e:
                if e.response['Error']['Code'] != 'AlreadyExistsException':
                    raise
            
            # Create Glue table
            table_location = location.format(
                bucket_name=bucket_name,
                table_name=table_name
            )
            
            self.glue.create_table(
                DatabaseName=database_name,
                TableInput={
                    'Name': table_name,
                    'TableType': 'EXTERNAL_TABLE',
                    'StorageDescriptor': {
                        'Columns': schema,
                        'Location': table_location,
                        'InputFormat': (
                            'org.apache.hadoop.mapred.TextInputFormat'
                        ),
                        'OutputFormat': (
                            'org.apache.hadoop.hive.ql.io.'
                            'HiveIgnoreKeyTextOutputFormat'
                        ),
                        'SerdeInfo': {
                            'SerializationLibrary': (
                                'org.apache.hadoop.hive.serde2.lazy.'
                                'LazySimpleSerDe'
                            ),
                            'Parameters': {
                                'serialization.format': ',',
                                'field.delim': ','
                            }
                        }
                    },
                    'Parameters': {
                        'classification': 'csv',
                        'typeOfData': 'file'
                    }
                }
            )
            
            return {
                'status': 'success',
                'bucket_name': bucket_name,
                'database_name': database_name,
                'table_name': table_name,
                'location': table_location
            }
            
        except Exception as e:
            logger.error(f"Error creating data lake: {str(e)}")
            raise
    
    async def execute_query(
        self,
        query: str,
        database: str,
        output_location: str
    ) -> Dict[str, Any]:
        """Execute a query using Athena.
        
        Args:
            query: SQL query to execute
            database: Database name
            output_location: S3 location for query results
            
        Returns:
            Query execution results
        """
        try:
            # Start query execution
            response = self.athena.start_query_execution(
                QueryString=query,
                QueryExecutionContext={
                    'Database': database
                },
                ResultConfiguration={
                    'OutputLocation': output_location
                }
            )
            
            query_execution_id = response['QueryExecutionId']
            
            # Wait for query to complete
            while True:
                query_status = self.athena.get_query_execution(
                    QueryExecutionId=query_execution_id
                )
                state = query_status['QueryExecution']['Status']['State']
                
                if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                    break
                    
                await asyncio.sleep(1)
            
            if state == 'SUCCEEDED':
                # Get query results
                results = self.athena.get_query_results(
                    QueryExecutionId=query_execution_id
                )
                
                return {
                    'status': 'success',
                    'results': results['ResultSet']['Rows'],
                    'metadata': {
                        'execution_time': (
                            query_status['QueryExecution']['Statistics']
                            ['TotalExecutionTimeInMillis']
                        ),
                        'data_scanned': (
                            query_status['QueryExecution']['Statistics']
                            ['DataScannedInBytes']
                        )
                    }
                }
            else:
                error = query_status['QueryExecution']['Status'].get(
                    'StateChangeReason', 'Unknown error'
                )
                raise Exception(f"Query failed: {error}")
                
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise
    
    async def get_table_schema(
        self,
        database: str,
        table: str
    ) -> Dict[str, Any]:
        """Get schema information for a table.
        
        Args:
            database: Database name
            table: Table name
            
        Returns:
            Table schema information
        """
        try:
            response = self.glue.get_table(
                DatabaseName=database,
                Name=table
            )
            
            return {
                'status': 'success',
                'schema': response['Table']['StorageDescriptor']['Columns'],
                'metadata': {
                    'table_type': response['Table']['TableType'],
                    'location': response['Table']['StorageDescriptor']['Location'],
                    'last_updated': response['Table'].get(
                        'UpdateTime', datetime.utcnow()
                    ).isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting table schema: {str(e)}")
            raise
    
    async def check_data_quality(
        self,
        database: str,
        table: str,
        metrics: List[str] = ['completeness', 'accuracy', 'consistency']
    ) -> Dict[str, Any]:
        """Check data quality metrics for a table.
        
        Args:
            database: Database name
            table: Table name
            metrics: List of metrics to check
            
        Returns:
            Data quality metrics
        """
        try:
            # Build and execute quality check queries
            quality_metrics = {}
            for metric in metrics:
                if metric == 'completeness':
                    # Check for NULL values
                    query = f"""
                    SELECT 
                        COUNT(*) as total_rows,
                        COUNT(CASE WHEN column_name IS NULL THEN 1 END) 
                        as null_count
                    FROM {database}.{table}
                    """
                    results = await self.execute_query(
                        query=query,
                        database=database,
                        output_location=(
                            f"s3://{self.bucket_name}/quality-checks"
                        )
                    )
                    
                    quality_metrics['completeness'] = {
                        'null_percentage': (
                            results['results'][1]['Data'][1]['VarCharValue'] /
                            results['results'][1]['Data'][0]['VarCharValue']
                        ) * 100
                    }
                
                # Add more metric checks as needed
            
            return {
                'status': 'success',
                'table_name': table,
                'metrics': quality_metrics,
                'metadata': {
                    'last_checked': datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error checking data quality: {str(e)}")
            raise 