"""Test script for demonstrating the data lakehouse builder UI functionality."""

import logging
import os
from pathlib import Path
import pandas as pd
from datetime import datetime
from etl_architect_agent_v2.core.schema_registry import SchemaRegistry
from etl_architect_agent_v2.core.db_connection_manager import (
    DatabaseConnectionManager,
    DatabaseConfig
)
from etl_architect_agent_v2.core.file_processor import FileProcessor
from etl_architect_agent_v2.agents.schema_extractor.file_processor_agent import (
    FileProcessorAgent
)
from etl_architect_agent_v2.core.llm.manager import LLMManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UIDemo:
    """Demo class for testing UI functionality."""
    
    def __init__(self):
        """Initialize demo components."""
        # Initialize components
        self.schema_registry = SchemaRegistry()
        self.db_manager = DatabaseConnectionManager()
        self.file_processor = FileProcessor()
        self.llm_manager = LLMManager()
        self.file_processor_agent = FileProcessorAgent(self.llm_manager)
        
        # Load AWS credentials from environment
        self.aws_credentials = {
            "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "aws_session_token": os.getenv("AWS_SESSION_TOKEN"),
            "region_name": os.getenv("AWS_REGION", "us-east-1")
        }
        
        # Create test directory
        self.test_dir = Path("test_data")
        self.test_dir.mkdir(exist_ok=True)
    
    def create_sample_file(self) -> Path:
        """Create a sample Excel file for testing."""
        # Create sample data
        data = {
            "id": [1, 2, 3, 4, 5],
            "name": ["John", "Jane", "Bob", "Alice", "Charlie"],
            "age": [30, 25, 35, 28, 32],
            "salary": [50000, 60000, 55000, 65000, 70000],
            "department": ["IT", "HR", "Finance", "Marketing", "Sales"],
            "join_date": [
                "2023-01-15",
                "2023-02-01",
                "2023-03-10",
                "2023-04-05",
                "2023-05-20"
            ]
        }
        
        # Create DataFrame and save to Excel
        df = pd.DataFrame(data)
        file_path = self.test_dir / "sample_employees.xlsx"
        df.to_excel(file_path, index=False)
        logger.info(f"Created sample file: {file_path}")
        return file_path
    
    async def test_file_processing(self, file_path: Path):
        """Test file processing functionality."""
        logger.info("Testing file processing...")
        
        # Validate file
        validation_result = await self.file_processor_agent.validate_file(str(file_path))
        if not validation_result.is_valid:
            logger.error(f"File validation failed: {validation_result.error_messages}")
            return
        
        logger.info(f"File type: {validation_result.file_type}")
        logger.info("Sample data:")
        for record in validation_result.sample_data[:2]:
            logger.info(record)
    
    def setup_database_connections(self):
        """Set up database connections."""
        logger.info("Setting up database connections...")
        
        # Add S3 connection
        s3_config = DatabaseConfig(
            type="s3",
            database="lakehouse",
            username=self.aws_credentials["aws_access_key_id"],
            password=self.aws_credentials["aws_secret_access_key"],
            region=self.aws_credentials["region_name"],
            bucket=os.getenv("AWS_S3_BUCKET")
        )
        self.db_manager.add_connection("s3_lakehouse", s3_config)
        
        # Add MySQL connection if credentials are available
        if all(os.getenv(var) for var in ["MYSQL_HOST", "MYSQL_PORT", "MYSQL_DB", "MYSQL_USER", "MYSQL_PASSWORD"]):
            mysql_config = DatabaseConfig(
                type="mysql",
                host=os.getenv("MYSQL_HOST"),
                port=int(os.getenv("MYSQL_PORT")),
                database=os.getenv("MYSQL_DB"),
                username=os.getenv("MYSQL_USER"),
                password=os.getenv("MYSQL_PASSWORD")
            )
            self.db_manager.add_connection("mysql_db", mysql_config)
        
        # Add PostgreSQL connection if credentials are available
        if all(os.getenv(var) for var in ["PG_HOST", "PG_PORT", "PG_DB", "PG_USER", "PG_PASSWORD"]):
            pg_config = DatabaseConfig(
                type="postgres",
                host=os.getenv("PG_HOST"),
                port=int(os.getenv("PG_PORT")),
                database=os.getenv("PG_DB"),
                username=os.getenv("PG_USER"),
                password=os.getenv("PG_PASSWORD")
            )
            self.db_manager.add_connection("postgres_db", pg_config)
        
        logger.info("Available connections:")
        for name, type_ in self.db_manager.list_connections().items():
            logger.info(f"- {name}: {type_}")
    
    def register_schema(self, file_path: Path):
        """Register schema from processed file."""
        logger.info("Registering schema...")
        
        # Get file type and validate
        file_type = self.file_processor.detect_file_type(file_path)
        validation_result = self.file_processor.validate_file(file_path)
        
        if not validation_result.is_valid:
            logger.error(f"File validation failed: {validation_result.error_message}")
            return
        
        # Register schema
        schema_name = f"employees_{datetime.now().strftime('%Y%m%d')}"
        try:
            metadata = self.schema_registry.register_schema(
                name=schema_name,
                schema=validation_result.schema,
                schema_type="table",
                database="lakehouse",
                user="test_user",
                description=f"Schema for {file_path.name}"
            )
            logger.info(f"Registered schema: {schema_name}")
            logger.info(f"Current version: {metadata.current_version}")
        except ValueError as e:
            logger.error(f"Schema registration failed: {str(e)}")
    
    async def run_demo(self):
        """Run the complete demo."""
        try:
            # Create sample file
            file_path = self.create_sample_file()
            
            # Test file processing
            await self.test_file_processing(file_path)
            
            # Setup database connections
            self.setup_database_connections()
            
            # Register schema
            self.register_schema(file_path)
            
            logger.info("Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {str(e)}", exc_info=True)
        finally:
            # Cleanup
            if file_path.exists():
                file_path.unlink()
            if self.test_dir.exists():
                self.test_dir.rmdir()


if __name__ == "__main__":
    import asyncio
    
    # Run the demo
    demo = UIDemo()
    asyncio.run(demo.run_demo()) 