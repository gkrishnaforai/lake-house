"""Test Database Configuration Agent.

This module demonstrates how to use the database configuration agent.
"""

import asyncio
import logging
from core.llm.manager import LLMManager
from etl_architect_agent_v2.agents.database_config_agent import (
    DatabaseConfigAgent
)


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseConfigDemo:
    """Demo class for database configuration."""
    
    def __init__(self):
        """Initialize the demo."""
        self.llm_manager = LLMManager()
        self.db_config_agent = DatabaseConfigAgent(self.llm_manager)
    
    async def test_postgres_config(self):
        """Test PostgreSQL configuration."""
        try:
            result = await self.db_config_agent.configure_database(
                db_type='postgresql',
                database='my_database',
                host='localhost',
                port=5432,
                username='postgres',
                password='password',
                additional_config={
                    'max_connections': 100,
                    'shared_buffers': '256MB'
                }
            )
            
            logger.info("PostgreSQL Configuration Result:")
            logger.info(f"Success: {result.success}")
            logger.info(f"Message: {result.message}")
            logger.info(f"Connection String: {result.connection_string}")
            logger.info("Setup Commands:")
            for cmd in result.setup_commands:
                logger.info(f"  {cmd}")
            logger.info("Validation Queries:")
            for query in result.validation_queries:
                logger.info(f"  {query}")
            
        except Exception as e:
            logger.error(f"Error configuring PostgreSQL: {str(e)}", exc_info=True)
            raise
    
    async def test_mysql_config(self):
        """Test MySQL configuration."""
        try:
            result = await self.db_config_agent.configure_database(
                db_type='mysql',
                database='my_database',
                host='localhost',
                port=3306,
                username='root',
                password='password',
                additional_config={
                    'max_connections': 100,
                    'innodb_buffer_pool_size': '256M'
                }
            )
            
            logger.info("MySQL Configuration Result:")
            logger.info(f"Success: {result.success}")
            logger.info(f"Message: {result.message}")
            logger.info(f"Connection String: {result.connection_string}")
            logger.info("Setup Commands:")
            for cmd in result.setup_commands:
                logger.info(f"  {cmd}")
            logger.info("Validation Queries:")
            for query in result.validation_queries:
                logger.info(f"  {query}")
            
        except Exception as e:
            logger.error(f"Error configuring MySQL: {str(e)}", exc_info=True)
            raise
    
    async def test_lakehouse_config(self):
        """Test AWS S3 Lakehouse configuration."""
        try:
            result = await self.db_config_agent.configure_database(
                db_type='aws_s3_lakehouse',
                database='lakehouse_db',
                s3_bucket='my-data-lakehouse',
                region='us-east-1',
                additional_config={
                    'glue_catalog': True,
                    'athena_workgroup': 'primary'
                }
            )
            
            logger.info("Lakehouse Configuration Result:")
            logger.info(f"Success: {result.success}")
            logger.info(f"Message: {result.message}")
            logger.info(f"Connection String: {result.connection_string}")
            logger.info("Setup Commands:")
            for cmd in result.setup_commands:
                logger.info(f"  {cmd}")
            logger.info("Validation Queries:")
            for query in result.validation_queries:
                logger.info(f"  {query}")
            
        except Exception as e:
            logger.error(f"Error configuring Lakehouse: {str(e)}", exc_info=True)
            raise
    
    async def test_database_config(self):
        """Test all database configurations."""
        try:
            # Test PostgreSQL
            logger.info("Testing PostgreSQL configuration...")
            await self.test_postgres_config()
            
            # Test MySQL
            logger.info("\nTesting MySQL configuration...")
            await self.test_mysql_config()
            
            # Test Lakehouse
            logger.info("\nTesting Lakehouse configuration...")
            await self.test_lakehouse_config()
            
        except Exception as e:
            logger.error(f"Error in database config test: {str(e)}", exc_info=True)
            raise


async def main():
    """Main function."""
    try:
        demo = DatabaseConfigDemo()
        await demo.test_database_config()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main()) 