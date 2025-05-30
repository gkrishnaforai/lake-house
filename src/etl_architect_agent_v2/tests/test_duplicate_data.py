import asyncio
import logging
import os
from typing import Dict, Any
import pandas as pd
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from etl_architect_agent_v2.backend.services.catalog_service import CatalogService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_query_execution():
    """Test query execution and analyze results for duplicates."""
    try:
        # Get configuration from environment variables
        bucket = os.getenv('CATALOG_BUCKET', 'lambda-code-q')  # Default to lambda-code-q
        aws_region = os.getenv('AWS_REGION', 'us-east-1')
        
        logger.info(f"Using bucket: {bucket}")
        logger.info(f"Using AWS region: {aws_region}")
        
        # Initialize CatalogService
        catalog_service = CatalogService(
            bucket=bucket,
            aws_region=aws_region
        )
        
        # Test query
        query = "SELECT * FROM user_test_user.sample11111"
        logger.info(f"Executing query: {query}")
        
        # Execute query
        result = await catalog_service.execute_query(query, user_id="test_user")
        
        if result["status"] == "error":
            logger.error(f"Query execution failed: {result.get('message')}")
            return
        
        # Get results
        results = result.get("results", [])
        columns = result.get("columns", [])
        
        # Log basic information
        logger.info(f"Number of rows returned: {len(results)}")
        logger.info(f"Columns: {columns}")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(results, columns=columns)
        
        # Check for duplicates
        duplicates = df.duplicated()
        duplicate_count = duplicates.sum()
        
        if duplicate_count > 0:
            logger.warning(f"Found {duplicate_count} duplicate rows!")
            
            # Get duplicate rows
            duplicate_rows = df[duplicates]
            logger.info("\nDuplicate rows:")
            logger.info(duplicate_rows)
            
            # Get unique rows
            unique_rows = df.drop_duplicates()
            logger.info(f"\nNumber of unique rows: {len(unique_rows)}")
            
            # Compare with original
            logger.info(f"Original row count: {len(df)}")
            logger.info(f"Unique row count: {len(unique_rows)}")
            
            # Analyze which columns might be causing duplicates
            for col in columns:
                unique_values = df[col].nunique()
                logger.info(f"Column '{col}' has {unique_values} unique values")
                
            # Check for exact duplicates in the raw results
            raw_duplicates = set()
            duplicate_indices = []
            for i, row in enumerate(results):
                row_tuple = tuple(row)
                if row_tuple in raw_duplicates:
                    duplicate_indices.append(i)
                else:
                    raw_duplicates.add(row_tuple)
            
            if duplicate_indices:
                logger.info("\nFound duplicate rows at indices:")
                for idx in duplicate_indices:
                    logger.info(f"Index {idx}: {results[idx]}")
        
        else:
            logger.info("No duplicate rows found in the results")
        
        # Print first few rows for inspection
        logger.info("\nFirst few rows of the result:")
        logger.info(df.head())
        
        # Analyze the query execution
        logger.info("\nQuery Execution Analysis:")
        logger.info(f"Query: {query}")
        logger.info(f"Modified Query: {result.get('modified_query', query)}")
        logger.info(f"Query Execution ID: {result.get('query_execution_id')}")
        
    except Exception as e:
        logger.error(f"Error during test: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_query_execution()) 