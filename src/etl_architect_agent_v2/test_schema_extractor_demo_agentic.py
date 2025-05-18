"""Schema Extractor Demo (Agentic Style).

This module demonstrates the agentic style of the schema extractor with
individual agent testing and workflow testing.
"""

import json
import tempfile
import os
import asyncio
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from etl_architect_agent_v2.agents.schema_extractor import (
    JsonSchemaExtractorAgent,
    DatabaseSchemaAgent,
    SQLGeneratorAgent,
    ExcelSchemaExtractorAgent
)
from etl_architect_agent_v2.workflows.schema_workflow import (
    SchemaWorkflow,
    run_schema_workflow
)
from core.llm.manager import LLMManager


# Configure logging for all modules
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set DEBUG level for all relevant loggers
loggers = [
    logging.getLogger(__name__),  # Main test logger
    logging.getLogger('agents.schema_extractor'),  # Schema extractor module
    logging.getLogger(
        'agents.schema_extractor.db_schema_agent'
    ),  # DB schema agent
    logging.getLogger(
        'agents.schema_extractor.json_schema_agent'
    ),  # JSON schema agent
    logging.getLogger(
        'agents.schema_extractor.excel_schema_agent'
    ),  # Excel schema agent
    logging.getLogger(
        'agents.schema_extractor.sql_generator_agent'
    ),  # SQL generator agent
    logging.getLogger('core.llm.manager'),  # LLM manager
]

for logger in loggers:
    logger.setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


class SchemaExtractorDemo:
    """Demo class for schema extractor agents."""
    
    def __init__(self, api_key: str):
        """Initialize the demo with API key."""
        self.llm_manager = LLMManager(api_key=api_key)
        self.json_agent = JsonSchemaExtractorAgent(self.llm_manager)
        self.excel_agent = ExcelSchemaExtractorAgent(self.llm_manager)
        self.db_agent = DatabaseSchemaAgent(self.llm_manager)
        self.sql_agent = SQLGeneratorAgent(self.llm_manager)
        self.output_dir = Path(
            "/Users/krishnag/tools/llm/lanng-chain/web-app/ai_project/"
            "web-app/aws_architect_agent/output"
        )
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize state
        self.state = {
            "file_path": None,
            "json_schema": None,
            "db_schema": None,
            "sql_statements": None
        }
    
    def _validate_state(self, required_keys: List[str]) -> None:
        """Validate that required state keys are present and valid."""
        missing_keys = [key for key in required_keys if not self.state.get(key)]
        if missing_keys:
            raise ValueError(
                f"Missing required state: {', '.join(missing_keys)}"
            )
    
    def _create_sample_excel(self) -> str:
        """Create a sample Excel file for testing."""
        # Create a temporary file
        fd, path = tempfile.mkstemp(suffix='.xlsx')
        os.close(fd)
        
        # Create sample data
        data = {
            'Sheet1': pd.DataFrame({
                'id': [1, 2, 3],
                'name': ['John', 'Jane', 'Bob'],
                'age': [30, 25, 35],
                'email': ['john@example.com', 'jane@example.com', 'bob@example.com']
            }),
            'Sheet2': pd.DataFrame({
                'order_id': [1, 2, 3],
                'customer_id': [1, 2, 1],
                'product': ['Laptop', 'Phone', 'Tablet'],
                'amount': [1000.50, 500.75, 300.25],
                'date': pd.date_range('2024-01-01', periods=3)
            })
        }
        
        # Write to Excel file
        with pd.ExcelWriter(path) as writer:
            for sheet_name, df in data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        logger.info(f"Created sample Excel file: {path}")
        return path
    
    async def test_excel_schema_agent(
        self,
        file_path: str = None
    ) -> Dict[str, Any]:
        """Test the Excel schema extraction agent."""
        logger.info("Testing Excel Schema Agent")
        try:
            # Create sample Excel if no file provided
            if file_path is None:
                file_path = self._create_sample_excel()
            
            # Update state
            self.state["file_path"] = file_path
            
            schema = await self.excel_agent.extract_schema(file_path)
            
            # Write schema to file
            output_path = self.output_dir / "agentic_excel_schema.json"
            with open(output_path, 'w') as f:
                json.dump(schema.model_dump(), f, indent=2)
            logger.info(f"Excel schema written to: {output_path}")
            
            # Update state
            self.state["json_schema"] = schema.model_dump()
            logger.debug(
                f"Updated state with Excel schema: "
                f"{json.dumps(self.state['json_schema'], indent=2)}"
            )
            
            return schema.model_dump()
        except Exception as e:
            logger.error(
                f"Error in Excel schema agent: {str(e)}", exc_info=True
            )
            raise
        finally:
            # Clean up temporary file if we created it
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
    
    async def test_json_schema_agent(
        self,
        file_path: str
    ) -> Dict[str, Any]:
        """Test the JSON schema extraction agent."""
        logger.info("Testing JSON Schema Agent")
        try:
            # Update state
            self.state["file_path"] = file_path
            
            schema = await self.json_agent.extract_schema(file_path)
            
            # Write schema to file
            output_path = self.output_dir / "agentic_json_schema.json"
            with open(output_path, 'w') as f:
                json.dump(schema.model_dump(), f, indent=2)
            logger.info(f"JSON schema written to: {output_path}")
            
            # Update state
            self.state["json_schema"] = schema.model_dump()
            logger.debug(
                f"Updated state with JSON schema: "
                f"{json.dumps(self.state['json_schema'], indent=2)}"
            )
            
            return schema.model_dump()
        except Exception as e:
            logger.error(
                f"Error in JSON schema agent: {str(e)}", exc_info=True
            )
            raise
    
    async def test_db_schema_agent(
        self,
        json_schema: Dict[str, Any] = None,
        db_type: str = None
    ) -> List[Dict[str, Any]]:
        """Test the database schema conversion agent."""
        logger.info("Testing Database Schema Agent")
        try:
            # Use state if no arguments provided
            if json_schema is None:
                self._validate_state(["json_schema"])
                json_schema = self.state["json_schema"]
            
            if db_type is None:
                raise ValueError("db_type is required")
            
            tables = await self.db_agent.convert_schema(
                json_schema=json_schema,
                db_type=db_type
            )
            
            # Write schema to file
            output_path = self.output_dir / "agentic_db_schema.json"
            with open(output_path, 'w') as f:
                json.dump([table.model_dump() for table in tables], f, indent=2)
            logger.info(f"Database schema written to: {output_path}")
            
            # Update state
            self.state["db_schema"] = [table.model_dump() for table in tables]
            logger.debug(
                f"Updated state with DB schema: "
                f"{json.dumps(self.state['db_schema'], indent=2)}"
            )
            
            return [table.model_dump() for table in tables]
        except Exception as e:
            logger.error(
                f"Error in DB schema agent: {str(e)}", exc_info=True
            )
            raise
    
    async def test_sql_generator_agent(
        self,
        db_schema: List[Dict[str, Any]] = None,
        db_type: str = None,
        operation: str = None
    ) -> Dict[str, List[str]]:
        """Test the SQL generator agent."""
        logger.info("Testing SQL Generator Agent")
        try:
            # Use state if no arguments provided
            if db_schema is None:
                self._validate_state(["db_schema"])
                db_schema = self.state["db_schema"]
            
            if db_type is None or operation is None:
                raise ValueError("db_type and operation are required")
            
            statements = await self.sql_agent.generate_sql(
                db_schema=db_schema,
                db_type=db_type,
                operation=operation
            )
            
            # Write SQL to file
            output_path = self.output_dir / "agentic_sql.sql"
            with open(output_path, 'w') as f:
                # Write CREATE TABLE statements
                for stmt in statements.create_tables:
                    f.write(f"{stmt};\n\n")
                
                # Write CREATE INDEX statements
                for stmt in statements.create_indexes:
                    f.write(f"{stmt};\n\n")
                
                # Write constraint statements
                for stmt in statements.create_constraints:
                    f.write(f"{stmt};\n\n")
                
                # Write INSERT statements if any
                if statements.insert_data:
                    f.write("-- Sample INSERT statements\n")
                    for stmt in statements.insert_data:
                        f.write(f"{stmt};\n\n")
                
                # Write UPDATE statements if any
                if statements.update_data:
                    f.write("-- Sample UPDATE statements\n")
                    for stmt in statements.update_data:
                        f.write(f"{stmt};\n\n")
            
            logger.info(f"SQL statements written to: {output_path}")
            
            # Update state
            self.state["sql_statements"] = statements.model_dump()
            logger.debug(
                f"Updated state with SQL statements: "
                f"{json.dumps(self.state['sql_statements'], indent=2)}"
            )
            
            return statements.model_dump()
        except Exception as e:
            logger.error(
                f"Error in SQL generator agent: {str(e)}", exc_info=True
            )
            raise
    
    async def test_workflow(
        self,
        file_path: str,
        table_name: str,
        db_type: str,
        operation: str
    ) -> Dict[str, Any]:
        """Test the complete workflow."""
        logger.info("Testing Complete Workflow")
        try:
            # Reset state for new workflow
            self.state = {
                "file_path": None,
                "json_schema": None,
                "db_schema": None,
                "sql_statements": None
            }
            
            # Create workflow instance with our LLM manager
            workflow = SchemaWorkflow(self.llm_manager)
            
            # Run workflow
            result = await workflow.run_workflow(
                file_path=file_path,
                table_name=table_name,
                db_type=db_type,
                operation=operation
            )
            
            # Update state with workflow results
            self.state.update({
                "file_path": file_path,
                "json_schema": result["json_schema"],
                "db_schema": result["db_schema"],
                "sql_statements": result["sql_statements"]
            })
            logger.debug(f"Updated state with workflow results: {json.dumps(self.state, indent=2)}")
            
            # Write all outputs to files
            json_schema_path = self.output_dir / "agentic_workflow_json_schema.json"
            with open(json_schema_path, 'w') as f:
                json.dump(result["json_schema"], f, indent=2)
            
            db_schema_path = self.output_dir / "agentic_workflow_db_schema.json"
            with open(db_schema_path, 'w') as f:
                json.dump(result["db_schema"], f, indent=2)
            
            sql_path = self.output_dir / "agentic_workflow_sql.sql"
            with open(sql_path, 'w') as f:
                # Write CREATE TABLE statements
                for stmt in result["sql_statements"]["create_tables"]:
                    f.write(f"{stmt};\n\n")
                
                # Write CREATE INDEX statements
                for stmt in result["sql_statements"]["create_indexes"]:
                    f.write(f"{stmt};\n\n")
                
                # Write constraint statements
                for stmt in result["sql_statements"]["create_constraints"]:
                    f.write(f"{stmt};\n\n")
                
                # Write INSERT statements if any
                if result["sql_statements"]["insert_data"]:
                    f.write("-- Sample INSERT statements\n")
                    for stmt in result["sql_statements"]["insert_data"]:
                        f.write(f"{stmt};\n\n")
                
                # Write UPDATE statements if any
                if result["sql_statements"]["update_data"]:
                    f.write("-- Sample UPDATE statements\n")
                    for stmt in result["sql_statements"]["update_data"]:
                        f.write(f"{stmt};\n\n")
            
            logger.info("Workflow outputs written to files")
            
            return result
        except Exception as e:
            logger.error(
                f"Error in workflow: {str(e)}", exc_info=True
            )
            raise


async def main():
    """Run the schema extractor demo."""
    try:
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Create demo instance
        demo = SchemaExtractorDemo(api_key)
        
        # Test Excel schema extraction
        logger.info("\n=== Testing Excel Schema Extraction ===")
        excel_schema = await demo.test_excel_schema_agent()
        
        # Test database schema conversion
        logger.info("\n=== Testing Database Schema Conversion ===")
        db_schema = await demo.test_db_schema_agent(
            json_schema=excel_schema,
            db_type="postgresql"
        )
        
        # Test SQL generation
        logger.info("\n=== Testing SQL Generation ===")
        sql_statements = await demo.test_sql_generator_agent(
            db_schema=db_schema,
            db_type="postgresql",
            operation="create"
        )
        
        # Test complete workflow with Excel
        logger.info("\n=== Testing Complete Workflow with Excel ===")
        workflow_result = await demo.test_workflow(
            file_path=demo._create_sample_excel(),
            table_name="excel_data",
            db_type="postgresql",
            operation="create"
        )
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main()) 