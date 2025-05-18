"""Test Schema Workflow.

This module tests the schema extraction and conversion workflow.
"""

import json
import tempfile
import os
import asyncio
import logging
from pathlib import Path
from agents.schema_extractor import run_schema_workflow


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_schema_workflow():
    """Test the schema workflow."""
    temp_file_path = None
    try:
        # Create test data
        test_data = {
            "employees": [
                {
                    "id": 12345,
                    "personal_info": {
                        "name": "John Doe",
                        "age": 35,
                        "contact": {
                            "email": "john.doe@example.com",
                            "phone": {
                                "mobile": "+1234567890",
                                "work": "+0987654321"
                            }
                        }
                    },
                    "employment": {
                        "department": "Engineering",
                        "position": "Senior Developer",
                        "salary": 120000.50,
                        "start_date": "2020-01-15",
                        "is_active": True,
                        "skills": ["Python", "Java", "SQL", "AWS"],
                        "projects": [
                            {
                                "name": "Project A",
                                "role": "Lead",
                                "duration_months": 12
                            }
                        ]
                    }
                }
            ]
        }
        
        logger.debug("Created test data")
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False
        ) as temp_file:
            json.dump(test_data, temp_file, indent=2)
            temp_file_path = temp_file.name
            logger.debug(f"Created temporary file: {temp_file_path}")
        
        try:
            print("\nTesting Schema Workflow")
            print("======================")
            
            # Get API key from environment variable
            api_key = os.getenv("OPENAI_API_KEY", "test_api_key")
            logger.debug("Retrieved API key from environment")
            
            # Run workflow
            logger.info("Starting schema workflow")
            result = await run_schema_workflow(
                file_path=temp_file_path,
                table_name="employees",
                db_type="postgresql",
                operation="create",
                api_key=api_key
            )
            logger.info("Schema workflow completed")
            
            # Set output directory
            output_dir = Path(
                "/Users/krishnag/tools/llm/lanng-chain/web-app/ai_project/"
                "web-app/aws_architect_agent/output"
            )
            output_dir.mkdir(exist_ok=True)
            logger.debug(f"Output directory: {output_dir}")
            
            # Write JSON schema to file
            json_schema_path = output_dir / "generated_json_schema.json"
            try:
                with open(json_schema_path, 'w') as f:
                    json.dump(result["json_schema"], f, indent=2)
                logger.info(f"JSON schema written to: {json_schema_path}")
            except Exception as e:
                logger.error(
                    f"Error writing JSON schema: {str(e)}", exc_info=True
                )
                raise
            
            # Write database schema to file
            db_schema_path = output_dir / "generated_db_schema.json"
            try:
                with open(db_schema_path, 'w') as f:
                    json.dump(result["db_schema"], f, indent=2)
                logger.info(f"Database schema written to: {db_schema_path}")
            except Exception as e:
                logger.error(
                    f"Error writing DB schema: {str(e)}", exc_info=True
                )
                raise
            
            # Write SQL statements to file
            sql_path = output_dir / "generated_sql.sql"
            try:
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
                
                logger.info(f"SQL statements written to: {sql_path}")
            except Exception as e:
                logger.error(
                    f"Error writing SQL statements: {str(e)}", exc_info=True
                )
                raise
            
            # Print results
            print("\nJSON Schema:")
            print("-----------")
            json_schema = result["json_schema"]
            print(f"Schema Version: {json_schema['schema']}")
            print(f"Type: {json_schema['type']}")
            print("\nProperties:")
            for prop_name, prop_value in json_schema["properties"].items():
                print(f"- {prop_name}: {prop_value['type']}")
                if "items" in prop_value:
                    print(f"  Items: {prop_value['items']['type']}")
            print("\nRequired Fields:")
            for field in json_schema["required"]:
                print(f"- {field}")
            
            print("\nDatabase Schema:")
            print("---------------")
            db_schema = result["db_schema"]
            for table in db_schema:
                print(f"\nTable: {table['table_name']}")
                print("Columns:")
                for col in table["columns"]:
                    print(f"- {col['name']}: {col['type']} "
                          f"(Nullable: {col['nullable']})")
                
                if table["primary_key"]:
                    print("\nPrimary Key:")
                    print(f"- {', '.join(table['primary_key'])}")
                
                if table["foreign_keys"]:
                    print("\nForeign Keys:")
                    for fk in table["foreign_keys"]:
                        print(f"- {fk['column']} -> {fk['references']}")
                
                if table["indexes"]:
                    print("\nIndexes:")
                    for idx in table["indexes"]:
                        print(f"- {idx['name']}: {', '.join(idx['columns'])}")
            
            print("\nSQL Statements:")
            print("--------------")
            sql_statements = result["sql_statements"]
            print("\nCREATE TABLE Statements:")
            for stmt in sql_statements["create_tables"]:
                print(f"\n{stmt};")
            
            print("\nCREATE INDEX Statements:")
            for stmt in sql_statements["create_indexes"]:
                print(f"\n{stmt};")
            
            print("\nConstraint Statements:")
            for stmt in sql_statements["create_constraints"]:
                print(f"\n{stmt};")
            
            if sql_statements["insert_data"]:
                print("\nSample INSERT Statements:")
                for stmt in sql_statements["insert_data"]:
                    print(f"\n{stmt};")
            
            if sql_statements["update_data"]:
                print("\nSample UPDATE Statements:")
                for stmt in sql_statements["update_data"]:
                    print(f"\n{stmt};")
        
        except Exception as e:
            logger.error(
                f"Error during schema workflow: {str(e)}", exc_info=True
            )
            raise
    
    finally:
        # Clean up the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.debug(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.error(
                    f"Error cleaning up temp file: {str(e)}", exc_info=True
                )


if __name__ == "__main__":
    asyncio.run(test_schema_workflow()) 