import json
import tempfile
import os
import logging
import sys
from pathlib import Path
from agents.schema_extractor_agent import run_schema_extractor


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('schema_extractor.log')
    ]
)
logger = logging.getLogger(__name__)


async def main():
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
            print("\nTesting Schema Extractor")
            print("=======================")

            # Test 1: Basic Schema Extraction
            print("\n1. Basic Schema Extraction:")
            print("-------------------------")
            
            # Get API key from environment variable
            api_key = os.getenv("OPENAI_API_KEY", "test_api_key")
            logger.debug("Retrieved API key from environment")
            
            logger.info("Starting schema extraction")
            result = run_schema_extractor(
                file_path=temp_file_path,
                table_name="employees",
                api_key=api_key
            )
            logger.info("Schema extraction completed")
            
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
            
            # Print JSON Schema
            print("\nJSON Schema:")
            print("-----------")
            json_schema = result["json_schema"]
            logger.debug(f"JSON Schema: {json.dumps(json_schema, indent=2)}")
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

            # Print Database Schema
            print("\nDatabase Schema:")
            print("---------------")
            db_schema = result["db_schema"]
            logger.debug(f"DB Schema: {json.dumps(db_schema, indent=2)}")
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

        except Exception as e:
            logger.error(
                f"Error during schema extraction: {str(e)}", exc_info=True
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
    import asyncio
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1) 