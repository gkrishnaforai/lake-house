import unittest
import tempfile
import json
import os
import asyncio

from src.etl_architect_agent_v2.agents.schema_extractor_agent import (
    run_schema_extractor,
    SchemaExtractorState,
    JsonSchemaModel,
    TableSchemaModel
)


class TestSchemaExtractorAgent(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.test_json_path = os.path.join(self.temp_dir, "test_employees.json")
        
        # Write test data to file
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
        
        with open(self.test_json_path, 'w') as f:
            json.dump(test_data, f)

    def tearDown(self):
        # Clean up temporary files
        try:
            if os.path.exists(self.test_json_path):
                os.remove(self.test_json_path)
            if os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)
        except OSError:
            # Ignore directory not empty errors during cleanup
            pass

    async def test_basic_schema_extraction(self):
        """Test basic schema extraction with JSON and DB schema generation."""
        result = await run_schema_extractor(
            file_path=self.test_json_path,
            table_name="employees",
            api_key="test_api_key"
        )
        
        self.assertIsNotNone(result)
        self.assertIn("json_schema", result)
        self.assertIn("db_schema", result)
        
        # Verify JSON Schema
        json_schema = result["json_schema"]
        self.assertEqual(json_schema["schema"], "http://json-schema.org/draft-04/schema#")
        self.assertEqual(json_schema["type"], "object")
        self.assertIn("properties", json_schema)
        self.assertIn("employees", json_schema["properties"])
        
        # Verify DB Schema
        db_schema = result["db_schema"]
        self.assertTrue(len(db_schema) > 0)
        
        # Check for main tables
        table_names = [table["table_name"] for table in db_schema]
        self.assertIn("employees", table_names)
        self.assertIn("personal_info", table_names)
        self.assertIn("contact", table_names)
        self.assertIn("employment", table_names)
        self.assertIn("projects", table_names)
        
        # Check for relationships
        for table in db_schema:
            if table["table_name"] == "employees":
                self.assertTrue(len(table["foreign_keys"]) > 0)
                self.assertTrue(len(table["primary_key"]) > 0)

    async def test_invalid_json_file(self):
        """Test handling of invalid JSON file."""
        invalid_path = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_path, 'w') as f:
            f.write("invalid json content")
            
        with self.assertRaises(ValueError):
            await run_schema_extractor(
                file_path=invalid_path,
                table_name="test",
                api_key="test_api_key"
            )

    async def test_missing_file(self):
        """Test handling of missing file."""
        with self.assertRaises(ValueError):
            await run_schema_extractor(
                file_path="nonexistent.json",
                table_name="test",
                api_key="test_api_key"
            )

    async def test_empty_json(self):
        """Test handling of empty JSON file."""
        empty_path = os.path.join(self.temp_dir, "empty.json")
        with open(empty_path, 'w') as f:
            f.write("{}")
            
        with self.assertRaises(ValueError):
            await run_schema_extractor(
                file_path=empty_path,
                table_name="test",
                api_key="test_api_key"
            )

    async def test_schema_structure(self):
        """Test the structure of generated schemas."""
        result = await run_schema_extractor(
            file_path=self.test_json_path,
            table_name="employees",
            api_key="test_api_key"
        )
        
        # Verify JSON Schema structure
        json_schema = result["json_schema"]
        self.assertIn("required", json_schema)
        self.assertIn("employees", json_schema["properties"])
        employees_schema = json_schema["properties"]["employees"]
        self.assertEqual(employees_schema["type"], "array")
        self.assertIn("items", employees_schema)
        
        # Verify DB Schema structure
        db_schema = result["db_schema"]
        for table in db_schema:
            self.assertIn("table_name", table)
            self.assertIn("columns", table)
            self.assertIn("primary_key", table)
            self.assertIn("foreign_keys", table)
            self.assertIn("indexes", table)
            
            # Check column structure
            for column in table["columns"]:
                self.assertIn("name", column)
                self.assertIn("type", column)
                self.assertIn("nullable", column)


if __name__ == '__main__':
    unittest.main()