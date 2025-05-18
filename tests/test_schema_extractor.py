import unittest
import tempfile
import json
import os
import sys
from agents.SchemaExtractorAgent import (
    run_schema_extractor,
    SchemaExtractorState
)

class TestSchemaExtractorAgent(unittest.TestCase):
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
        if os.path.exists(self.test_json_path):
            os.remove(self.test_json_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_basic_schema_extraction(self):
        """Test basic schema extraction without LLM enhancement."""
        result = run_schema_extractor(
            file_path=self.test_json_path,
            table_name="employees",
            llm_enhanced=False
        )
        
        self.assertIsNotNone(result)
        self.assertIn("schema", result)
        self.assertNotIn("error", result)
        
        schema = result["schema"]
        self.assertEqual(schema.table_name, "employees")
        self.assertTrue(len(schema.columns) > 0)
        
        # Check for nested columns
        nested_columns = [col for col in schema.columns if "." in col.name]
        self.assertTrue(len(nested_columns) > 0)
        
        # Check specific column types
        column_types = {col.name: col.data_type for col in schema.columns}
        self.assertEqual(column_types["id"], "INTEGER")
        self.assertEqual(column_types["personal_info.name"], "VARCHAR")
        self.assertEqual(column_types["employment.salary"], "FLOAT")
        self.assertEqual(column_types["employment.is_active"], "BOOLEAN")

    def test_llm_enhanced_schema_extraction(self):
        """Test schema extraction with LLM enhancement."""
        result = run_schema_extractor(
            file_path=self.test_json_path,
            table_name="employees",
            llm_enhanced=True
        )
        
        self.assertIsNotNone(result)
        self.assertIn("schema", result)
        self.assertIn("llm_suggestions", result)
        self.assertNotIn("error", result)
        
        # Verify LLM suggestions format
        suggestions = result["llm_suggestions"]
        self.assertIsInstance(suggestions, str)
        self.assertTrue(len(suggestions) > 0)

    def test_invalid_json_file(self):
        """Test handling of invalid JSON file."""
        invalid_path = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_path, 'w') as f:
            f.write("invalid json content")
            
        result = run_schema_extractor(
            file_path=invalid_path,
            table_name="test"
        )
        
        self.assertIn("error", result)
        self.assertIsNone(result.get("schema"))

    def test_missing_file(self):
        """Test handling of missing file."""
        result = run_schema_extractor(
            file_path="nonexistent.json",
            table_name="test"
        )
        
        self.assertIn("error", result)
        self.assertIsNone(result.get("schema"))

    def test_empty_json(self):
        """Test handling of empty JSON file."""
        empty_path = os.path.join(self.temp_dir, "empty.json")
        with open(empty_path, 'w') as f:
            f.write("{}")
            
        result = run_schema_extractor(
            file_path=empty_path,
            table_name="test"
        )
        
        self.assertIn("error", result)
        self.assertIsNone(result.get("schema"))

if __name__ == '__main__':
    unittest.main() 