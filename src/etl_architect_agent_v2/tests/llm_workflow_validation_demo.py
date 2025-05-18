"""Demo script for LLM workflow validation."""

import sys
import os
import asyncio
import json

# Add src to Python path
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../.."
        )
    )
)

from etl_architect_agent_v2.core.llm_workflow import (
    LLMWorkflow,
    JSONValidator,
    ValidationIssue
)


# Test cases with various JSON formatting issues
TEST_CASES = [
    # Unterminated string
    {
        "name": "Unterminated String",
        "content": '''{
            "name": "Test",
            "description": "This is a test with an unterminated string
        }'''
    },
    # Missing quotes
    {
        "name": "Missing Quotes",
        "content": '''{
            name: "Test",
            description: "This is a test with missing quotes"
        }'''
    },
    # Trailing comma
    {
        "name": "Trailing Comma",
        "content": '''{
            "name": "Test",
            "description": "This is a test with trailing comma",
        }'''
    },
    # Boolean values
    {
        "name": "Boolean Values",
        "content": '''{
            "is_active": TRUE,
            "is_deleted": FALSE,
            "is_verified": true,
            "is_blocked": false
        }'''
    },
    # Numeric values
    {
        "name": "Numeric Values",
        "content": '''{
            "id": 00123,
            "price": .99,
            "quantity": 1e3,
            "score": 0.5
        }'''
    },
    # Null values
    {
        "name": "Null Values",
        "content": '''{
            "name": "",
            "description": NULL,
            "value": null
        }'''
    },
    # Array formatting
    {
        "name": "Array Formatting",
        "content": '''{
            "tags": ["tag1" "tag2" "tag3"],
            "numbers": [1 2 3],
            "empty": [ ]
        }'''
    },
    # Object formatting
    {
        "name": "Object Formatting",
        "content": '''{
            "user" "name" "John",
            "settings" { },
            "metadata" "type" "admin"
        }'''
    },
    # Incomplete objects
    {
        "name": "Incomplete Objects",
        "content": '''{
            "columns": [
                {
                    "name": "Last Funding Type",
                    "type": "string",
                    "description": "Type of the last funding",
                    "sample_value": "Seed",
                    "quality_metrics": {
                        "completeness":
                    }
                }
            ]
        }'''
    },
    # Complex nested structure
    {
        "name": "Complex Nested Structure",
        "content": '''{
            "user": {
                "id": 001,
                "name": "John Doe",
                "is_active": TRUE,
                "settings": {
                    "theme": "dark",
                    "notifications": true,
                },
                "tags": ["admin" "user" "premium"],
                "metadata": {
                    "created_at": "",
                    "last_login": NULL
                }
            }
        }'''
    },
    # Valid JSON
    {
        "name": "Valid JSON",
        "content": '''{
            "name": "Test",
            "description": "This is a valid JSON"
        }'''
    }
]


def print_issues(issues: list[ValidationIssue]) -> None:
    """Print validation issues in a formatted way."""
    for i, issue in enumerate(issues, 1):
        print(f"\nIssue {i}:")
        print(f"  Level: {issue.level.value}")
        print(f"  Message: {issue.message}")
        if issue.field:
            print(f"  Field: {issue.field}")
        if issue.fix_suggestion:
            print(f"  Fix Suggestion: {issue.fix_suggestion}")


def print_attempt_history(history: list[dict]) -> None:
    """Print attempt history in a formatted way."""
    print("\nAttempt History:")
    for attempt in history:
        print(f"\nAttempt {attempt['attempt']}:")
        print(f"  Content: {json.dumps(attempt['content'], indent=2)}")
        if attempt.get('validation_result'):
            print(f"  Valid: {attempt['validation_result']['is_valid']}")
            if attempt['validation_result'].get('issues'):
                print("  Issues:")
                for issue in attempt['validation_result']['issues']:
                    print(f"    - {issue['message']}")
        if attempt.get('fixes_applied'):
            print(f"  Fixes Applied: {attempt['fixes_applied']}")
        print(f"  LLM Used: {attempt.get('llm_used', False)}")


async def generate_content(metadata: dict) -> dict:
    """Async generator function that returns content directly."""
    content = metadata.get("content", {})
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"raw_content": content}
    return content


async def run_validation() -> None:
    """Run validation on test cases."""
    validator = JSONValidator()
    
    for test_case in TEST_CASES:
        print("\n" + "="*80)
        print(f"Testing: {test_case['name']}")
        print("="*80)
        
        # Create workflow with async generator
        workflow = LLMWorkflow(
            validator=validator,
            generator=generate_content,
            max_attempts=3
        )
        
        # Run workflow
        result = await workflow.run({"content": test_case["content"]})
        
        # Print results
        print("\nInput JSON:")
        print(json.dumps(test_case["content"], indent=2))
        
        print("\nValidation Result:")
        if result.validation_result:
            print(f"Valid: {result.validation_result.is_valid}")
            print(f"Error Category: {result.validation_result.error_category}")
            if result.validation_result.error_details:
                print("\nError Details:")
                print(json.dumps(result.validation_result.error_details, indent=2))
            if result.validation_result.issues:
                print("\nIssues Found:")
                print_issues(result.validation_result.issues)
            if result.validation_result.fixed_content:
                print("\nFixed Content:")
                print(
                    json.dumps(
                        result.validation_result.fixed_content,
                        indent=2
                    )
                )
            if result.validation_result.fixes_applied:
                print("\nFixes Applied:")
                for fix in result.validation_result.fixes_applied:
                    print(f"  - {fix}")
        else:
            print("No validation result available")
            
        if result.error:
            print(f"\nError: {result.error}")
            
        print("\nWorkflow History:")
        for step in result.workflow_history:
            print(f"\nStep: {step['step']}")
            print(f"Timestamp: {step['timestamp']}")
            if 'error' in step:
                print(f"Error: {step['error']}")
                
        if result.attempt_history:
            print_attempt_history(result.attempt_history)
            
        print(f"\nFinal Status: {result.final_status}")
        print(f"LLM Used: {result.llm_used}")


if __name__ == "__main__":
    asyncio.run(run_validation()) 