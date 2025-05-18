# Building Your First LangGraph Workflow: A Step-by-Step Guide

## Introduction

LangGraph is a powerful framework for building complex workflows with language models. This tutorial will guide you through building a documentation generation workflow, starting from basic concepts and gradually adding more advanced features.

## Prerequisites

Before we begin, make sure you have:

- Python 3.8+
- LangGraph installed (`pip install langgraph`)
- An OpenAI API key
- Basic understanding of Python

## Step 1: Basic Graph Structure

Let's start with the simplest possible graph - a linear workflow that takes input and produces output.

```python
from typing import TypedDict, Annotated
from langgraph.graph import Graph, StateGraph

# Define our state
class WorkflowState(TypedDict):
    input: str
    output: str

# Create a simple node
def process_input(state: WorkflowState) -> WorkflowState:
    state["output"] = f"Processed: {state['input']}"
    return state

# Build the graph
workflow = StateGraph(WorkflowState)
workflow.add_node("process", process_input)
workflow.set_entry_point("process")
workflow.set_finish_point("process")

# Compile and run
app = workflow.compile()
result = app.invoke({"input": "Hello World"})
print(result["output"])  # Output: "Processed: Hello World"
```

**Key Concepts:**

- `StateGraph`: The main graph class that manages the workflow
- `TypedDict`: Used to define the structure of our state
- Nodes: Functions that process the state
- Entry/Finish points: Define where the workflow starts and ends

## Step 2: Adding Conditional Routing

Now let's add some decision-making to our workflow.

```python
from typing import Literal

class WorkflowState(TypedDict):
    input: str
    output: str
    route: Literal["A", "B"]

def process_a(state: WorkflowState) -> WorkflowState:
    state["output"] = f"Route A: {state['input']}"
    return state

def process_b(state: WorkflowState) -> WorkflowState:
    state["output"] = f"Route B: {state['input']}"
    return state

def decide_route(state: WorkflowState) -> Literal["A", "B"]:
    return "A" if len(state["input"]) > 5 else "B"

# Build the graph
workflow = StateGraph(WorkflowState)
workflow.add_node("process_a", process_a)
workflow.add_node("process_b", process_b)
workflow.add_conditional_edges(
    "start",
    decide_route,
    {
        "A": "process_a",
        "B": "process_b"
    }
)
workflow.set_entry_point("start")
workflow.set_finish_point("process_a")
workflow.set_finish_point("process_b")

# Run the workflow
app = workflow.compile()
result = app.invoke({"input": "Hello", "route": None})
print(result["output"])  # Output: "Route B: Hello"
```

**Key Concepts:**

- Conditional edges: Allow branching based on conditions
- Literal types: Define specific values for routing
- Multiple finish points: Workflow can end in different states

## Step 3: Adding Human Interaction

Let's add a step where a human can review and modify the output.

```python
def human_review(state: WorkflowState) -> WorkflowState:
    print(f"\nCurrent output: {state['output']}")
    response = input("Do you want to modify the output? (y/n): ")
    if response.lower() == 'y':
        new_output = input("Enter new output: ")
        state["output"] = new_output
    return state

# Update the graph
workflow = StateGraph(WorkflowState)
workflow.add_node("process", process_input)
workflow.add_node("review", human_review)
workflow.add_edge("process", "review")
workflow.set_entry_point("process")
workflow.set_finish_point("review")
```

**Key Concepts:**

- Human-in-the-loop: Adding manual review steps
- State modification: How to update state based on human input
- Edge connections: How to chain nodes together

## Step 4: Integrating with LLMs

Now let's add an LLM to generate content.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_template(
    "Generate documentation for: {input}"
)

def generate_docs(state: WorkflowState) -> WorkflowState:
    chain = prompt | llm
    state["output"] = chain.invoke({"input": state["input"]}).content
    return state

# Update the graph
workflow = StateGraph(WorkflowState)
workflow.add_node("generate", generate_docs)
workflow.add_node("review", human_review)
workflow.add_edge("generate", "review")
workflow.set_entry_point("generate")
workflow.set_finish_point("review")
```

**Key Concepts:**

- LLM integration: Using LangChain with LangGraph
- Prompt templates: Creating reusable prompts
- Chain composition: Combining prompts and LLMs

## Step 5: Building a Complete Documentation Workflow

Let's combine everything into a complete documentation generation workflow.

```python
class DocState(TypedDict):
    topic: str
    documentation: str
    validation_feedback: str
    review_feedback: str
    is_approved: bool

def generate_documentation(state: DocState) -> DocState:
    prompt = ChatPromptTemplate.from_template(
        "Generate comprehensive documentation for {topic}"
    )
    chain = prompt | llm
    state["documentation"] = chain.invoke({"topic": state["topic"]}).content
    return state

def validate_documentation(state: DocState) -> DocState:
    prompt = ChatPromptTemplate.from_template(
        "Validate this documentation for {topic}:\n{doc}"
    )
    chain = prompt | llm
    state["validation_feedback"] = chain.invoke({
        "topic": state["topic"],
        "doc": state["documentation"]
    }).content
    return state

def human_review(state: DocState) -> DocState:
    print(f"\nDocumentation Preview:\n{state['documentation']}")
    print(f"\nValidation Feedback:\n{state['validation_feedback']}")
    response = input("Approve documentation? (y/n): ")
    state["is_approved"] = response.lower() == 'y'
    return state

# Build the complete workflow
workflow = StateGraph(DocState)
workflow.add_node("generate", generate_documentation)
workflow.add_node("validate", validate_documentation)
workflow.add_node("review", human_review)
workflow.add_edge("generate", "validate")
workflow.add_edge("validate", "review")
workflow.set_entry_point("generate")
workflow.set_finish_point("review")

# Run the workflow
app = workflow.compile()
result = app.invoke({
    "topic": "AWS Lambda Functions",
    "documentation": "",
    "validation_feedback": "",
    "review_feedback": "",
    "is_approved": False
})
```

**Key Concepts:**

- Complex state management: Handling multiple pieces of information
- Multi-step workflows: Combining generation, validation, and review
- Conditional completion: Workflow can end based on human approval

## Best Practices

1. **State Management**

   - Keep state types clear and well-defined
   - Use TypedDict for type safety
   - Minimize state mutations

2. **Node Design**

   - Make nodes focused and single-purpose
   - Handle errors gracefully
   - Document input/output expectations

3. **Workflow Structure**

   - Plan the workflow before implementation
   - Use meaningful node names
   - Consider error handling paths

4. **Testing**
   - Test each node independently
   - Test the complete workflow
   - Mock external services

## Next Steps

1. Add error handling and retries
2. Implement parallel processing
3. Add persistence for long-running workflows
4. Create a web interface for human interactions

## Handling User Input in Python Console

When building console applications, there are several ways to handle user input. Here are the most common methods:

### 1. Basic Input with `input()`

The simplest way to get user input is using the built-in `input()` function:

```python
# Basic input
name = input("Enter your name: ")
print(f"Hello, {name}!")

# Input with type conversion
age = int(input("Enter your age: "))
print(f"You are {age} years old")
```

**Important Notes:**

- `input()` always returns a string
- Use type conversion (e.g., `int()`, `float()`) when you need numbers
- Handle potential errors when converting types

### 2. Input with Validation

Here's how to add validation to user input:

```python
def get_valid_number(prompt: str, min_val: int = None, max_val: int = None) -> int:
    while True:
        try:
            value = int(input(prompt))
            if min_val is not None and value < min_val:
                print(f"Value must be at least {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Value must be at most {max_val}")
                continue
            return value
        except ValueError:
            print("Please enter a valid number")

# Usage
age = get_valid_number("Enter your age (1-120): ", 1, 120)
```

### 3. Multiple Choice Input

For multiple choice scenarios:

```python
def get_choice(prompt: str, options: list[str]) -> str:
    while True:
        print(prompt)
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")

        try:
            choice = int(input("Enter your choice: "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            print(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("Please enter a valid number")

# Usage
options = ["Generate Documentation", "Review Documentation", "Exit"]
choice = get_choice("What would you like to do?", options)
print(f"You chose: {choice}")
```

### 4. Yes/No Questions

For simple yes/no questions:

```python
def get_yes_no(prompt: str) -> bool:
    while True:
        response = input(f"{prompt} (y/n): ").lower()
        if response in ['y', 'yes']:
            return True
        if response in ['n', 'no']:
            return False
        print("Please enter 'y' or 'n'")

# Usage
should_proceed = get_yes_no("Do you want to continue?")
if should_proceed:
    print("Continuing...")
else:
    print("Stopping...")
```

### 5. Input with Timeout

For scenarios where you want to timeout if the user doesn't respond:

```python
import signal
from typing import Any

class TimeoutError(Exception):
    pass

def timeout_handler(signum: int, frame: Any) -> None:
    raise TimeoutError()

def get_input_with_timeout(prompt: str, timeout: int = 10) -> str:
    # Set the signal handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)  # Set timeout in seconds

    try:
        response = input(prompt)
        signal.alarm(0)  # Disable the alarm
        return response
    except TimeoutError:
        print("\nTimeout! No input received.")
        return ""

# Usage
try:
    response = get_input_with_timeout("Enter something (10 seconds): ")
    if response:
        print(f"You entered: {response}")
    else:
        print("No input received")
except KeyboardInterrupt:
    print("\nOperation cancelled by user")
```

### Best Practices for Console Input

1. **Always Validate Input**

   - Check for correct types
   - Validate ranges and constraints
   - Handle edge cases

2. **Provide Clear Prompts**

   - Be specific about what you're asking for
   - Show valid options when applicable
   - Include examples if helpful

3. **Handle Errors Gracefully**

   - Catch and handle exceptions
   - Provide helpful error messages
   - Allow retries when appropriate

4. **Consider User Experience**

   - Use consistent formatting
   - Provide clear instructions
   - Allow for cancellation (Ctrl+C)

5. **Document Your Functions**
   - Include docstrings
   - Document expected input types
   - Document return values

## Understanding Chain.invoke()

The `chain.invoke()` method is a fundamental part of LangChain that executes a chain of operations. Let's break down how it works:

### Basic Structure

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Create a prompt template
prompt = ChatPromptTemplate.from_template(
    "Generate documentation for {topic}"
)

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Create a chain
chain = prompt | llm

# Invoke the chain
result = chain.invoke({"topic": "AWS Lambda"})
```

### Input Format

The input to `chain.invoke()` is a dictionary containing the variables needed by the prompt template:

```python
# Example inputs
input_data = {
    "topic": "AWS Lambda",  # Matches {topic} in the template
    "format": "markdown",   # Additional variables can be added
    "length": "brief"       # These will be available in the template
}

# Updated template using all variables
prompt = ChatPromptTemplate.from_template(
    "Generate {length} documentation in {format} format for {topic}"
)

# The input dictionary keys must match the template variables
result = chain.invoke(input_data)
```

### Output Format

The output from `chain.invoke()` is a `ChatGeneration` object with several useful properties:

```python
# Get the raw response
response = chain.invoke({"topic": "AWS Lambda"})

# Access different parts of the response
print(response.content)        # The actual text content
print(response.generation_info) # Additional info about the generation
print(response.message)        # The full message object
```

### Common Use Cases

1. **Simple Text Generation**

```python
# Basic text generation
result = chain.invoke({"topic": "AWS Lambda"})
print(result.content)  # The generated documentation
```

2. **Structured Output**

```python
# Using a structured prompt
prompt = ChatPromptTemplate.from_template("""
Generate documentation for {topic} in the following format:
Title: [title]
Description: [description]
Examples: [examples]
""")

result = chain.invoke({"topic": "AWS Lambda"})
# The output will follow the specified structure
```

3. **Multiple Inputs**

```python
# Chain with multiple inputs
prompt = ChatPromptTemplate.from_template("""
Compare {topic1} and {topic2}:
- Similarities: [similarities]
- Differences: [differences]
""")

result = chain.invoke({
    "topic1": "AWS Lambda",
    "topic2": "AWS ECS"
})
```

### Error Handling

It's important to handle potential errors when using `chain.invoke()`:

```python
try:
    result = chain.invoke({"topic": "AWS Lambda"})
    print(result.content)
except Exception as e:
    print(f"Error generating content: {e}")
```

### Best Practices

1. **Input Validation**

   - Ensure all required template variables are provided
   - Validate input data types and formats
   - Handle missing or invalid inputs gracefully

2. **Output Processing**

   - Always check for successful completion
   - Process the output based on your needs
   - Handle different response formats

3. **Error Handling**

   - Catch and handle exceptions
   - Provide meaningful error messages
   - Implement retry logic when appropriate

4. **Performance Considerations**
   - Cache results when possible
   - Batch process when dealing with multiple inputs
   - Monitor token usage and costs

### Example: Complete Documentation Chain

Here's a complete example showing input, processing, and output handling:

```python
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def generate_documentation(input_data: Dict[str, Any]) -> str:
    # Define the prompt template
    prompt = ChatPromptTemplate.from_template("""
    Generate comprehensive documentation for {topic}.
    Include:
    1. Overview
    2. Key Features
    3. Use Cases
    4. Code Examples

    Format: {format}
    Level: {level}
    """)

    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # Create the chain
    chain = prompt | llm

    try:
        # Invoke the chain
        result = chain.invoke(input_data)

        # Process the output
        if result and result.content:
            return result.content
        else:
            raise ValueError("No content generated")

    except Exception as e:
        print(f"Error generating documentation: {e}")
        return ""

# Usage
input_data = {
    "topic": "AWS Lambda",
    "format": "markdown",
    "level": "intermediate"
}

documentation = generate_documentation(input_data)
print(documentation)
```

This example shows:

- Input validation and processing
- Error handling
- Output processing
- A complete workflow from input to output

## Handling Missing Variables in Prompts

When using `chain.invoke()`, it's crucial to handle cases where variables in the prompt template are missing from the input dictionary. Let's explore what happens and how to handle it:

### What Happens When Variables Are Missing?

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Template with multiple variables
prompt = ChatPromptTemplate.from_template("""
Generate documentation for {topic} in {format} format.
Include {sections} sections.
Level: {level}
""")

llm = ChatOpenAI(model="gpt-3.5-turbo")
chain = prompt | llm

# Missing 'level' and 'sections' variables
try:
    result = chain.invoke({
        "topic": "AWS Lambda",
        "format": "markdown"
    })
except Exception as e:
    print(f"Error: {e}")  # Will raise KeyError for missing variables
```

### Handling Missing Variables

1. **Default Values in Template**

```python
# Using default values in the template
prompt = ChatPromptTemplate.from_template("""
Generate documentation for {topic} in {format} format.
Include {sections} sections.
Level: {level}
""",
default_values={
    "format": "markdown",
    "sections": "all",
    "level": "intermediate"
})

# Now we only need to provide 'topic'
result = chain.invoke({"topic": "AWS Lambda"})
```

2. **Input Validation**

```python
def validate_input(input_data: dict, required_vars: list) -> dict:
    # Check for missing required variables
    missing = [var for var in required_vars if var not in input_data]
    if missing:
        raise ValueError(f"Missing required variables: {missing}")
    return input_data

# Usage
required_vars = ["topic", "format"]
input_data = {"topic": "AWS Lambda"}
try:
    validated_data = validate_input(input_data, required_vars)
    result = chain.invoke(validated_data)
except ValueError as e:
    print(f"Validation error: {e}")
```

3. **Fallback Values**

```python
def get_input_with_fallbacks(input_data: dict) -> dict:
    defaults = {
        "format": "markdown",
        "sections": "all",
        "level": "intermediate",
        "style": "technical"
    }

    # Merge defaults with input data (input data takes precedence)
    return {**defaults, **input_data}

# Usage
input_data = {"topic": "AWS Lambda"}
complete_data = get_input_with_fallbacks(input_data)
result = chain.invoke(complete_data)
```

4. **Dynamic Prompt Building**

```python
def build_prompt(input_data: dict) -> str:
    # Start with base template
    template = "Generate documentation for {topic}"

    # Add optional sections based on available data
    if "format" in input_data:
        template += " in {format} format"
    if "sections" in input_data:
        template += " with {sections} sections"

    return template

# Usage
input_data = {"topic": "AWS Lambda", "format": "markdown"}
prompt = ChatPromptTemplate.from_template(build_prompt(input_data))
result = chain.invoke(input_data)
```

### Best Practices for Handling Missing Variables

1. **Document Required Variables**

```python
def generate_documentation(input_data: dict) -> str:
    """
    Generate documentation using the provided input data.

    Required variables:
    - topic: The subject to document

    Optional variables:
    - format: Output format (default: markdown)
    - sections: Number of sections (default: all)
    - level: Complexity level (default: intermediate)
    """
    # ... implementation ...
```

2. **Validate Early**

```python
def validate_and_process_input(input_data: dict) -> dict:
    required = ["topic"]
    optional = {
        "format": "markdown",
        "sections": "all",
        "level": "intermediate"
    }

    # Check required variables
    missing = [var for var in required if var not in input_data]
    if missing:
        raise ValueError(f"Missing required variables: {missing}")

    # Add defaults for optional variables
    return {**optional, **input_data}
```

3. **Provide Clear Error Messages**

```python
def handle_missing_variables(input_data: dict, template: str) -> None:
    import re
    # Find all variables in the template
    variables = re.findall(r'\{(\w+)\}', template)

    # Check for missing variables
    missing = [var for var in variables if var not in input_data]
    if missing:
        raise ValueError(
            f"Missing variables in input: {missing}\n"
            f"Required variables: {variables}\n"
            f"Provided variables: {list(input_data.keys())}"
        )
```

4. **Use Type Hints and Validation**

```python
from typing import TypedDict, Optional
from pydantic import BaseModel

class DocumentationInput(BaseModel):
    topic: str
    format: Optional[str] = "markdown"
    sections: Optional[str] = "all"
    level: Optional[str] = "intermediate"

def process_input(input_data: dict) -> DocumentationInput:
    try:
        return DocumentationInput(**input_data)
    except Exception as e:
        raise ValueError(f"Invalid input data: {e}")
```

### Complete Example with Error Handling

```python
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def generate_documentation(input_data: Dict[str, Any]) -> str:
    # Define required and optional variables
    required_vars = ["topic"]
    optional_vars = {
        "format": "markdown",
        "sections": "all",
        "level": "intermediate"
    }

    # Validate input
    missing = [var for var in required_vars if var not in input_data]
    if missing:
        raise ValueError(f"Missing required variables: {missing}")

    # Add defaults for optional variables
    complete_data = {**optional_vars, **input_data}

    # Create prompt template
    prompt = ChatPromptTemplate.from_template("""
    Generate documentation for {topic} in {format} format.
    Include {sections} sections.
    Level: {level}
    """)

    # Initialize chain
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    chain = prompt | llm

    try:
        # Generate documentation
        result = chain.invoke(complete_data)
        return result.content
    except Exception as e:
        raise RuntimeError(f"Failed to generate documentation: {e}")

# Usage examples
try:
    # Valid input
    result = generate_documentation({
        "topic": "AWS Lambda",
        "format": "markdown"
    })
    print(result)

    # Invalid input (missing required variable)
    result = generate_documentation({
        "format": "markdown"
    })
except ValueError as e:
    print(f"Input error: {e}")
except RuntimeError as e:
    print(f"Generation error: {e}")
```

This example shows:

- Required vs optional variable handling
- Input validation
- Default value management
- Comprehensive error handling
- Type safety with type hints

## Understanding ChatPromptTemplate

`ChatPromptTemplate` is a class in LangChain that helps create structured prompts for chat models. Here's a concise overview:

### Why Use from_template?

`from_template` is a factory method that creates a template from a string with variables:

```python
from langchain_core.prompts import ChatPromptTemplate

# Simple template
template = ChatPromptTemplate.from_template(
    "Generate documentation for {topic}"
)

# Multi-message template
template = ChatPromptTemplate.from_template([
    ("system", "You are a helpful documentation assistant."),
    ("human", "Generate documentation for {topic}")
])
```

### Alternative Ways to Create Templates

1. **Using Messages Directly**

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# Create individual message templates
system_template = SystemMessagePromptTemplate.from_template(
    "You are a helpful {role}"
)
human_template = HumanMessagePromptTemplate.from_template(
    "Generate documentation for {topic}"
)

# Combine them
template = ChatPromptTemplate.from_messages([
    system_template,
    human_template
])
```

2. **Using PromptValue**

```python
from langchain_core.prompts import ChatPromptValue

# Create a prompt value directly
prompt_value = ChatPromptValue(
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "human", "content": "Generate docs for {topic}"}
    ]
)
```

3. **Using String Template**

```python
from langchain_core.prompts import StringPromptTemplate

# Create a string template
string_template = StringPromptTemplate.from_template(
    "Generate documentation for {topic}"
)

# Convert to chat template
chat_template = ChatPromptTemplate.from_template(string_template)
```

### Key Differences

1. **from_template**

   - Simplest way to create templates
   - Good for single-message prompts
   - Easy to use with variables

2. **from_messages**

   - More control over message structure
   - Better for multi-message conversations
   - Can mix different message types

3. **PromptValue**
   - Direct control over message format
   - Useful for custom message structures
   - More verbose but flexible

### When to Use Which?

- Use `from_template` for:

  - Simple prompts
  - Single-message interactions
  - Quick prototyping

- Use `from_messages` for:

  - Complex conversations
  - Multi-role interactions
  - Structured prompts

- Use `PromptValue` for:
  - Custom message formats
  - Advanced use cases
  - When you need full control

### Example: Choosing the Right Approach

```python
# Simple case - use from_template
simple_prompt = ChatPromptTemplate.from_template(
    "Generate docs for {topic}"
)

# Complex case - use from_messages
complex_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}"),
    ("human", "Generate docs for {topic}"),
    ("assistant", "I'll help you with that."),
    ("human", "Make it {style}")
])

# Custom case - use PromptValue
custom_prompt = ChatPromptValue(
    messages=[
        {"role": "system", "content": "You are {role}"},
        {"role": "user", "content": "Generate {topic}"},
        {"role": "assistant", "content": "I'll help"},
        {"role": "user", "content": "Make it {style}"}
    ]
)
```

### Best Practices

1. Start with `from_template` for simple cases
2. Use `from_messages` when you need structure
3. Only use `PromptValue` when necessary
4. Keep templates readable and maintainable
5. Document template variables and their purposes

## Creating Structured Prompts and Controlling Output Format

### 1. Basic Structured Prompt

````python
from langchain_core.prompts import ChatPromptTemplate

# Simple structured prompt
template = ChatPromptTemplate.from_template("""
Generate documentation for {topic} in the following format:

Title: [Title]
Description: [Brief description]
Features:
- [Feature 1]
- [Feature 2]
- [Feature 3]

Examples:
```python
[Code example]
````

""")

# Usage

result = chain.invoke({"topic": "AWS Lambda"})

````

### 2. Markdown Format

```python
# Markdown structured prompt
template = ChatPromptTemplate.from_template("""
Generate documentation for {topic} in markdown format.

# {topic}

## Overview
[Provide overview]

## Features
### Core Features
- [Feature 1]
- [Feature 2]

### Advanced Features
- [Feature 3]
- [Feature 4]

## Examples
```python
# Example 1
[Code example]

# Example 2
[Code example]
````

## Best Practices

1. [Best practice 1]
2. [Best practice 2]
   """)

````

### 3. JSON Output Format

```python
# JSON structured prompt
template = ChatPromptTemplate.from_template("""
Generate documentation for {topic} in JSON format with the following structure:
{{
    "title": "string",
    "description": "string",
    "features": ["string"],
    "examples": [
        {{
            "title": "string",
            "code": "string",
            "explanation": "string"
        }}
    ],
    "best_practices": ["string"]
}}

Return ONLY the JSON object, no additional text.
""")

# Parse JSON output
import json
result = chain.invoke({"topic": "AWS Lambda"})
data = json.loads(result.content)
````

### 4. YAML Output Format

```python
# YAML structured prompt
template = ChatPromptTemplate.from_template("""
Generate documentation for {topic} in YAML format:

title: string
description: string
features:
  - string
  - string
examples:
  - title: string
    code: string
    explanation: string
best_practices:
  - string
  - string

Return ONLY the YAML content, no additional text.
""")
```

### 5. Custom Format with Examples

````python
# Custom format with examples
template = ChatPromptTemplate.from_template("""
Generate documentation for {topic} following this exact format:

DOCUMENTATION START
TITLE: {topic}

DESCRIPTION:
[2-3 sentences describing the topic]

KEY FEATURES:
1. [Feature 1]
   - [Sub-feature]
   - [Sub-feature]
2. [Feature 2]
   - [Sub-feature]
   - [Sub-feature]

CODE EXAMPLES:
Example 1: [Example name]
```python
[Code]
````

Explanation: [Brief explanation]

Example 2: [Example name]

```python
[Code]
```

Explanation: [Brief explanation]

BEST PRACTICES:

- [Practice 1]
- [Practice 2]
- [Practice 3]

DOCUMENTATION END
""")

````

### 6. Format with Validation

```python
# Format with validation instructions
template = ChatPromptTemplate.from_template("""
Generate documentation for {topic} following these rules:

1. Format Requirements:
   - Use markdown syntax
   - Include code blocks with language specification
   - Use proper heading hierarchy (h1, h2, h3)
   - Include bullet points for lists

2. Content Requirements:
   - Start with a clear title
   - Include a brief overview
   - List key features with descriptions
   - Provide at least 2 code examples
   - Include best practices
   - End with a summary

3. Style Requirements:
   - Use clear, concise language
   - Include technical terms where appropriate
   - Maintain consistent formatting
   - Use proper markdown syntax for code blocks

Generate the documentation following these requirements exactly.
""")
````

### 7. Multi-Part Documentation

```python
# Multi-part documentation
template = ChatPromptTemplate.from_template("""
Generate comprehensive documentation for {topic} in multiple sections:

SECTION 1: INTRODUCTION
[Provide introduction]

SECTION 2: CORE CONCEPTS
[Explain core concepts]

SECTION 3: IMPLEMENTATION
[Provide implementation details]

SECTION 4: EXAMPLES
[Include examples]

SECTION 5: TROUBLESHOOTING
[Include troubleshooting guide]

Each section should be clearly marked with its header.
Use markdown formatting throughout.
""")
```

### Best Practices for Format Control

1. **Be Explicit**

   - Clearly specify the desired format
   - Provide examples of the expected structure
   - Include formatting rules

2. **Use Placeholders**

   - Show the expected structure with placeholders
   - Make it clear what goes where
   - Include type hints for content

3. **Add Validation Rules**

   - Specify content requirements
   - Define formatting rules
   - Include style guidelines

4. **Handle Edge Cases**
   - Specify how to handle missing information
   - Define fallback behaviors
   - Include error handling instructions

### Example: Complete Format Control

````python
def generate_formatted_docs(topic: str, format_type: str = "markdown") -> str:
    # Define format-specific templates
    templates = {
        "markdown": """
        # {topic}

        ## Overview
        [Provide overview]

        ## Features
        - [Feature 1]
        - [Feature 2]

        ## Examples
        ```python
        [Code example]
        ```
        """,

        "json": """
        Generate documentation in JSON format:
        {{
            "title": "{topic}",
            "overview": "string",
            "features": ["string"],
            "examples": ["string"]
        }}
        """,

        "yaml": """
        title: {topic}
        overview: string
        features:
          - string
        examples:
          - string
        """
    }

    # Create template
    template = ChatPromptTemplate.from_template(templates[format_type])

    # Generate documentation
    result = chain.invoke({"topic": topic})

    # Post-process based on format
    if format_type == "json":
        import json
        return json.dumps(json.loads(result.content), indent=2)
    return result.content

# Usage
markdown_docs = generate_formatted_docs("AWS Lambda", "markdown")
json_docs = generate_formatted_docs("AWS Lambda", "json")
yaml_docs = generate_formatted_docs("AWS Lambda", "yaml")
````

This example shows:

- Multiple output formats
- Format-specific templates
- Post-processing for structured formats
- Flexible format selection

## Adding Output Format Control to the Chain

### 1. Basic Output Parser

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Create the chain with output parser
chain = (
    ChatPromptTemplate.from_template("Generate docs for {topic}")
    | ChatOpenAI(model="gpt-3.5-turbo")
    | StrOutputParser()
)

# Usage
result = chain.invoke({"topic": "AWS Lambda"})
```

### 2. JSON Output Parser

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

# Define output schema
class Documentation(BaseModel):
    title: str = Field(description="Title of the documentation")
    description: str = Field(description="Brief description")
    features: List[str] = Field(description="List of features")
    examples: List[str] = Field(description="List of examples")

# Create the chain with JSON parser
chain = (
    ChatPromptTemplate.from_template("""
    Generate documentation for {topic} in JSON format.
    {format_instructions}
    """)
    | ChatOpenAI(model="gpt-3.5-turbo")
    | JsonOutputParser(pydantic_object=Documentation)
)

# Usage
result = chain.invoke({
    "topic": "AWS Lambda",
    "format_instructions": JsonOutputParser(pydantic_object=Documentation).get_format_instructions()
})
```

### 3. Structured Output Parser

```python
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema

# Define output schema
response_schemas = [
    ResponseSchema(
        name="title",
        description="The title of the documentation"
    ),
    ResponseSchema(
        name="overview",
        description="A brief overview of the topic"
    ),
    ResponseSchema(
        name="prerequisites",
        description="List of prerequisites needed to understand the topic"
    ),
    ResponseSchema(
        name="features",
        description="List of main features with descriptions",
        type="List[Dict[str, str]]"
    ),
    ResponseSchema(
        name="examples",
        description="List of code examples with explanations",
        type="List[Dict[str, str]]"
    ),
    ResponseSchema(
        name="best_practices",
        description="List of best practices to follow"
    ),
    ResponseSchema(
        name="common_issues",
        description="List of common issues and their solutions",
        type="List[Dict[str, str]]"
    )
]

# Create parser
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Create the chain
chain = (
    ChatPromptTemplate.from_template("""
    Generate documentation for {topic}.
    {format_instructions}
    """)
    | ChatOpenAI(model="gpt-3.5-turbo")
    | output_parser
)

# Usage
result = chain.invoke({
    "topic": "AWS Lambda",
    "format_instructions": output_parser.get_format_instructions()
})
```

### 4. Custom Output Parser

```python
from langchain_core.output_parsers import BaseOutputParser
from typing import List
import json

class CustomDocumentationParser(BaseOutputParser):
    def parse(self, text: str) -> dict:
        # Custom parsing logic
        try:
            # Split into sections
            sections = text.split("\n\n")
            result = {
                "title": sections[0].strip(),
                "description": sections[1].strip(),
                "features": [f.strip() for f in sections[2].split("\n") if f.strip()],
                "examples": [e.strip() for e in sections[3].split("\n") if e.strip()]
            }
            return result
        except Exception as e:
            raise ValueError(f"Failed to parse output: {e}")

# Create the chain
chain = (
    ChatPromptTemplate.from_template("""
    Generate documentation for {topic} in the following format:

    Title: [title]
    Description: [description]
    Features:
    - [feature1]
    - [feature2]
    Examples:
    - [example1]
    - [example2]
    """)
    | ChatOpenAI(model="gpt-3.5-turbo")
    | CustomDocumentationParser()
)

# Usage
result = chain.invoke({"topic": "AWS Lambda"})
```

### 5. Multiple Output Parsers

```python
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser
)
from langchain_core.pydantic_v1 import BaseModel, Field

# Define schema
class Documentation(BaseModel):
    title: str = Field(description="Title")
    content: str = Field(description="Main content")
    metadata: dict = Field(description="Additional metadata")

# Create parsers
str_parser = StrOutputParser()
json_parser = JsonOutputParser()
pydantic_parser = PydanticOutputParser(pydantic_object=Documentation)

# Create chain with multiple parsers
chain = (
    ChatPromptTemplate.from_template("""
    Generate documentation for {topic}.
    {format_instructions}
    """)
    | ChatOpenAI(model="gpt-3.5-turbo")
    | {
        "raw": str_parser,
        "json": json_parser,
        "structured": pydantic_parser
    }
)

# Usage
result = chain.invoke({
    "topic": "AWS Lambda",
    "format_instructions": pydantic_parser.get_format_instructions()
})
```

### 6. Output Format with Validation

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing import List

class ValidatedDocumentation(BaseModel):
    title: str = Field(description="Title of the documentation")
    description: str = Field(description="Brief description")
    features: List[str] = Field(description="List of features")
    examples: List[str] = Field(description="List of examples")

    @validator('features')
    def validate_features(cls, v):
        if len(v) < 2:
            raise ValueError("Must have at least 2 features")
        return v

    @validator('examples')
    def validate_examples(cls, v):
        if len(v) < 1:
            raise ValueError("Must have at least 1 example")
        return v

# Create parser
parser = PydanticOutputParser(pydantic_object=ValidatedDocumentation)

# Create chain
chain = (
    ChatPromptTemplate.from_template("""
    Generate documentation for {topic}.
    {format_instructions}
    """)
    | ChatOpenAI(model="gpt-3.5-turbo")
    | parser
)

# Usage
try:
    result = chain.invoke({
        "topic": "AWS Lambda",
        "format_instructions": parser.get_format_instructions()
    })
except ValueError as e:
    print(f"Validation error: {e}")
```

### Best Practices for Output Parsing

1. **Choose the Right Parser**

   - Use `StrOutputParser` for simple text
   - Use `JsonOutputParser` for structured data
   - Use `PydanticOutputParser` for validated data
   - Create custom parsers for specific formats

2. **Validate Output**

   - Add validation rules
   - Handle parsing errors
   - Provide fallback values

3. **Format Instructions**

   - Include clear format instructions
   - Show examples in the prompt
   - Specify required fields

4. **Error Handling**
   - Catch parsing errors
   - Provide helpful error messages
   - Implement retry logic

### Example: Complete Output Control

```python
from typing import Dict, Any
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator

class DocumentationOutput(BaseModel):
    title: str = Field(description="Documentation title")
    content: str = Field(description="Main content in markdown")
    metadata: Dict[str, Any] = Field(
        description="Additional metadata like tags, categories"
    )

    @validator('title')
    def validate_title(cls, v):
        if len(v) < 3:
            raise ValueError("Title must be at least 3 characters")
        return v

def create_documentation_chain(format_type: str = "markdown"):
    # Create parser
    parser = PydanticOutputParser(pydantic_object=DocumentationOutput)

    # Create prompt template
    template = ChatPromptTemplate.from_template("""
    Generate documentation for {topic} in {format_type} format.

    Requirements:
    - Title must be clear and descriptive
    - Content must be in {format_type} format
    - Include at least 2 examples
    - Add relevant metadata

    {format_instructions}
    """)

    # Create chain
    return (
        template
        | ChatOpenAI(model="gpt-3.5-turbo")
        | parser
    )

# Usage
chain = create_documentation_chain("markdown")
try:
    result = chain.invoke({
        "topic": "AWS Lambda",
        "format_type": "markdown",
        "format_instructions": parser.get_format_instructions()
    })
    print(result.title)
    print(result.content)
    print(result.metadata)
except ValueError as e:
    print(f"Error: {e}")
```

This example shows:

- Output schema definition
- Validation rules
- Format control
- Error handling
- Complete chain creation

## Detailed Example: StructuredOutputParser

Here's a comprehensive example of using StructuredOutputParser to generate well-formatted documentation:

````python
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import List, Dict

# Define the response schemas
response_schemas = [
    ResponseSchema(
        name="title",
        description="The title of the documentation"
    ),
    ResponseSchema(
        name="overview",
        description="A brief overview of the topic"
    ),
    ResponseSchema(
        name="prerequisites",
        description="List of prerequisites needed to understand the topic"
    ),
    ResponseSchema(
        name="features",
        description="List of main features with descriptions",
        type="List[Dict[str, str]]"
    ),
    ResponseSchema(
        name="examples",
        description="List of code examples with explanations",
        type="List[Dict[str, str]]"
    ),
    ResponseSchema(
        name="best_practices",
        description="List of best practices to follow"
    ),
    ResponseSchema(
        name="common_issues",
        description="List of common issues and their solutions",
        type="List[Dict[str, str]]"
    )
]

# Create the output parser
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Get the format instructions
format_instructions = output_parser.get_format_instructions()

# Create the prompt template
template = ChatPromptTemplate.from_template("""
Generate comprehensive documentation for {topic} following this exact structure:

{format_instructions}

Additional Requirements:
1. Features should include both basic and advanced features
2. Examples should be practical and well-commented
3. Best practices should be actionable and specific
4. Common issues should include both prevention and solutions

Topic: {topic}
""")

# Create the chain
chain = (
    template
    | ChatOpenAI(model="gpt-3.5-turbo")
    | output_parser
)

# Function to generate documentation
def generate_documentation(topic: str) -> Dict:
    try:
        # Invoke the chain
        result = chain.invoke({
            "topic": topic,
            "format_instructions": format_instructions
        })

        # Validate the output
        if not result.get("title"):
            raise ValueError("Missing title in output")
        if not result.get("features"):
            raise ValueError("Missing features in output")
        if not result.get("examples"):
            raise ValueError("Missing examples in output")

        return result

    except Exception as e:
        print(f"Error generating documentation: {e}")
        return None

# Function to format the output
def format_documentation(doc: Dict) -> str:
    if not doc:
        return "Failed to generate documentation"

    output = []
    output.append(f"# {doc['title']}\n")
    output.append(f"## Overview\n{doc['overview']}\n")

    if doc['prerequisites']:
        output.append("## Prerequisites")
        for item in doc['prerequisites']:
            output.append(f"- {item}")
        output.append("")

    output.append("## Features")
    for feature in doc['features']:
        output.append(f"### {feature['name']}")
        output.append(f"{feature['description']}\n")

    output.append("## Examples")
    for example in doc['examples']:
        output.append(f"### {example['title']}")
        output.append("```python")
        output.append(example['code'])
        output.append("```")
        output.append(f"**Explanation:** {example['explanation']}\n")

    output.append("## Best Practices")
    for practice in doc['best_practices']:
        output.append(f"- {practice}")
    output.append("")

    output.append("## Common Issues and Solutions")
    for issue in doc['common_issues']:
        output.append(f"### {issue['problem']}")
        output.append(f"**Solution:** {issue['solution']}\n")

    return "\n".join(output)

# Usage example
if __name__ == "__main__":
    # Generate documentation
    doc = generate_documentation("AWS Lambda")

    if doc:
        # Format and print
        formatted_doc = format_documentation(doc)
        print(formatted_doc)

        # Save to file
        with open("lambda_documentation.md", "w") as f:
            f.write(formatted_doc)
````

### Key Components Explained

1. **Response Schemas**

   - Define the structure of the output
   - Specify types and descriptions
   - Support nested structures (List[Dict])

2. **Format Instructions**

   - Automatically generated from schemas
   - Tell the LLM how to structure the output
   - Include type information

3. **Prompt Template**

   - Combines format instructions with requirements
   - Provides additional context
   - Ensures consistent output

4. **Output Validation**

   - Checks for required fields
   - Validates structure
   - Handles errors gracefully

5. **Formatting Function**
   - Converts structured data to markdown
   - Handles different content types
   - Creates readable output

### Example Output Structure

```python
{
    "title": "AWS Lambda Documentation",
    "overview": "AWS Lambda is a serverless compute service...",
    "prerequisites": [
        "AWS Account",
        "Basic understanding of serverless concepts"
    ],
    "features": [
        {
            "name": "Automatic Scaling",
            "description": "Lambda automatically scales..."
        },
        {
            "name": "Pay-per-Use",
            "description": "You only pay for the compute time..."
        }
    ],
    "examples": [
        {
            "title": "Basic Lambda Function",
            "code": "def lambda_handler(event, context):\n    return 'Hello World'",
            "explanation": "This is a simple Lambda function..."
        }
    ],
    "best_practices": [
        "Keep functions small and focused",
        "Use environment variables for configuration"
    ],
    "common_issues": [
        {
            "problem": "Cold Start",
            "solution": "Use provisioned concurrency..."
        }
    ]
}
```

### Best Practices

1. **Schema Design**

   - Make schemas specific and clear
   - Include descriptions for each field
   - Use appropriate types

2. **Prompt Engineering**

   - Be explicit about requirements
   - Provide examples when possible
   - Include validation rules

3. **Error Handling**

   - Validate output structure
   - Handle missing fields
   - Provide helpful error messages

4. **Output Formatting**
   - Use consistent formatting
   - Handle different content types
   - Make output readable

## Conclusion

LangGraph provides a powerful way to build complex workflows with language models. By following this tutorial, you've learned how to:

- Create basic graphs
- Add conditional routing
- Integrate human review
- Use LLMs in your workflow
- Build a complete documentation generation system

Remember to start simple and gradually add complexity as you become more comfortable with the framework.

## Monitoring and Printing Graph State

### 1. Basic State Printing

```python
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated

class WorkflowState(TypedDict):
    input: str
    output: str
    step: str

def print_state(state: WorkflowState) -> WorkflowState:
    print(f"\nCurrent State:")
    print(f"Step: {state['step']}")
    print(f"Input: {state['input']}")
    print(f"Output: {state['output']}")
    return state

# Create graph with state printing
workflow = StateGraph(WorkflowState)
workflow.add_node("process", process_input)
workflow.add_node("print", print_state)
workflow.add_edge("process", "print")
workflow.set_entry_point("process")
workflow.set_finish_point("print")
```

### 2. State Logging with Timestamps

```python
from datetime import datetime
import json

def log_state(state: WorkflowState) -> WorkflowState:
    timestamp = datetime.now().isoformat()
    state_data = {
        "timestamp": timestamp,
        "step": state["step"],
        "input": state["input"],
        "output": state["output"]
    }
    print(f"\nState Log ({timestamp}):")
    print(json.dumps(state_data, indent=2))
    return state

# Add logging to workflow
workflow.add_node("log", log_state)
workflow.add_edge("print", "log")
```

### 3. State History Tracking

```python
from typing import List

class WorkflowState(TypedDict):
    input: str
    output: str
    step: str
    history: List[dict]

def track_state(state: WorkflowState) -> WorkflowState:
    state_entry = {
        "step": state["step"],
        "input": state["input"],
        "output": state["output"],
        "timestamp": datetime.now().isoformat()
    }
    state["history"].append(state_entry)
    return state

# Initialize state with history
initial_state = {
    "input": "Hello",
    "output": "",
    "step": "start",
    "history": []
}
```

### 4. Conditional State Printing

```python
def conditional_print(state: WorkflowState) -> WorkflowState:
    # Print only if output contains specific content
    if "error" in state["output"].lower():
        print(f"\nError State Detected:")
        print(f"Step: {state['step']}")
        print(f"Output: {state['output']}")
    return state

# Add conditional printing
workflow.add_node("conditional_print", conditional_print)
workflow.add_edge("log", "conditional_print")
```

### 5. State Visualization

```python
def visualize_state(state: WorkflowState) -> WorkflowState:
    print("\nWorkflow State Visualization:")
    print("=" * 50)
    print(f"Current Step: {state['step']}")
    print("-" * 50)
    print("State History:")
    for entry in state["history"]:
        print(f"\nTime: {entry['timestamp']}")
        print(f"Step: {entry['step']}")
        print(f"Input: {entry['input']}")
        print(f"Output: {entry['output']}")
    print("=" * 50)
    return state

# Add visualization
workflow.add_node("visualize", visualize_state)
workflow.add_edge("conditional_print", "visualize")
```

### 6. Complete State Monitoring Example

```python
from typing import TypedDict, List
from datetime import datetime
import json

class WorkflowState(TypedDict):
    input: str
    output: str
    step: str
    history: List[dict]
    error: str

def process_input(state: WorkflowState) -> WorkflowState:
    state["output"] = f"Processed: {state['input']}"
    state["step"] = "process"
    return state

def log_state(state: WorkflowState) -> WorkflowState:
    # Log current state
    state_entry = {
        "timestamp": datetime.now().isoformat(),
        "step": state["step"],
        "input": state["input"],
        "output": state["output"],
        "error": state.get("error", "")
    }
    state["history"].append(state_entry)

    # Print state
    print("\nState Log:")
    print(json.dumps(state_entry, indent=2))
    return state

def check_errors(state: WorkflowState) -> WorkflowState:
    if "error" in state["output"].lower():
        state["error"] = "Error detected in output"
        print("\nError Detected!")
        print(f"Step: {state['step']}")
        print(f"Error: {state['error']}")
    return state

def visualize_history(state: WorkflowState) -> WorkflowState:
    print("\nWorkflow History:")
    print("=" * 50)
    for entry in state["history"]:
        print(f"\nTime: {entry['timestamp']}")
        print(f"Step: {entry['step']}")
        if entry.get("error"):
            print(f"Error: {entry['error']}")
        print(f"Input: {entry['input']}")
        print(f"Output: {entry['output']}")
    print("=" * 50)
    return state

# Create workflow
workflow = StateGraph(WorkflowState)

# Add nodes
workflow.add_node("process", process_input)
workflow.add_node("log", log_state)
workflow.add_node("check", check_errors)
workflow.add_node("visualize", visualize_history)

# Add edges
workflow.add_edge("process", "log")
workflow.add_edge("log", "check")
workflow.add_edge("check", "visualize")

# Set entry and finish points
workflow.set_entry_point("process")
workflow.set_finish_point("visualize")

# Initialize state
initial_state = {
    "input": "Hello World",
    "output": "",
    "step": "start",
    "history": [],
    "error": ""
}

# Run workflow
app = workflow.compile()
result = app.invoke(initial_state)
```

### 7. State Monitoring with Callbacks

```python
from typing import Callable, Dict, Any

class StateMonitor:
    def __init__(self):
        self.history = []

    def on_state_change(self, state: Dict[str, Any]) -> None:
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "state": state.copy()
        })
        self.print_state(state)

    def print_state(self, state: Dict[str, Any]) -> None:
        print("\nState Change Detected:")
        print(json.dumps(state, indent=2))

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history

# Usage
monitor = StateMonitor()

def monitored_process(state: WorkflowState) -> WorkflowState:
    state["output"] = f"Processed: {state['input']}"
    state["step"] = "process"
    monitor.on_state_change(state)
    return state

# Add monitored node
workflow.add_node("monitored_process", monitored_process)
```

### Best Practices for State Monitoring

1. **State Structure**

   - Keep state structure clear and consistent
   - Include timestamps for tracking
   - Maintain history when needed

2. **Monitoring Points**

   - Add monitoring at key points
   - Include error checking
   - Track important transitions

3. **Output Format**

   - Use consistent formatting
   - Include relevant context
   - Make output readable

4. **Error Handling**

   - Monitor for errors
   - Log error states
   - Provide helpful messages

5. **Performance**
   - Minimize logging overhead
   - Use conditional logging
   - Consider batch processing
