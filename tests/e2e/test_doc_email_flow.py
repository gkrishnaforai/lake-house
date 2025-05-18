from typing import TypedDict, List, Optional
from typing_extensions import Annotated
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, BaseMessage
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
import pytest
import tempfile
import os
from src.aws_architect_agent.utils.mcp_utils import mcp_server_manager


def safe_concat(a: Optional[List], b: Optional[List]) -> List:
    """Safely concatenate two lists, handling None values."""
    a = a or []
    b = b or []
    return a + b


def safe_add(
    a: Optional[BaseMessage], 
    b: Optional[BaseMessage]
) -> BaseMessage:
    """Safely add two messages, handling None values."""
    if a is None:
        return b
    if b is None:
        return a
    return b  # Always take the newer message


def safe_str_add(a: Optional[str], b: Optional[str]) -> str:
    """Safely add two strings, handling None values."""
    if a is None:
        return b or ""
    if b is None:
        return a
    return b  # Always take the newer string


def safe_bool_add(a: Optional[bool], b: Optional[bool]) -> bool:
    """Safely add two booleans, handling None values."""
    if a is None:
        return b or False
    if b is None:
        return a
    return b  # Always take the newer value


def safe_int_add(a: Optional[int], b: Optional[int]) -> int:
    """Safely add two integers, handling None values."""
    if a is None:
        return b or 0
    if b is None:
        return a
    return b  # Always take the newer value


class DocState(TypedDict):
    """State for the documentation workflow."""
    current_message: Annotated[BaseMessage, safe_add]
    next: Annotated[str, safe_str_add]
    doc_content: Annotated[str, safe_str_add]
    validation_feedback: Annotated[str, safe_str_add]
    review_feedback: Annotated[str, safe_str_add]
    email_sent: Annotated[bool, safe_bool_add]
    saved_file: Annotated[Optional[str], safe_str_add]
    conversation_history: Annotated[List[BaseMessage], safe_concat]
    revision_count: Annotated[int, safe_int_add]
    errors: Annotated[List[str], safe_concat]
    feedback: Annotated[str, safe_str_add]


def validate_state(state: DocState) -> bool:
    """Validate the workflow state and transitions."""
    # Check required fields
    if not state.get("current_message"):
        return False
    
    # Define valid state transitions
    valid_transitions = {
        "generate": ["validate"],
        "validate": ["review"],
        "review": ["review_decision"],
        "review_decision": ["generate", "save"],
        "save": ["email"],
        "email": [END]
    }
    
    next_state = state.get("next")
    if isinstance(next_state, list):
        return all(
            n in valid_transitions.get(state.get("next", ""), []) 
            for n in next_state
        )
    return next_state in valid_transitions.get(state.get("next", ""), [])


def handle_error(state: DocState) -> DocState:
    """Handle workflow errors with proper logging and recovery."""
    error = state.get("error", "Unknown error")
    state["errors"].append(error)
    
    # Log the error
    print(f"Error in {state['next']}: {error}")
    
    # If we've had too many errors, stop
    if len(state["errors"]) > 3:
        print("Too many errors, stopping workflow")
        return {**state, "next": END}
    
    # Otherwise, retry with proper context
    print(f"Retrying from {state['next']}")
    return {**state, "next": "generate"}


def analyze_feedback(feedback: str) -> List[dict]:
    """Analyze feedback to identify issues and their severity."""
    issues = []
    critical_keywords = ["error", "incorrect", "missing", "broken", "unsafe"]
    warning_keywords = ["improve", "suggest", "consider", "recommend"]
    
    for line in feedback.split("\n"):
        if any(keyword in line.lower() for keyword in critical_keywords):
            issues.append({"text": line, "severity": "critical"})
        elif any(keyword in line.lower() for keyword in warning_keywords):
            issues.append({"text": line, "severity": "warning"})
    
    return issues


def generate_documentation(topic: str) -> str:
    """Generate documentation for a given topic."""
    # Get the MCP server
    server = mcp_server_manager.get_server("primary")
    
    # Create documentation prompt
    system_prompt = """You are a technical documentation expert.
    Generate comprehensive documentation for the given topic, including:
    
    1. Introduction and Overview
    2. Key Concepts
    3. Getting Started
    4. Configuration
    5. Best Practices
    6. Security Considerations
    7. Troubleshooting
    8. References
    
    Use markdown format and provide code examples where appropriate."""
    
    # Create the chain
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Generate documentation for: {topic}")
    ])
    
    chain = prompt | server | StrOutputParser()
    
    # Generate documentation
    documentation = chain.invoke({})
    return documentation


def validate_documentation(doc_content: str) -> str:
    """Validate the documentation content."""
    # Get the validator MCP server
    server = mcp_server_manager.get_server("validator")
    
    # Create validation prompt
    system_prompt = """You are a documentation validator.
    Review the documentation for:
    
    1. Structure and Organization:
       - Clear sections and headings
       - Logical flow of information
       - Proper markdown formatting
    
    2. Content Quality:
       - Clarity and readability
       - Code example quality
       - Completeness of information
    
    3. Technical Accuracy:
       - Correctness of technical details
       - Up-to-date information
       - Valid code examples
    
    Provide validation feedback in a structured format."""
    
    # Create the chain
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Documentation to validate:\n{doc_content}")
    ])
    
    chain = prompt | server | StrOutputParser()
    
    # Get validation feedback
    validation_feedback = chain.invoke({})
    return validation_feedback


def review_documentation(doc_content: str, validation_feedback: str) -> str:
    """Review the documentation and validation feedback."""
    # Get the reviewer MCP server
    server = mcp_server_manager.get_server("reviewer")
    
    # Create review prompt
    system_prompt = """You are a senior documentation reviewer.
    Review the documentation and validation feedback, considering:
    
    1. Overall Quality:
       - Is the documentation production-ready?
       - Does it meet industry standards?
       - Is it suitable for the target audience?
    
    2. Validation Feedback:
       - Are the validation points valid?
       - Have all validation issues been addressed?
    
    3. Technical Accuracy:
       - Are the technical details correct?
       - Are there any misleading or incorrect statements?
    
    4. Completeness:
       - Is all necessary information included?
       - Are there any gaps in the documentation?
    
    Provide detailed feedback in a structured format."""
    
    # Create the chain
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"Documentation:\n{doc_content}\n\n"
            f"Validation Feedback:\n{validation_feedback}"
        ))
    ])
    
    chain = prompt | server | StrOutputParser()
    
    # Get the review feedback
    review_feedback = chain.invoke({})
    return review_feedback


def send_email(content: str, recipient: str) -> bool:
    """Send an email with the documentation."""
    # Mock email sending
    print(f"Sending email to {recipient}...")
    print(f"Content:\n{content}")
    return True


def save_documentation(content: str) -> str:
    """Save documentation to a temporary file."""
    # Create a temporary file in the current directory
    temp_dir = os.getcwd()
    print(f"\nSaving documentation in directory: {temp_dir}")
    
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.md', delete=False, dir=temp_dir
    ) as temp_file:
        temp_file.write(content)
        print(f"Documentation saved to: {temp_file.name}")
        return temp_file.name


def get_human_approval(prompt: str, options: list[str] = ["y", "n"]) -> bool:
    """Get human approval for a step.
    
    Args:
        prompt: The question to ask the user
        options: List of valid options (default: ["y", "n"])
    
    Returns:
        bool: True if approved, False otherwise
    """
    while True:
        response = input(f"\n{prompt} ({'/'.join(options)})? ").lower()
        if response in options:
            return response == options[0]  # First option is considered "yes"
        print(f"Invalid input. Please enter one of: {', '.join(options)}")


def create_doc_generator_node():
    """Create a node that generates documentation using LLM."""
    def generate_documentation(state: DocState) -> DocState:
        print("\n=== DOCUMENTATION GENERATION ===")
        print(f"Current state: {state['next']}")
        
        try:
            # Get the message content
            current_message = state["current_message"]
            if isinstance(current_message, (HumanMessage, AIMessage)):
                topic = current_message.content
            else:
                topic = str(current_message)  # Fallback for other types
            
            print(f"Input message: {topic[:100]}...")
            
            # Get the primary server for documentation generation
            server = mcp_server_manager.get_server("primary")
            
            # Create system prompt
            system_prompt = (
                "You are a technical documentation expert. Generate clear, "
                "comprehensive documentation that includes:\n"
                "- Overview and purpose\n"
                "- Key concepts and components\n"
                "- Code examples\n"
                "- Best practices\n"
                "- Security considerations\n"
                "- Troubleshooting tips\n\n"
                "Follow these guidelines:\n"
                "1. Use proper markdown formatting\n"
                "2. Include practical examples\n"
                "3. Be concise but thorough\n"
                "4. Focus on real-world use cases\n"
                "5. Ensure technical accuracy\n"
                "6. Make it easy to understand"
            )
            
            # Create the chain and generate documentation
            chain = server | StrOutputParser()
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Generate documentation for: {topic}")
            ]
            documentation = chain.invoke(messages)
            
            print(
                f"Generated documentation length: "
                f"{len(documentation)} chars"
            )
            print("\nGenerated Documentation Preview:")
            print("=" * 80)
            print(documentation[:500] + "...")
            print("=" * 80)
            
            # Create new message
            new_message = AIMessage(content=documentation)
            
            # Get existing history or initialize empty list
            history = state.get("conversation_history", [])
            if not history and state["current_message"] not in history:
                history = [state["current_message"]]
            
            # Update state with new message and history
            return {
                **state,
                "current_message": new_message,
                "doc_content": documentation,
                "next": "validate",
                "conversation_history": history + [new_message]
            }
            
        except Exception as e:
            return handle_error({**state, "error": str(e)})
    
    return generate_documentation


def create_validator_node():
    """Create a node that validates the documentation."""
    def validate_documentation(state: DocState) -> DocState:
        print("\n=== DOCUMENTATION VALIDATION ===")
        print(f"Current state: {state['next']}")
        print(f"Documentation length: {len(state['doc_content'])} chars")
        
        try:
            # Get the validator server
            server = mcp_server_manager.get_server("validator")
            
            # Create validation prompt
            validation_prompt = (
                "Review the following documentation for:\n"
                "1. Technical accuracy:\n"
                "   - Are the technical details correct?\n"
                "   - Are the code examples valid?\n"
                "   - Are the best practices current?\n\n"
                "2. Completeness:\n"
                "   - Does it cover all key aspects?\n"
                "   - Are there any major gaps?\n"
                "   - Are all sections properly developed?\n\n"
                "3. Clarity:\n"
                "   - Is the language clear and concise?\n"
                "   - Is the structure logical?\n"
                "   - Are examples well-explained?\n\n"
                "4. Formatting:\n"
                "   - Is markdown formatting correct?\n"
                "   - Are headings properly used?\n"
                "   - Is the layout consistent?\n\n"
                "Documentation:\n"
                "{doc_content}\n\n"
                "Provide specific, actionable feedback. "
                "Focus on critical issues only."
            )
            
            # Create the chain and validate
            chain = server | StrOutputParser()
            messages = [
                SystemMessage(content=validation_prompt.format(
                    doc_content=state["doc_content"]
                ))
            ]
            validation_feedback = chain.invoke(messages)
            
            print(
                f"Validation feedback length: "
                f"{len(validation_feedback)} chars"
            )
            print("\nValidation Feedback:")
            print("=" * 80)
            print(validation_feedback)
            print("=" * 80)
            
            # Return new state with validation feedback
            return {
                **state,
                "validation_feedback": validation_feedback,
                "next": "review"
            }
            
        except Exception as e:
            return handle_error({**state, "error": str(e)})
    
    return validate_documentation


def create_reviewer_node():
    """Create a node that reviews the documentation."""
    def review_documentation(state: DocState) -> DocState:
        print("\n=== DOCUMENTATION REVIEW ===")
        print(f"Current state: {state['next']}")
        print(f"Documentation length: {len(state['doc_content'])} chars")
        print(
            f"Validation feedback length: "
            f"{len(state['validation_feedback'])} chars"
        )
        
        try:
            # Get the reviewer server
            server = mcp_server_manager.get_server("reviewer")
            
            # Create review prompt
            review_prompt = (
                "Review the following documentation "
                "and validation feedback:\n\n"
                "Documentation:\n"
                "{doc_content}\n\n"
                "Validation Feedback:\n"
                "{validation_feedback}\n\n"
                "Provide a final review considering:\n"
                "1. Is the documentation production-ready?\n"
                "2. Are the validation points valid?\n"
                "3. Are there any critical issues?\n"
                "4. Would you approve this for publication?\n\n"
                "Be specific about what needs improvement, if anything."
            )
            
            # Create the chain and review
            chain = server | StrOutputParser()
            messages = [
                SystemMessage(content=review_prompt.format(
                    doc_content=state["doc_content"],
                    validation_feedback=state["validation_feedback"]
                ))
            ]
            review_feedback = chain.invoke(messages)
            
            print(f"Review feedback length: {len(review_feedback)} chars")
            print("\nReview Feedback:")
            print("=" * 80)
            print(review_feedback)
            print("=" * 80)
            
            # Return new state with review feedback
            return {
                **state,
                "review_feedback": review_feedback,
                "next": "review_decision"
            }
            
        except Exception as e:
            return handle_error({**state, "error": str(e)})
    
    return review_documentation


def create_review_decision_node():
    """Create a node that makes a decision based on review feedback."""
    def review_decision(state: DocState) -> DocState:
        print("\n=== REVIEW DECISION ===")
        print("Current state: review_decision")
        print(f"Review feedback length: {len(state['review_feedback'])} chars")
        
        try:
            # Get revision count
            revision_count = state.get("revision_count", 0)
            
            # Analyze both validation and review feedback
            validation_issues = analyze_feedback(state["validation_feedback"])
            review_issues = analyze_feedback(state["review_feedback"])
            
            # Combine issues
            all_issues = validation_issues + review_issues
            
            # Check for critical issues
            has_critical_issues = any(
                issue["severity"] == "critical" for issue in all_issues
            )
            
            # If we've hit the revision limit, force approval
            if revision_count >= 3:
                print("Decision: Forcing approval (revision limit reached)")
                print("Next step: save")
                return {
                    **state,
                    "next": "save",
                    "feedback": "Revision limit reached, forcing approval"
                }
            
            # If there are critical issues and we haven't hit the limit, revise
            if has_critical_issues:
                print(
                    "Decision: Documentation needs revision "
                    "(critical issues)"
                )
                print("Next step: generate")
                print(f"Revision count: {revision_count + 1}")
                return {
                    **state,
                    "next": "generate",
                    "revision_count": revision_count + 1,
                    "feedback": "Critical issues found, needs revision"
                }
            
            # No critical issues or revision limit reached, approve
            print("Decision: Documentation approved")
            print("Next step: save")
            return {
                **state,
                "next": "save",
                "feedback": "Documentation approved"
            }
            
        except Exception as e:
            return handle_error({**state, "error": str(e)})
    
    return review_decision


def create_save_documentation_node():
    """Create a node that saves the documentation."""
    def save_doc(state: DocState) -> DocState:
        print("\n=== SAVING DOCUMENTATION ===")
        print(f"Current state: {state['next']}")
        print(f"Documentation length: {len(state['doc_content'])} chars")
        
        try:
            # Show preview of what will be saved
            print("\nDocumentation Preview:")
            print("=" * 80)
            print(state["doc_content"][:500] + "...")
            print("=" * 80)
            
            # Create a temporary file with .md extension in /tmp
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.md',
                delete=False,
                dir='/tmp'
            ) as temp_file:
                temp_file.write(state["doc_content"])
                saved_file = temp_file.name
            
            print(f"Saved to file: {saved_file}")
            
            return {
                **state,
                "saved_file": saved_file,
                "next": "email"
            }
            
        except Exception as e:
            return handle_error({**state, "error": str(e)})
    
    return save_doc


def create_email_node(recipient_email: str):
    """Create a node that sends the documentation via email."""
    def send_email(state: DocState) -> DocState:
        print("\n=== SENDING EMAIL ===")
        print(f"Current state: {state['next']}")
        print(f"Recipient: {recipient_email}")
        print(f"File: {state['saved_file']}")
        
        try:
            # Simulate sending email
            print(f"Sending documentation to {recipient_email}")
            print(f"File: {state['saved_file']}")
            
            return {
                **state,
                "email_sent": True,
                "next": END
            }
            
        except Exception as e:
            return handle_error({**state, "error": str(e)})
    
    return send_email


def create_human_task_node(name: str, prompt: str):
    """Create a human task node for approval."""
    def human_task(state: DocState) -> DocState:
        print(f"\n=== {name.upper()} ===")
        print(f"Current state: {state['next']}")
        
        try:
            # Get human approval
            response = input(f"\n{prompt} (y/n)? ").lower()
            approved = response == "y"
            
            if not approved:
                return {
                    **state,
                    "next": "generate",
                    "feedback": f"Human rejected {name}"
                }
            
            return {
                **state,
                "next": state["next"].replace("_approval", "")
            }
            
        except Exception as e:
            return handle_error({**state, "error": str(e)})
    
    return human_task


def test_doc_email_flow():
    """Test the documentation generation and email workflow."""
    # Check for required API keys
    missing_keys = []
    if not os.getenv("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")
    if not os.getenv("ANTHROPIC_API_KEY"):
        missing_keys.append("ANTHROPIC_API_KEY")
    
    if missing_keys:
        pytest.skip(
            f"Missing required API keys: {', '.join(missing_keys)}. "
            "Please set the environment variables and try again."
        )
    
    # Configuration
    recipient_email = "user@example.com"
    
    # Create the workflow
    workflow = StateGraph(DocState)
    
    # Add nodes
    workflow.add_node("generate", create_doc_generator_node())
    workflow.add_node("validate", create_validator_node())
    workflow.add_node("review", create_reviewer_node())
    workflow.add_node("review_decision", create_review_decision_node())
    workflow.add_node("save", create_save_documentation_node())
    workflow.add_node("email", create_email_node(recipient_email))
    
    # Add edges with conditional routing
    workflow.add_edge("generate", "validate")
    workflow.add_edge("validate", "review")
    workflow.add_edge("review", "review_decision")
    workflow.add_conditional_edges(
        "review_decision",
        lambda x: "generate" if x["next"] == "generate" else "save",
        {
            "generate": "generate",
            "save": "save"
        }
    )
    workflow.add_edge("save", "email")
    workflow.add_edge("email", END)
    
    # Set the entry point
    workflow.set_entry_point("generate")
    
    # Compile the workflow
    app = workflow.compile()
    
    try:
        # Create initial message
        initial_message = HumanMessage(
            content="Generate documentation for AWS Lambda functions"
        )
        
        # Run the workflow with increased recursion limit
        result = app.invoke(
            {
                "current_message": initial_message,
                "next": "generate",
                "doc_content": "",
                "validation_feedback": "",
                "review_feedback": "",
                "email_sent": False,
                "saved_file": None,
                "conversation_history": [],
                "revision_count": 0,
                "errors": [],
                "feedback": ""
            },
            {"recursion_limit": 10}  # Lower recursion limit
        )
        
        # Verify the workflow completed successfully
        assert result["email_sent"] is True
        assert result["saved_file"] is not None
        assert len(result["conversation_history"]) > 1
        assert isinstance(result["current_message"], AIMessage)
        assert len(result["errors"]) == 0
        
        # Print the location of the saved file
        print(f"\nDocumentation saved to: {result['saved_file']}")
        print(
            "The documentation file has been saved and "
            "will be kept for your reference."
        )
        
    except Exception as e:
        # Clean up in case of failure
        if "result" in locals() and result.get("saved_file"):
            if os.path.exists(result["saved_file"]):
                os.remove(result["saved_file"])
        raise e


def main():
    """Run the documentation workflow as a standalone script."""
    # Configuration
    recipient_email = "user@example.com"
    
    # Create the workflow
    workflow = StateGraph(DocState)
    
    # Create human task nodes
    doc_approval = create_human_task_node(
        "Documentation Approval",
        "Do you approve this documentation"
    )
    
    validation_approval = create_human_task_node(
        "Validation Approval",
        "Do you approve the validation feedback"
    )
    
    review_approval = create_human_task_node(
        "Review Approval",
        "Do you approve the review feedback"
    )
    
    save_approval = create_human_task_node(
        "Save Approval",
        "Do you want to save this documentation"
    )
    
    email_approval = create_human_task_node(
        "Email Approval",
        "Do you want to send the email"
    )
    
    # Add nodes
    workflow.add_node("generate", create_doc_generator_node())
    workflow.add_node("doc_approval", doc_approval)
    workflow.add_node("validate", create_validator_node())
    workflow.add_node("validation_approval", validation_approval)
    workflow.add_node("review", create_reviewer_node())
    workflow.add_node("review_approval", review_approval)
    workflow.add_node("review_decision", create_review_decision_node())
    workflow.add_node("save", create_save_documentation_node())
    workflow.add_node("save_approval", save_approval)
    workflow.add_node("email", create_email_node(recipient_email))
    workflow.add_node("email_approval", email_approval)
    
    # Add edges with conditional routing
    workflow.add_edge("generate", "doc_approval")
    workflow.add_conditional_edges(
        "doc_approval",
        lambda x: "generate" if x["next"] == "generate" else "validate",
        {
            "generate": "generate",
            "validate": "validate"
        }
    )
    workflow.add_edge("validate", "validation_approval")
    workflow.add_conditional_edges(
        "validation_approval",
        lambda x: "generate" if x["next"] == "generate" else "review",
        {
            "generate": "generate",
            "review": "review"
        }
    )
    workflow.add_edge("review", "review_approval")
    workflow.add_conditional_edges(
        "review_approval",
        lambda x: "generate" if x["next"] == "generate" else "review_decision",
        {
            "generate": "generate",
            "review_decision": "review_decision"
        }
    )
    workflow.add_conditional_edges(
        "review_decision",
        lambda x: "generate" if x["next"] == "generate" else "save",
        {
            "generate": "generate",
            "save": "save"
        }
    )
    workflow.add_edge("save", "save_approval")
    workflow.add_conditional_edges(
        "save_approval",
        lambda x: "generate" if x["next"] == "generate" else "email",
        {
            "generate": "generate",
            "email": "email"
        }
    )
    workflow.add_edge("email", "email_approval")
    workflow.add_conditional_edges(
        "email_approval",
        lambda x: "save" if x["next"] == "generate" else END,
        {
            "save": "save",
            "end": END
        }
    )
    
    # Set the entry point
    workflow.set_entry_point("generate")
    
    # Compile the workflow
    app = workflow.compile()
    
    try:
        # Create initial message
        initial_message = HumanMessage(
            content="Generate documentation for AWS Lambda functions"
        )
        
        # Run the workflow with increased recursion limit
        result = app.invoke(
            {
                "current_message": initial_message,
                "next": "generate",
                "doc_content": "",
                "validation_feedback": "",
                "review_feedback": "",
                "email_sent": False,
                "saved_file": None,
                "conversation_history": [],
                "revision_count": 0,
                "errors": [],
                "feedback": ""
            },
            {"recursion_limit": 10}  # Lower recursion limit
        )
        
        # Print the location of the saved file
        print(f"\nDocumentation saved to: {result['saved_file']}")
        print(
            "The documentation file has been saved and "
            "will be kept for your reference."
        )
        
    except Exception as e:
        # Clean up in case of failure
        if "result" in locals() and result.get("saved_file"):
            if os.path.exists(result["saved_file"]):
                os.remove(result["saved_file"])
        raise e


if __name__ == "__main__":
    main() 