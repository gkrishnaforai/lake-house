from typing import Dict, Any, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from aws_architect_agent.utils.mcp_utils import mcp_server_manager
from aws_architect_agent.config.mcp_config import ModelType


# Define the state
class AgentState(TypedDict):
    """State for the agent workflow."""
    messages: list[HumanMessage | AIMessage]
    next: str


# Define tools
@tool
def search_database(query: str) -> str:
    """Search the database for information."""
    # This is a mock implementation
    return f"Found results for: {query}"


@tool
def analyze_data(data: str) -> str:
    """Analyze the given data."""
    # This is a mock implementation
    return f"Analysis of: {data}"


# Define the nodes
def create_agent_node(server_name: str):
    """Create an agent node with the specified MCP server."""
    def agent(state: AgentState) -> Dict[str, Any]:
        # Get the MCP server
        server = mcp_server_manager.get_server(server_name)
        
        # Create the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant."),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Create the chain
        chain = prompt | server | StrOutputParser()
        
        # Get the response
        response = chain.invoke({"messages": state["messages"]})
        
        # Update the state
        return {
            "messages": state["messages"] + [AIMessage(content=response)],
            "next": "end" if "end" in response.lower() else "tools"
        }
    
    return agent


def create_tools_node():
    """Create a node that uses tools."""
    def tools(state: AgentState) -> Dict[str, Any]:
        # Get the last message
        last_message = state["messages"][-1].content
        
        # Use tools based on the message
        if "search" in last_message.lower():
            result = search_database(last_message)
        elif "analyze" in last_message.lower():
            result = analyze_data(last_message)
        else:
            result = "No tool matched the request"
        
        # Update the state
        return {
            "messages": state["messages"] + [HumanMessage(content=result)],
            "next": "agent"
        }
    
    return tools


def test_mcp_langgraph_flow():
    """Test the MCP LangGraph workflow."""
    # Create the workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", create_agent_node("primary"))
    workflow.add_node("tools", create_tools_node())
    
    # Add edges
    workflow.add_edge("agent", "tools")
    workflow.add_edge("tools", "agent")
    workflow.add_edge("agent", END)
    
    # Set the entry point
    workflow.set_entry_point("agent")
    
    # Compile the workflow
    app = workflow.compile()
    
    # Run the workflow
    result = app.invoke({
        "messages": [
            HumanMessage(content="Search for information about AWS ETL pipelines")
        ],
        "next": "agent"
    })
    
    # Verify the result
    assert len(result["messages"]) > 1, "Expected multiple messages in the conversation"
    assert any("search" in msg.content.lower() for msg in result["messages"]), (
        "Expected a search operation"
    )
    assert any("analyze" in msg.content.lower() for msg in result["messages"]), (
        "Expected an analysis operation"
    )
    assert result["next"] == "end", "Expected workflow to end" 