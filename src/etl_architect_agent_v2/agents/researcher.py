from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_core.messages import AIMessage
from langchain.chat_models import ChatOpenAI
from langchain.tools import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun
from core.llm.manager import LLMManager
import asyncio
import os

class AgentState(TypedDict):
    """State for the researcher agent."""
    query: str
    context: str
    research_needed: bool
    result: str

llm = LLMManager(api_key=os.getenv("OPENAI_API_KEY"))

search_tool = DuckDuckGoSearchRun()


async def chatbot_node(state: AgentState) -> AgentState:
    prompt = f"User asked: {state['query']}\n Do we need to perform research to answer this question? Respond with 'yes' or 'no'."
    query = {"query": state["query"]}
    decision = await llm.invoke(template=prompt, input_data=query)
    state["research_needed"] = "yes" in decision.lower()
    return state

def search_node(state: AgentState) -> AgentState:
    state["result"] = search_tool.invoke(state["query"])
    return state


async def response_node(state: AgentState) -> AgentState:
    if state["result"]:
        prompt = f"Perform research on the following topic: {state['query']}\n\nSearch results:\n{state['result']}"
    else:
        prompt = f"Answer the following question: {state['query']}"

    query = {"query": state["query"]}

    answer = await llm.invoke(template=prompt, input_data=query)
    state["result"] = answer.lower()
    return state

def route_decision_node(state: AgentState) -> AgentState:
    return "research" if state["research_needed"] else "answer"
        

def node(state: AgentState) -> AgentState:
    """Node for the researcher agent."""
    new_message = AIMessage(content="Hello, world!")
    state["result"] = "Hello, world! " + new_message.content
    return state


workflow = StateGraph(AgentState)

workflow.add_node("chatbot", chatbot_node)
workflow.add_node("search", search_node)
workflow.add_node("response", response_node)
workflow.add_node("route", route_decision_node)

workflow.set_entry_point("chatbot")
workflow.add_edge("chatbot", "search")
workflow.add_conditional_edges("search", route_decision_node, "search", "response")
workflow.add_edge("search", "response")
workflow.add_edge("response", END)

graph = workflow.compile()


async def main():
    state = AgentState(
        query="What would be price of meta in q4 2025?",
        context="",
        result=""
    )
    final_state = await graph.ainvoke(state)
    print(final_state)

if __name__ == "__main__":
    asyncio.run(main())
