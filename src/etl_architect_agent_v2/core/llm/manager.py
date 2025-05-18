"""LLM manager for language model operations."""

import os
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool

class LLMManager:
    """Manager for LLM operations and agent creation."""
    
    def __init__(self):
        """Initialize the LLM manager."""
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name=os.getenv('OPENAI_MODEL_NAME', 'gpt-3.5-turbo')
        )
        self.agents: Dict[str, AgentExecutor] = {}
    
    def create_agent(
        self,
        agent_id: str,
        tools: List[Tool],
        system_prompt: str
    ) -> AgentExecutor:
        """Create a new agent with specified tools and prompt.
        
        Args:
            agent_id: Unique identifier for the agent
            tools: List of tools for the agent
            system_prompt: System prompt for the agent
            
        Returns:
            Created agent executor
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(
            self.llm,
            tools,
            prompt
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True
        )
        
        self.agents[agent_id] = agent_executor
        return agent_executor
    
    def get_agent(self, agent_id: str) -> Optional[AgentExecutor]:
        """Get an existing agent by ID.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent executor if found, None otherwise
        """
        return self.agents.get(agent_id)
    
    async def run_agent(
        self,
        agent_id: str,
        input_text: str,
        chat_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Run an agent with input text and chat history.
        
        Args:
            agent_id: Agent identifier
            input_text: Input text for the agent
            chat_history: Optional chat history
            
        Returns:
            Agent response
        """
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        return await agent.ainvoke({
            "input": input_text,
            "chat_history": chat_history or []
        })
    
    def clear_agents(self):
        """Clear all agents."""
        self.agents.clear() 