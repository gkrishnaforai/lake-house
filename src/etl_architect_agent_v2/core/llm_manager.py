"""LLM management for the ETL Architect Agent V2."""

from typing import Dict, Any, Optional
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import Runnable, RunnableConfig

logger = logging.getLogger(__name__)


class LLMManager(Runnable):
    """Manages interactions with the LLM and handles token usage."""
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        api_key: Optional[str] = None
    ):
        """Initialize the LLM manager.
        
        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature for response generation
            max_tokens: Maximum tokens for response
            api_key: OpenAI API key
        """
        self.api_key = api_key
        self._model = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=api_key
        )
        self.token_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    
    @property
    def llm(self):
        """Get the LLM model instance."""
        return self._model
    
    @property
    def model(self):
        """Get the LLM model instance (alias for llm)."""
        return self._model
    
    def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Synchronous version of invoke.
        
        Args:
            input: Dictionary containing messages and optional system prompt
            config: Optional runnable config
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing the response
        """
        import asyncio
        return asyncio.run(self.ainvoke(input, config, **kwargs))
    
    async def ainvoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Invoke the LLM with input messages.
        
        Args:
            input: Dictionary containing messages and optional system prompt
            config: Optional runnable config
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing the response
        """
        try:
            # Handle different input types
            if isinstance(input, dict):
                messages = input.get("messages", [])
                system_prompt = input.get("system_prompt")
                
                # Convert messages to LangChain format
                langchain_messages = []
                if system_prompt:
                    langchain_messages.append(
                        SystemMessage(content=str(system_prompt))
                    )
                
                for msg in messages:
                    if isinstance(msg, tuple):
                        role, content = msg
                        if role == "system":
                            langchain_messages.append(
                                SystemMessage(content=str(content))
                            )
                        elif role == "user":
                            langchain_messages.append(
                                HumanMessage(content=str(content))
                            )
                        elif role == "assistant":
                            langchain_messages.append(
                                AIMessage(content=str(content))
                            )
                    elif isinstance(msg, dict):
                        if msg["role"] == "system":
                            langchain_messages.append(
                                SystemMessage(content=str(msg["content"]))
                            )
                        elif msg["role"] == "user":
                            langchain_messages.append(
                                HumanMessage(content=str(msg["content"]))
                            )
                        elif msg["role"] == "assistant":
                            langchain_messages.append(
                                AIMessage(content=str(msg["content"]))
                            )
            else:
                # Assume input is already in LangChain message format
                langchain_messages = input
            
            # Get completion
            response = await self.model.agenerate([langchain_messages])
            
            # Track token usage
            if response.llm_output and "token_usage" in response.llm_output:
                usage = response.llm_output["token_usage"]
                self.token_usage["prompt_tokens"] += usage.get(
                    "prompt_tokens", 0
                )
                self.token_usage["completion_tokens"] += usage.get(
                    "completion_tokens", 0
                )
                self.token_usage["total_tokens"] += usage.get(
                    "total_tokens", 0
                )
            
            # Extract content from LLMResult
            if not response.generations or not response.generations[0]:
                raise ValueError("No generations in LLM response")
            
            generation = response.generations[0][0]
            content = generation.text
            
            return {
                "content": content,
                "token_usage": self.token_usage.copy()
            }
            
        except Exception as e:
            logger.error(f"Error getting LLM completion: {e}")
            raise
    
    def get_token_usage(self) -> Dict[str, int]:
        """Get current token usage statistics.
        
        Returns:
            Dictionary with token usage counts
        """
        return self.token_usage.copy()
    
    def reset_token_usage(self) -> None:
        """Reset token usage counters."""
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated number of tokens
        """
        # Rough estimation: 4 characters per token
        return len(text) // 4 