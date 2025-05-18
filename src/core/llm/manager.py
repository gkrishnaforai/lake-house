from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableConfig
import os


class LLMManager(Runnable):
    """Manages LLM interactions and configurations."""
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.7
    ):
        # Get API key from environment if not provided
        api_key = "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"
        #api_key = api_key or os.getenv("OPENAI_API_KEY")
        print(f"API Key: {api_key}")
        if not api_key:
            raise ValueError(
                "OpenAI API key must be provided either directly or "
                "through OPENAI_API_KEY environment variable"
            )
            
        self.llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature
        )
        self.output_parser = StrOutputParser()

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
                        HumanMessage(content=str(system_prompt))
                    )
                
                for msg in messages:
                    if isinstance(msg, tuple):
                        role, content = msg
                        if role == "user":
                            langchain_messages.append(
                                HumanMessage(content=str(content))
                            )
                        elif role == "system":
                            langchain_messages.append(
                                SystemMessage(content=str(content))
                            )
                        elif role == "assistant":
                            langchain_messages.append(
                                AIMessage(content=str(content))
                            )
                    elif isinstance(msg, dict):
                        if msg["role"] == "user":
                            langchain_messages.append(
                                HumanMessage(content=str(msg["content"]))
                            )
                        elif msg["role"] == "system":
                            langchain_messages.append(
                                SystemMessage(content=str(msg["content"]))
                            )
                        elif msg["role"] == "assistant":
                            langchain_messages.append(
                                AIMessage(content=str(msg["content"]))
                            )
            elif isinstance(input, (list, tuple)):
                # Handle direct list/tuple of messages
                langchain_messages = []
                for msg in input:
                    if isinstance(msg, tuple):
                        role, content = msg
                        if role == "user":
                            langchain_messages.append(
                                HumanMessage(content=str(content))
                            )
                        elif role == "system":
                            langchain_messages.append(
                                SystemMessage(content=str(content))
                            )
                        elif role == "assistant":
                            langchain_messages.append(
                                AIMessage(content=str(content))
                            )
                    elif isinstance(msg, dict):
                        if msg["role"] == "user":
                            langchain_messages.append(
                                HumanMessage(content=str(msg["content"]))
                            )
                        elif msg["role"] == "system":
                            langchain_messages.append(
                                SystemMessage(content=str(msg["content"]))
                            )
                        elif msg["role"] == "assistant":
                            langchain_messages.append(
                                AIMessage(content=str(msg["content"]))
                            )
            else:
                # Assume input is already in LangChain message format
                langchain_messages = input
            
            # Get completion
            response = await self.llm.agenerate([langchain_messages])
            
            # Extract content from LLMResult
            if not response.generations or not response.generations[0]:
                raise ValueError("No generations in LLM response")
            
            generation = response.generations[0][0]
            content = generation.text
            
            return {
                "content": content,
                "token_usage": response.llm_output.get("token_usage", {})
            }
            
        except Exception as e:
            raise ValueError(f"Error getting LLM completion: {str(e)}")

    def create_chain(
        self,
        template: str,
        output_parser: Optional[StrOutputParser] = None
    ):
        """Create a chain with the given template."""
        human_message = HumanMessage(content=template)
        prompt = ChatPromptTemplate.from_messages([human_message])
        parser = output_parser or self.output_parser
        return prompt | self.llm | parser

    async def invoke_chain(
        self,
        template: str,
        input_data: Dict[str, Any],
        output_parser: Optional[StrOutputParser] = None
    ) -> str:
        """Invoke the LLM with the given template and input."""
        chain = self.create_chain(template, output_parser)
        return await chain.ainvoke(input_data) 