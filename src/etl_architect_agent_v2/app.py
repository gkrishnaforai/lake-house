"""Main application for the ETL Architect Agent V2."""

import asyncio
import argparse
import os
from typing import Optional
from core.agent_orchestrator import AgentOrchestrator
from utils.logger import setup_logger
from pathlib import Path

logger = setup_logger(__name__)

async def run_agent(message: str, output_file: Optional[Path] = None) -> None:
    """Run the agent with the given message.
    
    Args:
        message: Input message for the agent
        output_file: Optional file to save the output
    """
    try:
        # Initialize the agent
        agent = AgentOrchestrator()
        
        # Run the agent
        state = await agent.run(message)
        
        # Log the results
        logger.info("ETL pipeline design completed successfully")
        logger.info(f"Final state: {state}")
        
        # Save output if specified
        if output_file:
            with open(output_file, "w") as f:
                f.write(str(state))
            logger.info(f"Output saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in main application: {e}")
        raise

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="ETL Architect Agent V2 - Design and implement ETL pipelines"
    )
    parser.add_argument(
        "message",
        help="Input message describing the ETL pipeline requirements"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="File to save the output to"
    )
    
    args = parser.parse_args()
    
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is required")
        return
    
    # Run the agent
    asyncio.run(run_agent(args.message, args.output))

if __name__ == "__main__":
    main() 