import argparse
import sys

from aws_architect_agent.core.agent import AWSArchitectAgent
from aws_architect_agent.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="AWS Solution Architect Agent"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="OpenAI model to use",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Model temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2000,
        help="Maximum tokens for model generation",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Set up logging
    setup_logging(log_level="DEBUG" if args.debug else "INFO")
    
    # Initialize agent
    agent = AWSArchitectAgent(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    
    print("AWS Solution Architect Agent")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        try:
            message = input("You: ")
            if message.lower() == "quit":
                break
                
            response = agent.process_message(message)
            print(f"Agent: {response}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {str(e)}")
            if args.debug:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main() 