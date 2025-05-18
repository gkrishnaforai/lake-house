import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import sys
from time import sleep

from src.core.workflow.etl_state import ETLWorkflowState, ETLStep
from src.core.workflow.etl_utils import ETLUtils
from src.core.llm.manager import LLMManager


def print_loading(message: str):
    """Print loading message with animation."""
    sys.stdout.write(f"\r{message} ")
    sys.stdout.flush()
    for i in range(3):
        sys.stdout.write(".")
        sys.stdout.flush()
        sleep(0.5)
    print()


class ConversationManager:
    def __init__(self, save_dir: str = "conversations"):
        print_loading("Initializing ETL Architecture Agent")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        print_loading("Loading LLM Manager")
        self.llm_manager = LLMManager()
        print_loading("Initializing ETL Utilities")
        self.etl_utils = ETLUtils(self.llm_manager)
        self.state = ETLWorkflowState()
        self.conversation_id = None
        self.conversation_history = []
        print("\nInitialization complete!")

    def save_conversation(self):
        """Save current conversation state to file."""
        if not self.conversation_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.conversation_id = f"conv_{timestamp}"

        save_path = self.save_dir / f"{self.conversation_id}.json"
        data = {
            "conversation_id": self.conversation_id,
            "state": {
                "metadata": self.state.metadata,
                "current_step": self.state.current_step.name,
            },
            "history": self.conversation_history,
        }
        
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"\nConversation saved to: {save_path}")

    def load_conversation(self, conversation_id: str) -> bool:
        """Load conversation state from file."""
        save_path = self.save_dir / f"{conversation_id}.json"
        if not save_path.exists():
            return False

        with open(save_path) as f:
            data = json.load(f)
        
        self.conversation_id = data["conversation_id"]
        self.state.metadata = data["state"]["metadata"]
        self.state.current_step = ETLStep[data["state"]["current_step"]]
        self.conversation_history = data["history"]
        
        return True

    def list_conversations(self) -> list[str]:
        """List all saved conversations."""
        return [f.stem for f in self.save_dir.glob("conv_*.json")]

    async def process_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input and return response."""
        print_loading("Analyzing requirements")
        # Update state with user input
        self.state.metadata = {"project_description": user_input}
        self.state.current_step = ETLStep.ANALYSIS
        
        # Analyze requirements
        requirements = await self.etl_utils.analyze_project(self.state)
        
        print_loading("Generating follow-up questions")
        # Generate follow-up questions
        questions = await self.etl_utils.generate_questions(self.state)
        
        # Save to history
        self.conversation_history.append({
            "input": user_input,
            "requirements": requirements.__dict__,
            "questions": questions,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "requirements": requirements,
            "questions": questions
        }


def get_user_input() -> str:
    """Get multi-line input from user, handling carriage returns."""
    print("\nEnter your input (type 'submit' when done, 'exit' to cancel):")
    lines = []
    
    while True:
        try:
            # Read raw input
            raw_input = input("> ")
            
            # Handle carriage returns and newlines
            line = raw_input.replace('\r', '').strip()
            
            if line.lower() == 'submit':
                break
            if line.lower() == 'exit':
                print("\nCancelled.")
                return None
            if line:
                lines.append(line)
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return None
    
    return "\n".join(lines) if lines else None


async def main():
    """Interactive ETL Architecture Agent conversation."""
    print("\nWelcome to the ETL Architecture Agent!")
    print("=======================================")
    print("Let me help you design your ETL architecture.")
    
    manager = ConversationManager()
    
    # Check for existing conversations
    conversations = manager.list_conversations()
    if conversations:
        print("\nFound existing conversations:")
        for i, conv_id in enumerate(conversations, 1):
            print(f"{i}. {conv_id}")
        print("0. Start new conversation")
        
        choice = input("\nSelect conversation (number) or start new (0): ")
        if choice != "0":
            try:
                conv_id = conversations[int(choice) - 1]
                if manager.load_conversation(conv_id):
                    print(f"\nLoaded conversation: {conv_id}")
                else:
                    print("\nFailed to load conversation. Starting new one.")
            except (ValueError, IndexError):
                print("\nInvalid choice. Starting new conversation.")
    
    while True:
        print("\nOptions:")
        print("1. Enter new project description")
        print("2. Ask a specific question")
        print("3. Save and exit")
        print("4. Exit without saving")
        
        choice = input("\nSelect option (1-4): ")
        
        if choice == "3":
            manager.save_conversation()
            print("\nGoodbye!")
            return
        elif choice == "4":
            print("\nGoodbye!")
            return
        elif choice in ["1", "2"]:
            user_input = get_user_input()
            if user_input is None:
                continue
            
            try:
                response = await manager.process_input(user_input)
                
                print("\nAnalysis Results:")
                print("----------------")
                reqs = response['requirements']
                print(f"Data Sources: {reqs.data_sources}")
                print(f"Visualization Needs: {reqs.visualization_needs}")
                print(f"Update Frequency: {reqs.update_frequency}")
                print(f"Access Requirements: {reqs.access_requirements}")
                
                if reqs.missing_info:
                    print("\nMissing Information:")
                    for info in reqs.missing_info:
                        print(f"- {info}")
                
                if reqs.issues:
                    print("\nPotential Concerns:")
                    for issue in reqs.issues:
                        print(f"- {issue}")
                
                if response['questions']:
                    print("\nFollow-up Questions:")
                    for i, question in enumerate(response['questions'], 1):
                        print(f"{i}. {question}")
                
                # Auto-save after each interaction
                manager.save_conversation()
                
            except Exception as e:
                print(f"\nAn error occurred: {str(e)}")
                print("Please try rephrasing your input.")


if __name__ == "__main__":
    asyncio.run(main()) 