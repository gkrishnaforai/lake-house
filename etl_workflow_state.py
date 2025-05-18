from typing import Dict, Any, Optional
from core.llm.manager import LLMManager

class ETLWorkflowState:
    """State class for ETL workflow."""
    
    def __init__(self, llm: Optional[LLMManager] = None):
        """Initialize the state."""
        self.data: Dict[str, Any] = {}
        self.llm = llm or LLMManager() 