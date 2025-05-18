"""Prompt management for the ETL Architect Agent V2."""

from typing import Dict, Any, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class PromptManager:
    """Manages prompts and templates for LLM interactions."""
    
    def __init__(self, prompt_dir: Optional[Path] = None):
        """Initialize the prompt manager.
        
        Args:
            prompt_dir: Directory containing prompt templates. If None, uses default prompts.
        """
        self.prompt_dir = prompt_dir or Path(__file__).parent / "prompts"
        self.templates: Dict[str, str] = {}
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Load prompt templates from the prompt directory."""
        try:
            for prompt_file in self.prompt_dir.glob("*.json"):
                with open(prompt_file, "r") as f:
                    template_data = json.load(f)
                    self.templates.update(template_data)
        except Exception as e:
            logger.error(f"Error loading prompt templates: {e}")
            # Load default templates
            self._load_default_templates()
    
    def _load_default_templates(self) -> None:
        """Load default prompt templates."""
        self.templates = {
            "requirements_analysis": (
                "Analyze the following requirements for an ETL pipeline:\n"
                "{requirements}\n\n"
                "Consider the following aspects:\n"
                "- Data sources and formats\n"
                "- Data volume and growth\n"
                "- Processing needs\n"
                "- Latency requirements\n"
                "- Security requirements\n"
                "- Analytics needs"
            ),
            "architecture_design": (
                "Design an ETL architecture based on these requirements:\n"
                "{requirements}\n\n"
                "Include:\n"
                "- Data ingestion components\n"
                "- Processing components\n"
                "- Storage components\n"
                "- Security measures\n"
                "- Scalability considerations"
            ),
            "validation_check": (
                "Validate this ETL architecture:\n"
                "{architecture}\n\n"
                "Check for:\n"
                "- Completeness\n"
                "- Scalability\n"
                "- Security\n"
                "- Performance\n"
                "- Cost efficiency"
            ),
            "implementation_plan": (
                "Create an implementation plan for this ETL architecture:\n"
                "{architecture}\n\n"
                "Include:\n"
                "- Required components\n"
                "- Dependencies\n"
                "- Implementation steps\n"
                "- Testing strategy"
            )
        }
    
    def get_prompt(self, template_name: str, **kwargs: Any) -> str:
        """Get a formatted prompt from a template.
        
        Args:
            template_name: Name of the template to use
            **kwargs: Variables to format into the template
            
        Returns:
            Formatted prompt string
            
        Raises:
            KeyError: If template_name is not found
        """
        if template_name not in self.templates:
            raise KeyError(f"Template '{template_name}' not found")
        
        try:
            return self.templates[template_name].format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing required variable in prompt template: {e}")
            raise
        except Exception as e:
            logger.error(f"Error formatting prompt: {e}")
            raise
    
    def add_template(self, name: str, template: str) -> None:
        """Add a new prompt template.
        
        Args:
            name: Name of the template
            template: Template string
        """
        self.templates[name] = template
    
    def save_templates(self) -> None:
        """Save current templates to the prompt directory."""
        try:
            self.prompt_dir.mkdir(parents=True, exist_ok=True)
            with open(self.prompt_dir / "templates.json", "w") as f:
                json.dump(self.templates, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving templates: {e}")
            raise 