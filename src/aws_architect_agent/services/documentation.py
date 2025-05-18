from jinja2 import Environment, FileSystemLoader
import markdown

from aws_architect_agent.models.base import Architecture
from aws_architect_agent.utils.logging import get_logger

logger = get_logger(__name__)


class DocumentationGenerator:
    """Service for generating architecture documentation."""

    def __init__(self, architecture: Architecture) -> None:
        """Initialize the documentation generator.

        Args:
            architecture: Architecture to document
        """
        self.architecture = architecture
        self.env = Environment(
            loader=FileSystemLoader("templates"),
            autoescape=True,
        )

    def generate_markdown(self) -> str:
        """Generate markdown documentation.

        Returns:
            Markdown documentation as a string
        """
        template = self.env.get_template("architecture.md.j2")
        return template.render(architecture=self.architecture)

    def generate_html(self) -> str:
        """Generate HTML documentation.

        Returns:
            HTML documentation as a string
        """
        markdown_content = self.generate_markdown()
        return markdown.markdown(markdown_content)

    def generate_confluence(self) -> str:
        """Generate Confluence-compatible documentation.

        Returns:
            Confluence documentation as a string
        """
        template = self.env.get_template("confluence.xml.j2")
        return template.render(architecture=self.architecture)

    def generate_notion(self) -> str:
        """Generate Notion-compatible documentation.

        Returns:
            Notion documentation as a string
        """
        template = self.env.get_template("notion.md.j2")
        return template.render(architecture=self.architecture)

    def generate_diagram(self, format: str = "mermaid") -> str:
        """Generate architecture diagram.

        Args:
            format: Diagram format (e.g., 'mermaid', 'plantuml')

        Returns:
            Diagram definition as a string
        """
        if format == "mermaid":
            return self._generate_mermaid_diagram()
        elif format == "plantuml":
            return self._generate_plantuml_diagram()
        else:
            raise ValueError(f"Unsupported diagram format: {format}")

    def _generate_mermaid_diagram(self) -> str:
        """Generate Mermaid diagram.

        Returns:
            Mermaid diagram definition
        """
        diagram = ["graph TD"]
        
        # Add nodes
        for component in self.architecture.components:
            diagram.append(
                f"    {component.id}[{component.name}]"
            )
        
        # Add edges
        for component in self.architecture.components:
            for dep in component.dependencies:
                diagram.append(f"    {dep} --> {component.id}")
        
        return "\n".join(diagram)

    def _generate_plantuml_diagram(self) -> str:
        """Generate PlantUML diagram.

        Returns:
            PlantUML diagram definition
        """
        diagram = ["@startuml"]
        
        # Add nodes
        for component in self.architecture.components:
            diagram.append(
                f"component \"{component.name}\" as {component.id}"
            )
        
        # Add edges
        for component in self.architecture.components:
            for dep in component.dependencies:
                diagram.append(f"{dep} --> {component.id}")
        
        diagram.append("@enduml")
        return "\n".join(diagram) 