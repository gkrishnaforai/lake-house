from typing import Dict, List, Optional

import requests
from langchain.tools import BaseTool

from aws_architect_agent.models.base import (
    Architecture,
    ArchitectureState,
    ConversationState,
    SolutionType,
)
from aws_architect_agent.services.cdk_generator import CDKGenerator
from aws_architect_agent.services.documentation import DocumentationGenerator
from aws_architect_agent.services.terraform_generator import TerraformGenerator
from aws_architect_agent.utils.logging import get_logger

logger = get_logger(__name__)


class RequirementsGatheringTool(BaseTool):
    """Tool for gathering architecture requirements."""

    name = "gather_requirements"
    description = "Gather requirements for the architecture design"

    def _run(self, message: str) -> str:
        """Run the tool.

        Args:
            message: User message

        Returns:
            Tool response
        """
        return "requirements_complete"

    async def _arun(self, message: str) -> str:
        """Run the tool asynchronously.

        Args:
            message: User message

        Returns:
            Tool response
        """
        return self._run(message)


class ArchitectureDesignTool(BaseTool):
    """Tool for designing architecture."""

    name = "design_architecture"
    description = "Design the architecture based on requirements"

    def _run(self, state: ConversationState) -> str:
        """Run the tool.

        Args:
            state: Current conversation state

        Returns:
            Tool response
        """
        if not state.current_architecture:
            state.current_architecture = Architecture(
                id="arch_1",
                name="Initial Design",
                description="First iteration of the architecture",
                solution_type=SolutionType.ETL,
            )
        return "Architecture designed"

    async def _arun(self, state: ConversationState) -> str:
        """Run the tool asynchronously.

        Args:
            state: Current conversation state

        Returns:
            Tool response
        """
        return self._run(state)


class ArchitectureReviewTool(BaseTool):
    """Tool for reviewing architecture."""

    name = "review_architecture"
    description = "Review the architecture against best practices"

    def _run(self, state: ConversationState) -> str:
        """Run the tool.

        Args:
            state: Current conversation state

        Returns:
            Tool response
        """
        return "Review complete"

    async def _arun(self, state: ConversationState) -> str:
        """Run the tool asynchronously.

        Args:
            state: Current conversation state

        Returns:
            Tool response
        """
        return self._run(state)


class ArchitectureRefinementTool(BaseTool):
    """Tool for refining architecture."""

    name = "refine_architecture"
    description = "Refine the architecture based on feedback"

    def _run(self, state: ConversationState) -> str:
        """Run the tool.

        Args:
            state: Current conversation state

        Returns:
            Tool response
        """
        return "refinement_complete"

    async def _arun(self, state: ConversationState) -> str:
        """Run the tool asynchronously.

        Args:
            state: Current conversation state

        Returns:
            Tool response
        """
        return self._run(state)


class DocumentationTool(BaseTool):
    """Tool for generating documentation."""

    name = "generate_documentation"
    description = "Generate documentation for the architecture"

    def _run(
        self,
        state: ConversationState,
        format: str = "markdown",
    ) -> str:
        """Run the tool.

        Args:
            state: Current conversation state
            format: Documentation format

        Returns:
            Generated documentation
        """
        generator = DocumentationGenerator(state.current_architecture)
        if format == "markdown":
            return generator.generate_markdown()
        elif format == "html":
            return generator.generate_html()
        elif format == "confluence":
            return generator.generate_confluence()
        elif format == "notion":
            return generator.generate_notion()
        else:
            return "Unsupported format"

    async def _arun(
        self,
        state: ConversationState,
        format: str = "markdown",
    ) -> str:
        """Run the tool asynchronously.

        Args:
            state: Current conversation state
            format: Documentation format

        Returns:
            Generated documentation
        """
        return self._run(state, format)


class InfrastructureCodeTool(BaseTool):
    """Tool for generating infrastructure code."""

    name = "generate_infrastructure_code"
    description = "Generate infrastructure code for the architecture"

    def _run(
        self,
        state: ConversationState,
        format: str = "cdk",
    ) -> Dict:
        """Run the tool.

        Args:
            state: Current conversation state
            format: Code format (cdk or terraform)

        Returns:
            Generated infrastructure code
        """
        if format == "cdk":
            generator = CDKGenerator(state.current_architecture)
            return generator.generate_template()
        elif format == "terraform":
            generator = TerraformGenerator(state.current_architecture)
            return generator.generate()
        else:
            return {"error": "Unsupported format"}

    async def _arun(
        self,
        state: ConversationState,
        format: str = "cdk",
    ) -> Dict:
        """Run the tool asynchronously.

        Args:
            state: Current conversation state
            format: Code format (cdk or terraform)

        Returns:
            Generated infrastructure code
        """
        return self._run(state, format)


class StateManagementTool(BaseTool):
    """Tool for managing conversation state."""

    name = "manage_state"
    description = "Manage the conversation state"

    def _run(
        self,
        state: ConversationState,
        action: str,
        data: Optional[Dict] = None,
    ) -> ConversationState:
        """Run the tool.

        Args:
            state: Current conversation state
            action: Action to perform
            data: Additional data for the action

        Returns:
            Updated conversation state
        """
        if action == "update_state":
            state.current_state = ArchitectureState(data["state"])
        elif action == "add_feedback":
            state.feedback.append(data["feedback"])
        elif action == "update_requirements":
            state.requirements.update(data["requirements"])
        return state

    async def _arun(
        self,
        state: ConversationState,
        action: str,
        data: Optional[Dict] = None,
    ) -> ConversationState:
        """Run the tool asynchronously.

        Args:
            state: Current conversation state
            action: Action to perform
            data: Additional data for the action

        Returns:
            Updated conversation state
        """
        return self._run(state, action, data)


class AWSDocumentationSearchTool(BaseTool):
    """Tool for searching AWS documentation and best practices."""

    name = "search_aws_docs"
    description = (
        "Search AWS documentation, best practices, and well-architected framework"
    )

    def __init__(self) -> None:
        """Initialize the AWS documentation search tool."""
        self.base_urls = {
            "docs": "https://docs.aws.amazon.com",
            "well_architected": (
                "https://docs.aws.amazon.com/wellarchitected/latest/framework"
            ),
            "best_practices": (
                "https://aws.amazon.com/architecture/well-architected"
            ),
        }
        self.headers = {
            "User-Agent": "AWSArchitectAgent/1.0",
            "Accept": "application/json",
        }

    def _run(
        self,
        query: str,
        source: str = "all",
        max_results: int = 5,
    ) -> Dict[str, List[Dict]]:
        """Run the tool.

        Args:
            query: Search query
            source: Source to search (docs, well_architected, best_practices, or all)
            max_results: Maximum number of results to return

        Returns:
            Search results
        """
        results = {
            "docs": [],
            "well_architected": [],
            "best_practices": [],
        }

        try:
            if source in ["docs", "all"]:
                results["docs"] = self._search_docs(query, max_results)
            if source in ["well_architected", "all"]:
                results["well_architected"] = self._search_well_architected(
                    query, max_results
                )
            if source in ["best_practices", "all"]:
                results["best_practices"] = self._search_best_practices(
                    query, max_results
                )
        except Exception as e:
            logger.error(f"Error searching AWS documentation: {str(e)}")
            return {"error": str(e)}

        return results

    async def _arun(
        self,
        query: str,
        source: str = "all",
        max_results: int = 5,
    ) -> Dict[str, List[Dict]]:
        """Run the tool asynchronously.

        Args:
            query: Search query
            source: Source to search (docs, well_architected, best_practices, or all)
            max_results: Maximum number of results to return

        Returns:
            Search results
        """
        return self._run(query, source, max_results)

    def _search_docs(self, query: str, max_results: int) -> List[Dict]:
        """Search AWS documentation.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of search results
        """
        # This is a placeholder for actual AWS documentation search
        # In a real implementation, you would use AWS's search API or web scraping
        return [
            {
                "title": "AWS Documentation",
                "url": f"{self.base_urls['docs']}/search.html?q={query}",
                "snippet": (
                    f"Search results for {query} in AWS documentation"
                ),
            }
        ]

    def _search_well_architected(self, query: str, max_results: int) -> List[Dict]:
        """Search Well-Architected Framework documentation.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of search results
        """
        # This is a placeholder for actual Well-Architected Framework search
        return [
            {
                "title": "Well-Architected Framework",
                "url": f"{self.base_urls['well_architected']}/search.html?q={query}",
                "snippet": (
                    f"Search results for {query} in Well-Architected Framework"
                ),
            }
        ]

    def _search_best_practices(self, query: str, max_results: int) -> List[Dict]:
        """Search AWS best practices.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of search results
        """
        # This is a placeholder for actual best practices search
        return [
            {
                "title": "AWS Best Practices",
                "url": f"{self.base_urls['best_practices']}/search.html?q={query}",
                "snippet": (
                    f"Search results for {query} in AWS best practices"
                ),
            }
        ] 