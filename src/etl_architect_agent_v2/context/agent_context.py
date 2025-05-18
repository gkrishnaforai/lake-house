"""Agent context and goals for the ETL Architect Agent V2."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class AgentGoal:
    """Represents a goal for the agent."""
    description: str
    priority: int
    deadline: Optional[datetime] = None
    status: str = "pending"
    dependencies: List[str] = field(default_factory=list)

@dataclass
class AgentContext:
    """Manages the context and goals for the agent."""
    goals: List[AgentGoal] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    current_focus: Optional[str] = None
    
    def add_goal(self, goal: AgentGoal) -> None:
        """Add a new goal to the context.
        
        Args:
            goal: Goal to add
        """
        self.goals.append(goal)
        self.goals.sort(key=lambda g: g.priority, reverse=True)
    
    def update_goal_status(self, goal_description: str, status: str) -> None:
        """Update the status of a goal.
        
        Args:
            goal_description: Description of the goal to update
            status: New status
        """
        for goal in self.goals:
            if goal.description == goal_description:
                goal.status = status
                break
    
    def get_current_goal(self) -> Optional[AgentGoal]:
        """Get the current highest priority goal.
        
        Returns:
            Current goal or None if no goals
        """
        if not self.goals:
            return None
        return self.goals[0] 