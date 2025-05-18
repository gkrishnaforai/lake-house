from typing import Dict, List, Optional
from datetime import datetime
import uuid
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class ExecutionStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class ExecutionContext:
    user: str
    environment: str
    trigger: str
    metadata: Optional[Dict] = None


@dataclass
class Execution:
    execution_id: str
    start_time: str
    end_time: Optional[str] = None
    status: ExecutionStatus = ExecutionStatus.RUNNING
    parent_execution_id: Optional[str] = None
    context: Optional[ExecutionContext] = None
    states: List[Dict] = None

    def __post_init__(self):
        if self.states is None:
            self.states = []


class ExecutionManager:
    def __init__(self, storage_dir: str = "execution_states"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.executions: Dict[str, Execution] = {}

    def create_execution(
        self,
        user: str,
        environment: str = "development",
        trigger: str = "manual",
        parent_execution_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Execution:
        """Create a new execution."""
        execution_id = str(uuid.uuid4())
        context = ExecutionContext(
            user=user,
            environment=environment,
            trigger=trigger,
            metadata=metadata
        )
        
        execution = Execution(
            execution_id=execution_id,
            start_time=datetime.now().isoformat(),
            parent_execution_id=parent_execution_id,
            context=context
        )
        
        self.executions[execution_id] = execution
        self._save_execution(execution)
        return execution

    def add_state(self, execution_id: str, state: Dict) -> None:
        """Add a state to an execution."""
        if execution_id not in self.executions:
            raise ValueError(f"Execution {execution_id} not found")
        
        execution = self.executions[execution_id]
        execution.states.append({
            "timestamp": datetime.now().isoformat(),
            "state": state
        })
        self._save_execution(execution)

    def update_execution_status(
        self,
        execution_id: str,
        status: ExecutionStatus,
        end_time: Optional[str] = None
    ) -> None:
        """Update execution status."""
        if execution_id not in self.executions:
            raise ValueError(f"Execution {execution_id} not found")
        
        execution = self.executions[execution_id]
        execution.status = status
        if end_time:
            execution.end_time = end_time
        self._save_execution(execution)

    def get_execution(self, execution_id: str) -> Optional[Execution]:
        """Get execution by ID."""
        if execution_id in self.executions:
            return self.executions[execution_id]
        return self._load_execution(execution_id)

    def get_execution_states(self, execution_id: str) -> List[Dict]:
        """Get all states for an execution."""
        execution = self.get_execution(execution_id)
        if not execution:
            return []
        return execution.states

    def get_child_executions(self, parent_id: str) -> List[Execution]:
        """Get all child executions."""
        return [
            exec for exec in self.executions.values()
            if exec.parent_execution_id == parent_id
        ]

    def _save_execution(self, execution: Execution) -> None:
        """Save execution to storage."""
        execution_file = self.storage_dir / f"{execution.execution_id}.json"
        with open(execution_file, 'w') as f:
            json.dump(asdict(execution), f, indent=2, default=str)

    def _load_execution(self, execution_id: str) -> Optional[Execution]:
        """Load execution from storage."""
        execution_file = self.storage_dir / f"{execution_id}.json"
        if not execution_file.exists():
            return None
        
        with open(execution_file, 'r') as f:
            data = json.load(f)
            return Execution(**data)

    def cleanup_old_executions(self, days: int = 30) -> None:
        """Clean up executions older than specified days."""
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        for execution_file in self.storage_dir.glob("*.json"):
            if execution_file.stat().st_mtime < cutoff:
                execution_file.unlink() 