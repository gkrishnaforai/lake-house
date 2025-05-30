import logging
import json
from datetime import datetime
from langchain.tools import BaseTool
from typing import Any, Dict

class BaseLoggingTool(BaseTool):
    """
    BaseTool subclass that provides robust logging for all tool invocations.
    Logs input, output, and exceptions in both human-readable and JSON format.
    """
    log_file_path: str = "tool_logs.log"  # Can be overridden per tool

    def log_event(self, event_type: str, data: Dict[str, Any]):
        # Standard log
        logging.info(f"[{self.name}] {event_type}: {data}")
        # Structured JSON log
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "tool": self.name,
            "event_type": event_type,
            **data
        }
        with open(self.log_file_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def run(self, *args, **kwargs):
        self.log_event("run_start", {"args": args, "kwargs": kwargs})
        try:
            result = super().run(*args, **kwargs)
            self.log_event("run_success", {"result": result})
            return result
        except Exception as e:
            self.log_event("run_exception", {
                "exception": str(e),
                "args": args,
                "kwargs": kwargs
            })
            logging.error(f"[{self.name}] Exception in run", exc_info=True)
            raise

    def _run(self, *args, **kwargs):
        self.log_event("_run_start", {"args": args, "kwargs": kwargs})
        try:
            result = super()._run(*args, **kwargs)
            self.log_event("_run_success", {"result": result})
            return result
        except Exception as e:
            self.log_event("_run_exception", {
                "exception": str(e),
                "args": args,
                "kwargs": kwargs
            })
            logging.error(f"[{self.name}] Exception in _run", exc_info=True)
            raise 