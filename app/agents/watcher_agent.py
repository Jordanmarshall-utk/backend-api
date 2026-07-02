from typing import Any, Dict, Optional, Callable
import os

from app.agents.base import BaseAgent
from app.tools.memory import MemoryManager

# Lazy import for HTTP client and socratic
_requests = None
_socratic = None

def _lazy_import_requests():
    """Lazy import requests module"""
    global _requests
    if _requests is None:
        import requests
        _requests = requests
    return _requests

def _lazy_import_socratic():
    """Lazy import socratic module"""
    global _socratic
    if _socratic is None:
        from app.tools import socratic
        _socratic = socratic
    return _socratic

def _lazy_import_instruct():
    """Lazy import instruct module"""
    from app.tools.instruct import WATCHER_ROUTING_INSTRUCTIONS
    return WATCHER_ROUTING_INSTRUCTIONS


class WatcherAgent(BaseAgent):
    """
    Supervisor agent that reacts to filesystem signals and decides which
    downstream agent should run next via the router.

    This agent communicates with the FastAPI watcher server to handle
    filesystem events and uses LLM-based confidence routing.
    """

    def __init__(self, name: str = "Watcher Agent", desc: Optional[str] = None,
                 router_factory: Optional[Callable[[], Any]] = None,
                 server_url: Optional[str] = None):
        super().__init__(name, desc or "Supervises file events and triggers next agent.")
        self.router_factory = router_factory
        self.server_url = server_url or os.environ.get("WATCHER_SERVER_URL", "http://localhost:8000")

    def confidence(self, params: Dict[str, Any]) -> float:
        """
        Confidence scoring for filesystem events.
        Uses simple heuristics to avoid LLM dependency in headless mode.
        """
        if params.get("source") != "filesystem":
            return 0.0
        
        # Simple heuristic-based confidence (no LLM required)
        trigger_file = params.get("trigger_file", "") or params.get("data_file", "")
        event_type = params.get("event", "created")
        
        # High confidence for CSV/Excel files (likely data files for curve fitting)
        if trigger_file and trigger_file.lower().endswith(('.csv', '.xlsx', '.xls')):
            return 0.8
        
        # Medium confidence for other file types
        if trigger_file:
            return 0.5
        
        # Low confidence if no file specified
        return 0.3

    def run_agent(self, memory: MemoryManager) -> None:
        """
        Handle a single filesystem event by communicating with the FastAPI server.
        
        The server handles the actual filesystem watching, and this agent
        processes events and routes them to the appropriate next agent.
        """
        # Log that watcher agent handled the event
        memory.log_event(
            event_type="watcher",
            payload={"note": "filesystem event handled by WatcherAgent"},
            mode="watcher",
        )
        
