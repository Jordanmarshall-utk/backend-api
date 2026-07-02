from typing import Dict, Any, Optional

from app.agents.base import BaseAgent
from app.tools.memory import MemoryManager

class FallbackAgent(BaseAgent):
    """
    Fallback agent that handles routing failures and provides error recovery.
    This agent is called when:
    - No other agent has sufficient confidence to handle a request
    - An agent fails during execution
    - Manual workflow is exhausted
    """
    
    def __init__(self, name: str = "Fallback Agent", desc: Optional[str] = None):
        super().__init__(name, desc or "Handles routing failures and provides error recovery")
        self.memory = MemoryManager()

    def confidence(self, params: Dict[str, Any]) -> float:
        """
        Fallback agent has low confidence by default - it should only be used
        when no other agent can handle the request.
        """
        # Only confident if explicitly requested as fallback
        if params.get("force_fallback") or params.get("error_occurred"):
            return 1.0
        return 0.0

    def run_agent(self, memory: MemoryManager) -> Dict[str, Any]:
        """
        Handle fallback scenarios - provide user feedback and recovery options.
        """
        memory.log_event(
            event_type="fallback",
            payload={
                "message": "Fallback agent activated",
                "reason": "No suitable agent found or routing error occurred"
            },
            mode="fallback"
        )

        return {"status": "error", "message": "No suitable agent found or routing error occurred."}