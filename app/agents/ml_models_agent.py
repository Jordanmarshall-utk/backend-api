from typing import Any, Dict, Optional
import logging

from app.agents.base import BaseAgent
from app.tools.memory import MemoryManager

logger = logging.getLogger(__name__)

class MLModelsAgent(BaseAgent):
    """Agent wrapper for ML Models automation and routing."""

    def __init__(self, name: str = "ML Models", desc: str = "Runs ML model automation"):
        super().__init__(name, desc)
        self.memory = MemoryManager()

    def confidence(self, payload: Dict[str, Any]) -> float:
        conf = 0.1
        if payload and isinstance(payload, dict):
            action = payload.get("action") or payload.get("next_agent")
            if str(action).lower() in {"ml_models", "ml", "ml_models_agent"}:
                conf = 0.8
            if payload.get("results_json") or payload.get("json_path"):
                conf = max(conf, 0.6)
            if payload.get("results_csv") or payload.get("csv_path"):
                conf = max(conf, 0.6)

        if self.memory.get_var("ml_auto_json_path") or self.memory.get_var("ml_auto_csv_path"):
            conf = max(conf, 0.6)

        return conf

    def run_agent(self, memory: MemoryManager, payload: Optional[Dict[str, Any]] = None) -> Any:
        try:
            from app.tools.ml_automation import run_automated_ml_model

            model_choice = memory.get_var("optimization_model_choice")
            auto_config = memory.get_var("ml_model_config", {})
            json_path = memory.get_var("ml_auto_json_path")
            csv_path = memory.get_var("ml_auto_csv_path")
            composition_csv = memory.get_var("ml_auto_composition_path")

            if payload and isinstance(payload, dict):
                json_path = payload.get("results_json") or payload.get("json_path") or json_path
                csv_path = payload.get("results_csv") or payload.get("csv_path") or csv_path
                composition_csv = payload.get("composition_csv") or composition_csv

            if not model_choice:
                model_choice = "Single-objective GP (scikit-learn)"

            result = run_automated_ml_model(
                model_choice=model_choice,
                json_path=json_path,
                csv_path=csv_path,
                composition_csv=composition_csv,
                auto_config=auto_config,
            )

            if memory and hasattr(memory, "log_event"):
                memory.log_event("ml_automation", {"result": result}, mode="ml_models")
            return result
        except Exception as e:
            logger.error(f"MLModelsAgent failed: {e}")
            if memory and hasattr(memory, "log_event"):
                memory.log_event("ml_automation", {"error": str(e)}, mode="ml_models")
            return {"success": False, "error": str(e)}
