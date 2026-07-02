"""
Experiment Memory System - Tracks completed experiments to prevent duplicates
and enable automated workflow memory.
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

_logger = logging.getLogger(__name__)

from app.tools.paths import is_frozen, get_user_data_dir


def _resolve_memory_dir(memory_dir: str | None = None) -> Path:
    """Resolve the experiment memory directory to a writable location."""
    raw_dir = memory_dir or "data"
    candidate = Path(raw_dir).expanduser()
    if candidate.is_absolute():
        return candidate
    if is_frozen():
        return Path(get_user_data_dir()) / candidate
    return candidate


class ExperimentMemory:
    """Tracks completed experiments and their results."""

    def __init__(
        self,
        memory_file: str = "experiment_memory.json",
        memory_dir: str | None = None,
    ):
        memory_path = Path(memory_file).expanduser()
        if memory_path.is_absolute():
            self.memory_file = memory_path.name
            self.memory_dir = memory_path.parent
        else:
            self.memory_file = memory_path.name
            self.memory_dir = _resolve_memory_dir(memory_dir)
        self._ensure_memory_dir()

    def _ensure_memory_dir(self):
        """Ensure memory directory exists."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def _load_memory(self) -> Dict[str, Any]:
        """Load experiment memory from file."""
        file_path = self.memory_dir / self.memory_file
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                _logger.warning("Could not load experiment memory: %s", e)
                return {"experiments": [], "metadata": {}}
        return {"experiments": [], "metadata": {}}

    def _save_memory(self, memory_data: Dict[str, Any]):
        """Save experiment memory to file."""
        file_path = self.memory_dir / self.memory_file
        try:
            self._ensure_memory_dir()
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(memory_data, f, indent=2)
        except Exception as e:
            _logger.error("Could not save experiment memory: %s", e)
    
    def add_experiment(
        self,
        experiment_id: str,
        description: str,
        composition: Optional[Dict[str, Any]] = None,
        results: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add a completed experiment to memory.
        
        Returns True if experiment was added, False if it already exists.
        """
        memory_data = self._load_memory()
        
        # Check if experiment already exists
        existing_ids = [exp.get("experiment_id") for exp in memory_data.get("experiments", [])]
        if experiment_id in existing_ids:
            return False
        
        experiment_entry = {
            "experiment_id": experiment_id,
            "description": description,
            "composition": composition or {},
            "results": results or {},
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }
        
        memory_data.setdefault("experiments", []).append(experiment_entry)
        
        # Update metadata
        memory_data["metadata"] = {
            "last_updated": datetime.now().isoformat(),
            "total_experiments": len(memory_data["experiments"]),
        }
        
        self._save_memory(memory_data)
        return True
    
    def has_experiment(self, experiment_id: str) -> bool:
        """Check if an experiment has already been completed."""
        memory_data = self._load_memory()
        existing_ids = [exp.get("experiment_id") for exp in memory_data.get("experiments", [])]
        return experiment_id in existing_ids
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get details of a specific experiment."""
        memory_data = self._load_memory()
        for exp in memory_data.get("experiments", []):
            if exp.get("experiment_id") == experiment_id:
                return exp
        return None
    
    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """Get all completed experiments."""
        memory_data = self._load_memory()
        return memory_data.get("experiments", [])
    
    def get_experiment_summary(self) -> str:
        """Get a summary of all completed experiments."""
        experiments = self.get_all_experiments()
        if not experiments:
            return "No experiments completed yet."
        
        summary_parts = [f"**Total Experiments Completed:** {len(experiments)}\n"]
        
        for i, exp in enumerate(experiments[-10:], 1):  # Last 10 experiments
            exp_id = exp.get("experiment_id", "Unknown")
            desc = exp.get("description", "No description")
            timestamp = exp.get("timestamp", "Unknown")
            summary_parts.append(f"{i}. **{exp_id}**: {desc[:100]}... ({timestamp[:10]})")
        
        return "\n".join(summary_parts)
    
    def find_similar_experiments(
        self,
        composition: Dict[str, Any],
        threshold: float = 0.9,
    ) -> List[Dict[str, Any]]:
        """
        Find experiments with similar compositions.
        Returns experiments with composition similarity >= threshold.
        """
        experiments = self.get_all_experiments()
        similar = []
        
        for exp in experiments:
            exp_comp = exp.get("composition", {})
            if not exp_comp:
                continue
            
            # Simple similarity check (can be enhanced)
            similarity = self._calculate_composition_similarity(composition, exp_comp)
            if similarity >= threshold:
                similar.append({**exp, "similarity": similarity})
        
        return sorted(similar, key=lambda x: x.get("similarity", 0), reverse=True)
    
    def _calculate_composition_similarity(
        self,
        comp1: Dict[str, Any],
        comp2: Dict[str, Any],
    ) -> float:
        """Calculate similarity between two compositions."""
        if not comp1 or not comp2:
            return 0.0
        
        # Get all keys
        all_keys = set(comp1.keys()) | set(comp2.keys())
        if not all_keys:
            return 0.0
        
        # Calculate similarity
        matches = 0
        total_diff = 0.0
        
        for key in all_keys:
            val1 = comp1.get(key, 0)
            val2 = comp2.get(key, 0)
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                diff = abs(val1 - val2)
                total_diff += diff
                if diff < 0.01:  # Very close values
                    matches += 1
            elif val1 == val2:
                matches += 1
        
        # Normalize similarity
        similarity = matches / len(all_keys) if all_keys else 0.0
        return similarity
    
    def clear_memory(self):
        """Clear all experiment memory."""
        memory_data = {"experiments": [], "metadata": {}}
        self._save_memory(memory_data)


def get_experiment_memory(memory_manager=None) -> ExperimentMemory:
    """Get or create experiment memory instance backed by memory_manager (SQLite)."""
    if memory_manager is not None:
        try:
            exp_mem = memory_manager.get_var("experiment_memory")
            if exp_mem is None:
                memory_file = memory_manager.get_var("experiment_memory_file", "experiment_memory.json")
                data_dir = memory_manager.get_var("experiment_data_dir", "data")
                exp_mem = ExperimentMemory(memory_file=memory_file, memory_dir=data_dir)
                memory_manager.set_var("experiment_memory", exp_mem)
            return exp_mem
        except (RuntimeError, AttributeError):
            pass

    return ExperimentMemory()


