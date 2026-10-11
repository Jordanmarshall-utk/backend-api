"""
In-memory runtime state for non-serializable objects.
These are never persisted to the database.
"""

from typing import Any, Dict

# Keys that must stay in memory (subprocess, file handles, model objects, etc.)
RUNTIME_KEYS = frozenset({
    "watcher_server_process",
    "watcher_log_file_handle",
    "watcher_server_pid",
    "watcher_server_running",
    "watcher_log_file",
    "mcp_orchestrator_process",
    "mcp_orchestrator_log_file_handle",
    "mcp_orchestrator_pid",
    "mcp_orchestrator_log_file",
    "gp_model",
    "gp_training_data",
    "cf_data_file",
    "cf_composition_file",
    "experiment_memory",
    "metrics_prev",
    "last_literature_handoff",
})

_runtime: Dict[str, Any] = {}


def get(key: str, default: Any = None) -> Any:
    """Get a runtime-only variable."""
    return _runtime.get(key, default)


def set(key: str, value: Any) -> None:
    """Set a runtime-only variable."""
    _runtime[key] = value


def delete(key: str) -> None:
    """Delete a runtime variable."""
    if key in _runtime:
        del _runtime[key]


def clear_ephemeral() -> None:
    """Clear all runtime state (e.g. on session reset)."""
    _runtime.clear()


def is_runtime_key(key: str) -> bool:
    """Return True if key should be stored in runtime, not DB."""
    return key in RUNTIME_KEYS
