"""
Platform-specific paths for Polaris Ahmadi.
Supports both development and PyInstaller frozen executable.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Application name for user data directory
APP_NAME = "PolarisAhmadi"
DB_FILENAME = "polaris.db"
UPDATER_DIRNAME = "updates"


def is_frozen() -> bool:
    """Return True if running as a PyInstaller frozen executable."""
    return getattr(sys, "frozen", False)


def get_runtime_root() -> Path:
    """
    Resolve the runtime root for resources and scripts.

    Priority:
    1) Explicit override via POLARIS_RUNTIME_ROOT
    2) PyInstaller temp root when frozen
    3) Current working directory if it contains app entry resources
    4) Project root relative to this module
    """
    explicit_root = os.environ.get("POLARIS_RUNTIME_ROOT", "").strip()
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()

    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))).resolve()

    return Path(__file__).resolve().parent.parent


def get_resource_path(relative_path: str) -> str:
    """
    Get absolute path to a resource. Works for both dev and PyInstaller.
    For frozen: resources are in sys._MEIPASS.
    For dev: relative to project root.
    """
    return str((get_runtime_root() / relative_path).resolve())


def get_user_data_dir() -> str:
    """
    Get platform-specific user data directory for persistent storage.
    - Windows: %APPDATA%\\PolarisAhmadi
    - macOS: ~/Library/Application Support/PolarisAhmadi
    - Linux: ~/.local/share/polaris_ahmadi
    """
    if sys.platform == "win32":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
        path = Path(base) / APP_NAME
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Application Support" / APP_NAME
    else:
        path = Path.home() / ".local" / "share" / "polaris_ahmadi"

    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_results_dir() -> str:
    """Directory for curve-fitting plots, JSON exports, and ML inputs."""
    custom = os.environ.get("POLARIS_RESULTS_DIR", "").strip()
    if custom:
        path = Path(custom).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return str(path.resolve())
    root = get_runtime_root()
    path = root / "results"
    path.mkdir(parents=True, exist_ok=True)
    return str(path.resolve())


def get_db_path() -> str:
    """
    Get path to the SQLite database file.
    Use POLARIS_DB_PATH to point to a shared database (e.g. network drive).
    Example: POLARIS_DB_PATH=Z:\\shared\\polaris.db or \\\\server\\share\\polaris.db
    """
    custom = os.environ.get("POLARIS_DB_PATH", "").strip()
    if custom:
        return str(Path(custom).expanduser().resolve())
    return str(Path(get_user_data_dir()) / DB_FILENAME)


def get_env_path() -> str:
    """Get path to .env file in user data dir (for packaged app)."""
    return str(Path(get_user_data_dir()) / ".env")


def get_updates_dir() -> str:
    """Get the updater working directory in user data."""
    path = Path(get_user_data_dir()) / UPDATER_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_update_download_path(filename: str = "release.zip") -> str:
    """Get path for a downloaded update archive."""
    return str(Path(get_updates_dir()) / filename)


def get_update_staging_dir() -> str:
    """Get the directory used to extract a staged app update."""
    path = Path(get_updates_dir()) / "staged"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_update_pending_path() -> str:
    """Get path for the pending update marker file."""
    return str(Path(get_updates_dir()) / "pending_update.json")


def get_update_helper_runtime_path() -> str:
    """Get path for the copied standalone updater helper script."""
    return str(Path(get_updates_dir()) / "update_helper_runtime.py")


def get_updater_log_path() -> str:
    """Get path for updater-related logging."""
    return str(Path(get_updates_dir()) / "updater.log")


def get_current_app_bundle_path() -> str | None:
    """Return the current macOS .app bundle path when running frozen."""
    executable_path = Path(sys.executable).resolve()
    for parent in executable_path.parents:
        if parent.suffix == ".app":
            return str(parent)
    return None


def get_current_windows_install_dir() -> str | None:
    """Return the current Windows portable app directory when running frozen."""
    if sys.platform != "win32" or not is_frozen():
        return None
    return str(Path(sys.executable).resolve().parent)


def get_current_windows_executable_path() -> str | None:
    """Return the current Windows executable path when running frozen."""
    if sys.platform != "win32" or not is_frozen():
        return None
    return str(Path(sys.executable).resolve())


def get_current_install_target_path() -> str | None:
    """Return the current install target for packaged desktop builds."""
    if sys.platform == "darwin":
        return get_current_app_bundle_path()
    if sys.platform == "win32":
        return get_current_windows_install_dir()
    return None


def get_current_launch_path() -> str | None:
    """Return the path that should be launched after installing an update."""
    if sys.platform == "darwin":
        return get_current_app_bundle_path()
    if sys.platform == "win32":
        return get_current_windows_executable_path()
    return None
