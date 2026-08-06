"""
FastAPI server that watches the filesystem and triggers agents via HTTP endpoints.

This uses `watchdog`, which internally selects the best backend:
- Windows: FileSystemWatcher
- macOS: FSEvents
- Linux/other: inotify/polling

The server exposes REST endpoints that the WatcherAgent can call to trigger routing.
"""

import os
import json
import logging
import warnings
import time
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from app.tools.memory import MemoryManager
from app.agents.router import AgentRouter

# Configure logging with more verbose output to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Also print critical messages to stdout for immediate visibility
def log_and_print(message: str, level: str = "INFO"):
    """Log message and also print to stdout for immediate visibility."""
    # Handle Windows console encoding issues with emojis
    def safe_print(text: str):
        """Print text, handling Unicode encoding errors on Windows."""
        try:
            # Try to configure Windows console for UTF-8 if possible
            import sys
            if sys.platform == 'win32':
                try:
                    # Try to set UTF-8 encoding
                    if hasattr(sys.stdout, 'reconfigure'):
                        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
                except (AttributeError, ValueError, OSError):
                    pass  # Fall back to error handling
            print(text)
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            # Fallback: encode to ASCII, replacing problematic characters
            try:
                safe_text = text.encode('ascii', 'replace').decode('ascii')
                print(safe_text)
            except Exception:
                # Last resort: extract just the level and basic message
                import re
                # Try to extract readable parts
                level_match = re.search(r'\[(\w+)\]', text)
                msg_match = re.search(r'\]\s*(.+)', text)
                if level_match and msg_match:
                    level_part = level_match.group(1)
                    msg_part = msg_match.group(1).encode('ascii', 'replace').decode('ascii')
                    print(f"[{level_part}] {msg_part}")
                else:
                    print(f"[{level}] [Message contains non-ASCII characters]")
    
    if level == "INFO":
        logger.info(message)
        safe_print(f"[INFO] {message}")
    elif level == "WARNING":
        logger.warning(message)
        safe_print(f"[WARNING] {message}")
    elif level == "ERROR":
        logger.error(message)
        safe_print(f"[ERROR] {message}")
    elif level == "DEBUG":
        logger.debug(message)
        safe_print(f"[DEBUG] {message}")

# Ensure API key is available from environment for headless mode
def _ensure_api_key():
    """Ensure API key is available in environment for headless mode."""
    api_key = (
        os.getenv("HUGGINGFACE_API_KEY")
        or os.getenv("HF_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
    )
    if not api_key:
        logger.warning(
            "No API key found in environment variables (HUGGINGFACE_API_KEY, HF_API_KEY, or LLM_API_KEY). "
            "LLM features may not work. Set the API key as an environment variable before starting the watcher server."
        )
    else:
        logger.info("API key found in environment.")
    return api_key

# Check API key on module load
_ensure_api_key()

def _build_session_context(mem: MemoryManager) -> Dict[str, Any]:
    """Build session/experiment context from database for routing decisions."""
    try:
        mem.init_session()
        exp_id = mem.get_var("current_experiment_id") or 0
        user_id = mem.get_var("current_user_id") or ""
        return {
            "current_experiment_id": int(exp_id) if exp_id else None,
            "current_user_id": user_id,
            "stage": mem.get_var("stage", ""),
            "has_hypothesis": bool(mem.get_var("last_hypothesis") or mem.view_component("hypothesis")),
            "hypothesis_preview": (str(mem.get_var("last_hypothesis") or "") or str(mem.view_component("hypothesis") or ""))[:200] or None,
            "has_experimental_outputs": bool(mem.get_var("experimental_outputs")),
            "has_curve_fitting_results": bool(mem.get_var("curve_fitting_results")),
            "has_analysis_results": bool(mem.get_var("analysis_results")),
            "has_gp_results": bool(mem.get_var("gp_results")),
            "research_goal": mem.get_var("research_goal", "")[:200] or None,
            "workflow_step": mem.get_var("workflow_step", ""),
            "hypothesis_ready": mem.get_var("hypothesis_ready", False),
        }
    except Exception as e:
        logger.warning(f"Could not build session context: {e}")
        return {}


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup - init database/session so watcher can access stored data
    log_and_print("Watcher server starting up...", "INFO")
    try:
        memory.init_session()
    except Exception as e:
        log_and_print(f"Could not init memory session: {e}", "WARNING")
    yield
    # Shutdown
    log_and_print("Watcher server shutting down...", "INFO")
    global observer
    if observer is not None and observer.is_alive():
        observer.stop()
        observer.join()
        log_and_print("Observer stopped on shutdown", "INFO")

# FastAPI app with lifespan
app = FastAPI(title="POLARIS Watcher Server", version="1.0.0", lifespan=lifespan)

# Global state
memory = MemoryManager()
observer: Optional[Observer] = None
watch_dir: Optional[Path] = None

# Request/Response models
class FileEventRequest(BaseModel):
    file_path: str
    event_type: str = "created"
    metadata: Optional[Dict[str, Any]] = None

class RouteRequest(BaseModel):
    payload: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    watch_dir: Optional[str] = None
    observer_running: bool = False


class Watcher(FileSystemEventHandler):
    """
    Handles filesystem events. When certain files appear (e.g.
    `output_from_agent_A.json`), this notifies the router so the next agent
    in the workflow can run.
    """

    def __init__(self, memory: MemoryManager, watch_dir: Path):
        super().__init__()
        self.memory = memory
        self.watch_dir = watch_dir
        self.processed_files = set()  # Track processed files to avoid duplicates
        self.file_timestamps = {}  # Track file modification times
        logger.info(f"🔧 Watcher initialized for directory: {watch_dir}")
    
    def on_any_event(self, event):
        """Log all events for debugging - catches ALL file system events"""
        if not event.is_directory:
            log_and_print(f"📡 File system event: {event.event_type} - {event.src_path}", "DEBUG")

    def _build_router(self) -> AgentRouter:
        """
        Construct a router instance.  In this headless context we only need
        minimal agent objects – they can ignore Streamlit UI.
        """
        # Lazy imports to avoid Streamlit-only code at server startup
        from app.agents.hypothesis_agent import HypothesisAgent
        from app.agents.experiment_agent import ExperimentAgent
        from app.agents.curve_fitting_agent import CurveFittingAgent
        from app.agents.analysis_agent import AnalysisAgent
        from app.agents.ml_models_agent import MLModelsAgent
        from app.agents.watcher_agent import WatcherAgent
        from app.agents.fallback_agent import FallbackAgent

        # Dummy descriptions; router only needs names + confidence/run_agent
        hyp = HypothesisAgent(name="Hypothesis Agent", desc="Background", question="")
        exp = ExperimentAgent(name="Experiment Agent", desc="Background", params_const={})
        cf = CurveFittingAgent(name="Curve Fitting", desc="Background")
        analysis = AnalysisAgent(name="Analysis Agent", desc="Background")
        ml_models = MLModelsAgent(name="ML Models", desc="Background")
        fb = FallbackAgent(name="Fallback Agent", desc="Handles failures")

        return AgentRouter(
            agents=[WatcherAgent(), hyp, exp, cf, ml_models, analysis],
            fallback_agent=fb,
        )

    def _route_next(self, trigger_file: Path) -> None:
        payload: Dict[str, Any] = {
            "source": "filesystem",
            "trigger_file": str(trigger_file),
            "event": "created",
        }
        session_ctx = _build_session_context(self.memory)
        if session_ctx:
            payload["session_context"] = session_ctx
        router = self._build_router()
        try:
            router.route(payload, self.memory)
        except Exception as e:
            logger.error(f"Error routing: {e}")

    def on_created(self, event):
        """Handle file creation events"""
        log_and_print(f"🔔 File system event detected: {event.event_type} - {event.src_path}", "INFO")
        log_and_print(f"   Event is_directory: {event.is_directory}", "INFO")
        
        if event.is_directory:
            logger.debug(f"Ignoring directory creation: {event.src_path}")
            return

        path = Path(event.src_path)
        
        # Wait a moment for file to be fully written (especially on Windows/OneDrive)
        import time
        time.sleep(0.5)
        
        # Verify file exists and is readable
        if not path.exists():
            log_and_print(f"⚠️ File does not exist (may have been deleted): {path}", "WARNING")
            return
        
        try:
            # Try to access file to ensure it's fully written
            file_size = path.stat().st_size
        except (OSError, PermissionError) as e:
            log_and_print(f"⚠️ Cannot access file: {e}", "WARNING")
            return
        
        filename = path.name
        file_ext = path.suffix.lower()
        
        log_and_print(f"📄 Processing file: {filename} (extension: {file_ext}, size: {file_size:,} bytes = {file_size/1024:.2f} KB)", "INFO")

        # Detect curve fitting data files (CSV, Excel)
        curve_fitting_extensions = ['.csv', '.xlsx', '.xls']
        is_curve_fitting_file = file_ext in curve_fitting_extensions
        
        # Detect JSON output files from agents
        is_agent_output = filename.lower().startswith("output_from_") and file_ext == ".json"
        
        log_and_print(f"File type detection - Curve fitting: {is_curve_fitting_file}, Agent output: {is_agent_output}", "INFO")
        
        # Log the event
        try:
            self.memory.log_event(
                "watcher",
                {
                    "file": str(path),
                    "event": "created",
                    "file_type": "curve_fitting_data" if is_curve_fitting_file else "agent_output" if is_agent_output else "other"
                },
                mode="watcher",
            )
        except Exception as e:
            logger.warning(f"Could not log event to memory: {e}")
        
        # Route based on file type
        if is_curve_fitting_file:
            # Track this file to avoid duplicate processing
            try:
                stat_info = path.stat()
                file_id = f"{path.resolve()}_{stat_info.st_mtime}"
                
                if file_id in self.processed_files:
                    log_and_print(f"⏭️ Skipping already processed file: {filename}", "INFO")
                    return
                
                self.processed_files.add(file_id)
                self.file_timestamps[path] = (stat_info.st_size, stat_info.st_mtime)
            except Exception:
                pass  # Continue even if tracking fails
            
            # Automatically trigger curve fitting for data files
            log_and_print(f"✅ Detected curve fitting data file: {path.name}", "INFO")
            log_and_print(f"🚀 Triggering curve fitting for: {filename}", "INFO")
            self._trigger_curve_fitting(path)
        elif is_agent_output:
            # Route agent output files
            log_and_print(f"Routing agent output file: {path.name}", "INFO")
            self._route_next(path)
        else:
            log_and_print(f"File {filename} does not match curve fitting or agent output patterns. Ignoring.", "INFO")
    
    def on_modified(self, event):
        """Handle file modification events (OneDrive sync often triggers this instead of created)"""
        if event.is_directory:
            return
        
        path = Path(event.src_path)
        filename = path.name
        file_ext = path.suffix.lower()
        
        # Only process if file exists and is a data file
        if not path.exists():
            return
        
        curve_fitting_extensions = ['.csv', '.xlsx', '.xls']
        is_curve_fitting_file = file_ext in curve_fitting_extensions
        
        if is_curve_fitting_file:
            try:
                # Get file stats
                stat_info = path.stat()
                file_size = stat_info.st_size
                mod_time = stat_info.st_mtime
                
                # Create unique file identifier
                file_id = f"{path.resolve()}_{mod_time}"
                
                # Check if we've already processed this exact file version
                if file_id in self.processed_files:
                    log_and_print(f"⏭️ Skipping already processed file: {filename} (mod_time: {mod_time})", "DEBUG")
                    return
                
                # Check if this is a new modification (file size or time changed)
                if path in self.file_timestamps:
                    old_size, old_time = self.file_timestamps[path]
                    if old_size == file_size and old_time == mod_time:
                        log_and_print(f"⏭️ File {filename} hasn't changed since last check", "DEBUG")
                        return
                
                # Update tracking
                self.file_timestamps[path] = (file_size, mod_time)
                
                if file_size > 1000:  # Only process files larger than 1KB (to avoid empty/temp files)
                    log_and_print(f"📝 File modified event detected: {filename} (size: {file_size:,} bytes = {file_size/1024:.2f} KB)", "INFO")
                    
                    # Wait a bit for file to stabilize (OneDrive sync can take time)
                    import time
                    time.sleep(2)  # Increased wait time for OneDrive
                    
                    # Verify file still exists and hasn't changed
                    if path.exists():
                        new_stat = path.stat()
                        if new_stat.st_size == file_size and new_stat.st_mtime == mod_time:
                            # File is stable, process it
                            log_and_print(f"✅ File is stable, triggering curve fitting: {filename}", "INFO")
                            self.processed_files.add(file_id)
                            self._trigger_curve_fitting(path)
                        else:
                            log_and_print(f"⏳ File {filename} is still changing, will retry on next event", "INFO")
                    else:
                        log_and_print(f"⚠️ File {filename} disappeared during wait", "WARNING")
                else:
                    log_and_print(f"⏭️ Skipping small file (likely temp): {filename} ({file_size} bytes)", "DEBUG")
            except Exception as e:
                log_and_print(f"Error processing modified event for {filename}: {e}", "WARNING")
                logger.warning(f"Error processing modified event: {e}", exc_info=True)
    
    def _analyze_file_for_parameters(self, data_file: Path) -> Dict[str, Any]:
        """Analyze the data file to infer curve fitting parameters."""
        params = {
            "max_peaks": 4,
            "r2_target": 0.90,
            "max_attempts": 3,
            "read_type": "em_spectrum",
            "read_selection": "auto",
            "wells_to_analyze": None,
            "plate_format": "96-well (8x12)",
            "api_delay_seconds": 0.5,
        }
        
        try:
            import pandas as pd
            import re
            
            # Read a sample of the file to analyze structure
            if data_file.suffix.lower() == '.csv':
                # Read first 50 rows to analyze structure
                df_sample = pd.read_csv(data_file, header=None, nrows=50, engine='python')
            elif data_file.suffix.lower() in ['.xlsx', '.xls']:
                df_sample = pd.read_excel(data_file, header=None, nrows=50)
            else:
                logger.warning(f"Unknown file type: {data_file.suffix}")
                return params
            
            # Look for read headers (e.g., "Read 1: EM Spectrum")
            read_pattern = re.compile(r"^Read\s+(\d+):", re.I)
            em_spectrum_pattern = re.compile(r"EM\s+Spectrum", re.I)
            
            em_reads = []
            all_reads = []
            
            for idx, row in df_sample.iterrows():
                row_str = ' '.join([str(val) for val in row.values if pd.notna(val)])
                read_match = read_pattern.search(row_str)
                if read_match:
                    read_num = int(read_match.group(1))
                    all_reads.append(read_num)
                    if em_spectrum_pattern.search(row_str):
                        em_reads.append(read_num)
            
            # Determine read selection - use first and last read only for auto-triggered runs
            if em_reads:
                if len(em_reads) >= 2:
                    # Use first and last EM Spectrum read
                    first_read = em_reads[0]
                    last_read = em_reads[-1]
                    params["read_selection"] = f"{first_read},{last_read}"
                    logger.info(f"Found EM Spectrum reads: {em_reads}, using first and last: {first_read},{last_read}")
                    log_and_print(f"Will analyze only first and last reads: {first_read} and {last_read}", "INFO")
                else:
                    # Only one read found, use it
                    params["read_selection"] = f"{em_reads[0]}"
                    logger.info(f"Found single EM Spectrum read: {em_reads[0]}")
            elif all_reads:
                if len(all_reads) >= 2:
                    # Use first and last read
                    first_read = all_reads[0]
                    last_read = all_reads[-1]
                    params["read_selection"] = f"{first_read},{last_read}"
                    logger.info(f"Found reads: {all_reads}, using first and last: {first_read},{last_read}")
                    log_and_print(f"Will analyze only first and last reads: {first_read} and {last_read}", "INFO")
                else:
                    # Only one read found, use it
                    params["read_selection"] = f"{all_reads[0]}"
                    logger.info(f"Found single read: {all_reads[0]}")
            else:
                params["read_selection"] = "auto"
                logger.info("No read headers found, using auto detection")
            
            # Analyze plate format by looking at well columns
            # Read more rows to find well columns
            try:
                if data_file.suffix.lower() == '.csv':
                    df_full = pd.read_csv(data_file, header=None, nrows=200, engine='python')
                else:
                    df_full = pd.read_excel(data_file, header=None, nrows=200)
                
                # Look for well column headers (A1-H12 pattern)
                well_pattern = re.compile(r'^[A-H](?:[1-9]|1[0-2])$', re.I)
                well_columns = []
                
                for col_idx in range(min(20, len(df_full.columns))):  # Check first 20 columns
                    col_name = str(df_full.iloc[0, col_idx]) if len(df_full) > 0 else ""
                    if well_pattern.match(col_name.strip()):
                        well_columns.append(col_name.strip().upper())
                
                if well_columns:
                    # Determine plate format based on well range
                    rows = set([w[0] for w in well_columns])
                    cols = set([int(w[1:]) for w in well_columns])
                    
                    max_row = max([ord(r) - ord('A') + 1 for r in rows]) if rows else 0
                    max_col = max(cols) if cols else 0
                    
                    if max_row == 8 and max_col == 12:
                        params["plate_format"] = "96-well (8x12)"
                    elif max_row == 16 and max_col == 24:
                        params["plate_format"] = "384-well (16x24)"
                    elif max_row == 5 and max_col == 7:
                        params["plate_format"] = "35-well (5x7)"
                    else:
                        params["plate_format"] = f"{max_row * max_col}-well ({max_row}x{max_col})"
                    
                    logger.info(f"📋 Detected plate format: {params['plate_format']} from {len(well_columns)} well columns")
            except Exception as e:
                logger.warning(f"Could not analyze plate format: {e}")
            
            # Estimate max peaks based on data complexity (simple heuristic)
            # This is a placeholder - could be improved with actual spectral analysis
            try:
                if data_file.suffix.lower() == '.csv':
                    df_check = pd.read_csv(data_file, header=None, nrows=100, engine='python')
                else:
                    df_check = pd.read_excel(data_file, header=None, nrows=100)
                
                # Count non-null values as a proxy for data complexity
                non_null_ratio = df_check.notna().sum().sum() / (df_check.shape[0] * df_check.shape[1])
                if non_null_ratio > 0.8:
                    params["max_peaks"] = 6  # More data suggests more peaks possible
                elif non_null_ratio > 0.5:
                    params["max_peaks"] = 4
                else:
                    params["max_peaks"] = 3
            except Exception:
                pass  # Keep default
            
        except Exception as e:
            logger.warning(f"Error analyzing file for parameters: {e}, using defaults")
        
        return params
    
    def _trigger_curve_fitting(self, data_file: Path):
        """
        Notify Streamlit app about detected file - let Streamlit run curve fitting in its own context.
        This way curve fitting runs with full UI support instead of headless mode.
        """
        log_and_print(f"File detected: {data_file.name}", "INFO")
        try:
            # Verify file exists
            if not data_file.exists():
                log_and_print(f"File does not exist: {data_file}", "ERROR")
                return
            
            file_size = data_file.stat().st_size
            log_and_print(f"File exists: {data_file.name} (size: {file_size:,} bytes = {file_size/1024:.2f} KB)", "INFO")
            
            # Analyze file to determine parameters
            log_and_print(f"Analyzing file to determine curve fitting parameters...", "INFO")
            inferred_params = self._analyze_file_for_parameters(data_file)
            log_and_print(f"Inferred parameters: {inferred_params}", "INFO")
            
            # Store trigger info for Streamlit app to detect and run curve fitting
            # This includes the file path and inferred parameters
            trigger_info = {
                "triggered_file": str(data_file),
                "trigger_time": time.time(),
                "timestamp": datetime.now().isoformat(),
                "parameters": inferred_params,  # Include inferred parameters for Streamlit to use
                "file_size": file_size,
            }
            
            # Store in JSON file that Streamlit can read
            trigger_file = Path(__file__).parent.parent / "watcher_trigger_info.json"
            with open(trigger_file, 'w') as f:
                json.dump(trigger_info, f, indent=2)
            
            log_and_print(f"Stored trigger info for Streamlit app", "INFO")
            log_and_print(f"Streamlit will detect this and run curve fitting with UI support", "INFO")
            
            # Optionally upload to Jupyter if configured
            self._upload_to_jupyter_if_enabled(data_file)
            
            log_and_print(f"Successfully notified Streamlit about file: {data_file.name}", "INFO")
        except Exception as e:
            log_and_print(f"Error notifying Streamlit: {e}", "ERROR")
            logger.error(f"Error notifying Streamlit: {e}", exc_info=True)
    
    def _store_trigger_info(self, file_path: str):
        """Store trigger info in a JSON file for Streamlit app to detect."""
        try:
            import time
            trigger_info = {
                "triggered_file": file_path,
                "trigger_time": time.time(),
                "timestamp": datetime.now().isoformat()
            }
            # Store in a file that Streamlit can access
            trigger_file = Path(__file__).parent.parent / "watcher_trigger_info.json"
            with open(trigger_file, 'w') as f:
                json.dump(trigger_info, f)
            logger.info(f"💾 Stored trigger info: {file_path}")
        except Exception as e:
            logger.warning(f"Could not store trigger info: {e}")
    
    def _upload_to_jupyter_if_enabled(self, file_path: Path):
        """Upload file to Jupyter server if auto-upload is enabled."""
        try:
            # Check if Jupyter upload is enabled (would need to be passed or stored)
            # For now, check environment variables or use defaults
            upload_enabled = os.environ.get("JUPYTER_UPLOAD_ENABLED", "false").lower() == "true"
            if not upload_enabled:
                return
            
            jupyter_url = os.environ.get("JUPYTER_SERVER_URL", "")
            jupyter_token = os.environ.get("JUPYTER_TOKEN", "")
            notebook_path = os.environ.get("JUPYTER_NOTEBOOK_PATH", "Automated Agent")
            
            if not jupyter_url:
                logger.warning("Jupyter upload enabled but no server URL configured")
                return
            
            # Import here to avoid dependency if not needed
            import requests
            import base64
            
            # Read file content
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            # Determine if text or binary
            is_text = file_path.suffix.lower() in ['.csv', '.txt', '.py', '.json']
            
            # Construct API URL
            server_url = jupyter_url.rstrip('/')
            api_path = f"{notebook_path}/{file_path.name}"
            api_url = f"{server_url}/api/contents/{api_path}"
            
            # Prepare content
            if is_text:
                content_data = {
                    "type": "file",
                    "format": "text",
                    "content": file_content.decode('utf-8')
                }
            else:
                content_data = {
                    "type": "file",
                    "format": "base64",
                    "content": base64.b64encode(file_content).decode()
                }
            
            # Upload
            headers = {}
            if jupyter_token:
                headers["Authorization"] = f"token {jupyter_token}"
            
            response = requests.put(api_url, json=content_data, headers=headers, timeout=10)
            
            if response.status_code in [200, 201]:
                logger.info(f"Uploaded {file_path.name} to Jupyter: {api_path}")
            else:
                logger.warning(f"Failed to upload to Jupyter: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Error uploading to Jupyter: {e}")


# Global watcher instance
watcher_handler: Optional[Watcher] = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        watch_dir=str(watch_dir) if watch_dir else None,
        observer_running=observer is not None and observer.is_alive()
    )


@app.post("/watch/start")
async def start_watching(watch_directory: Optional[str] = None):
    """Start watching a directory for filesystem events"""
    global observer, watch_dir, watcher_handler
    
    logger.info(f"📥 Received start watching request with directory: {watch_directory}")
    
    # Check if observer is already running
    if observer is not None and observer.is_alive():
        current_watch_dir = str(watch_dir) if watch_dir else "unknown"
        logger.warning(f"⚠️ Observer is already running, watching: {current_watch_dir}")
        
        # If watching a different directory, stop and restart
        if watch_directory:
            watch_dir_str = watch_directory.strip().strip('"').strip("'")
            import urllib.parse
            watch_dir_str = urllib.parse.unquote(watch_dir_str)
            new_watch_dir = Path(watch_dir_str).expanduser()
            if watch_dir and watch_dir.resolve() != new_watch_dir.resolve():
                logger.info(f"🔄 Switching from {watch_dir} to {new_watch_dir}")
                observer.stop()
                observer.join()
                observer = None
                watcher_handler = None
            else:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Observer is already running",
                        "watch_dir": current_watch_dir,
                        "message": "Observer is already watching this directory"
                    }
                )
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Observer is already running",
                    "watch_dir": current_watch_dir,
                    "message": "Stop watching first, then start again"
                }
            )
    
    # Determine watch directory
    # Priority: 1) parameter, 2) environment variable, 3) default
    if watch_directory:
        # Strip quotes if present
        watch_dir_str = watch_directory.strip().strip('"').strip("'")
        watch_dir = Path(watch_dir_str)
    else:
        # Try to get from environment or use default
        default_dir = os.environ.get("WATCH_DIR") or os.environ.get("WATCHER_DIRECTORY")
        if default_dir:
            default_dir = default_dir.strip().strip('"').strip("'")
            watch_dir = Path(default_dir)
        else:
            # Default to watched directory in current working directory
            watch_dir = Path("watched")
    
    # Expand user path if it starts with ~
    watch_dir = watch_dir.expanduser()
    
    # Check if directory exists
    if not watch_dir.exists():
        # Try to create it
        try:
            watch_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created watch directory: {watch_dir}")
        except PermissionError as e:
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied: Cannot access or create watch directory {watch_dir}. Please check permissions."
            )
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Watch directory does not exist and could not be created: {watch_dir}. Error: {e}. Please create the directory manually or check the path."
            )
    
    # Verify it's actually a directory
    if not watch_dir.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Path exists but is not a directory: {watch_dir}"
        )
    
    watcher_handler = Watcher(memory=memory, watch_dir=watch_dir)
    observer = Observer()
    
    # Schedule the watcher - use absolute path
    abs_watch_dir = watch_dir.resolve()
    observer.schedule(watcher_handler, path=str(abs_watch_dir), recursive=False)
    observer.start()
    
    log_and_print(f"✅ Started watching directory: {abs_watch_dir}", "INFO")
    log_and_print(f"📁 Observer is {'alive' if observer.is_alive() else 'not alive'}", "INFO")
    log_and_print(f"👁️ Watching for files: .csv, .xlsx, .xls", "INFO")
    
    # Verify directory contents and show existing files
    try:
        files_in_dir = list(abs_watch_dir.glob("*"))
        log_and_print(f"📂 Directory contains {len(files_in_dir)} items", "INFO")
        csv_files = [f for f in files_in_dir if f.suffix.lower() in ['.csv', '.xlsx', '.xls']]
        if csv_files:
            log_and_print(f"📊 Found {len(csv_files)} CSV/Excel files already in directory", "INFO")
            log_and_print(f"💡 IMPORTANT: Files created BEFORE watching started will NOT be automatically processed!", "WARNING")
            log_and_print(f"💡 Use the '🔍 Scan Directory for Files' button in Watcher Control to process existing files.", "INFO")
            
            # Show list of files found
            print("")  # Empty line
            log_and_print(f"Existing CSV/Excel files found:", "INFO")
            for csv_file in sorted(csv_files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]:  # Show newest 10
                try:
                    file_size = csv_file.stat().st_size / 1024  # KB
                    mod_time = datetime.fromtimestamp(csv_file.stat().st_mtime)
                    log_and_print(f"   📄 {csv_file.name} ({file_size:.2f} KB, modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')})", "INFO")
                except Exception:
                    log_and_print(f"   📄 {csv_file.name}", "INFO")
            if len(csv_files) > 10:
                log_and_print(f"   ... and {len(csv_files) - 10} more files", "INFO")
            print("")  # Empty line
    except Exception as e:
        log_and_print(f"Could not list directory contents: {e}", "WARNING")
    
    return {
        "status": "started",
        "watch_dir": str(watch_dir),
        "observer_running": observer.is_alive(),
        "message": f"Watching directory: {watch_dir} for CSV/Excel files to trigger curve fitting"
    }


@app.post("/watch/stop")
async def stop_watching():
    """Stop watching for filesystem events"""
    global observer
    
    if observer is None or not observer.is_alive():
        return JSONResponse(
            status_code=400,
            content={"error": "Observer is not running"}
        )
    
    observer.stop()
    observer.join()
    observer = None
    
    logger.info("Stopped watching")
    
    return {"status": "stopped", "message": "Observer stopped"}


@app.get("/suggest-agent")
@app.post("/suggest-agent")
async def suggest_agent(request: Request):
    """
    Suggest which agent should run next based on stored session/experiment data.
    GET: uses current session context only.
    POST: accepts optional body {"payload": {...}} to enrich the suggestion.
    """
    payload: Dict[str, Any] = {}
    if request.method == "POST":
        try:
            body = await request.json()
            payload = body.get("payload", body) if isinstance(body, dict) else {}
        except Exception:
            pass
    session_ctx = _build_session_context(memory)
    if session_ctx:
        payload["session_context"] = session_ctx
    payload.setdefault("source", "suggest")
    payload.setdefault("event", "suggest")

    router = None
    if watcher_handler:
        router = watcher_handler._build_router()
    if not router:
        from app.agents.hypothesis_agent import HypothesisAgent
        from app.agents.experiment_agent import ExperimentAgent
        from app.agents.curve_fitting_agent import CurveFittingAgent
        from app.agents.analysis_agent import AnalysisAgent
        from app.agents.ml_models_agent import MLModelsAgent
        from app.agents.watcher_agent import WatcherAgent
        from app.agents.fallback_agent import FallbackAgent
        hyp = HypothesisAgent(name="Hypothesis Agent", desc="Background", question="")
        exp = ExperimentAgent(name="Experiment Agent", desc="Background", params_const={})
        cf = CurveFittingAgent(name="Curve Fitting", desc="Background")
        analysis = AnalysisAgent(name="Analysis Agent", desc="Background")
        ml_models = MLModelsAgent(name="ML Models", desc="Background")
        fb = FallbackAgent(name="Fallback Agent", desc="Handles failures")
        router = AgentRouter(
            agents=[WatcherAgent(), hyp, exp, cf, ml_models, analysis],
            fallback_agent=fb,
        )
    try:
        result = router.suggest_next_agent(payload, memory)
        return result
    except Exception as e:
        logger.error(f"Error suggesting agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/route")
async def route_event(request: RouteRequest, background_tasks: BackgroundTasks):
    """Route a filesystem event to the appropriate agent"""
    try:
        # Route in background
        background_tasks.add_task(_route_event_task, request.payload)
        
        return {
            "status": "routed",
            "payload": request.payload,
            "message": "Event routed to appropriate agent"
        }
    except Exception as e:
        logger.error(f"Error routing event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _route_event_task(payload: Dict[str, Any]):
    """Background task to route an event"""
    try:
        session_ctx = _build_session_context(memory)
        if session_ctx:
            payload = {**payload, "session_context": session_ctx}
        router = watcher_handler._build_router() if watcher_handler else None
        if not router:
            # Create a temporary router if watcher not initialized
            from app.agents.hypothesis_agent import HypothesisAgent
            from app.agents.experiment_agent import ExperimentAgent
            from app.agents.curve_fitting_agent import CurveFittingAgent
            from app.agents.analysis_agent import AnalysisAgent
            from app.agents.ml_models_agent import MLModelsAgent
            from app.agents.watcher_agent import WatcherAgent
            from app.agents.fallback_agent import FallbackAgent

            hyp = HypothesisAgent(name="Hypothesis Agent", desc="Background", question="")
            exp = ExperimentAgent(name="Experiment Agent", desc="Background", params_const={})
            cf = CurveFittingAgent(name="Curve Fitting", desc="Background")
            analysis = AnalysisAgent(name="Analysis Agent", desc="Background")
            ml_models = MLModelsAgent(name="ML Models", desc="Background")
            fb = FallbackAgent(name="Fallback Agent", desc="Handles failures")
            router = AgentRouter(
                agents=[WatcherAgent(), hyp, exp, cf, ml_models, analysis],
                fallback_agent=fb,
            )
        router.route(payload, memory)
    except Exception as e:
        logger.error(f"Error in routing task: {e}")

@app.post("/file-event")
async def handle_file_event(event: FileEventRequest, background_tasks: BackgroundTasks):
    """Handle a filesystem event (file created/modified)"""
    try:
        file_path = Path(event.file_path)
        
        # Log the event
        memory.log_event(
            "watcher",
            {
                "file": str(file_path),
                "event": event.event_type,
                "metadata": event.metadata or {}
            },
            mode="watcher",
        )
        
        # Build payload for routing
        payload: Dict[str, Any] = {
            "source": "filesystem",
            "trigger_file": str(file_path),
            "event": event.event_type,
            "metadata": event.metadata or {}
        }
        
        # Route in background
        background_tasks.add_task(_route_event_task, payload)
        
        return {
            "status": "processed",
            "file": str(file_path),
            "event": event.event_type,
            "message": "File event processed and routed"
        }
    except Exception as e:
        logger.error(f"Error handling file event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scan-directory")
async def scan_directory():
    """Manually scan the watch directory for CSV/Excel files and process them"""
    global watcher_handler, watch_dir
    
    if watcher_handler is None or watch_dir is None:
        raise HTTPException(status_code=400, detail="Watcher not initialized. Start watching first.")
    
    try:
        abs_watch_dir = watch_dir.resolve()
        logger.info(f"🔍 Scanning directory for files: {abs_watch_dir}")
        
        # Find all CSV/Excel files
        csv_files = []
        for ext in ['.csv', '.xlsx', '.xls']:
            csv_files.extend(abs_watch_dir.glob(f"*{ext}"))
            csv_files.extend(abs_watch_dir.glob(f"*{ext.upper()}"))
        
        logger.info(f"📊 Found {len(csv_files)} CSV/Excel files in directory")
        
        processed = []
        for file_path in csv_files:
            try:
                logger.info(f"🔄 Processing file: {file_path.name}")
                # Create a mock event
                from watchdog.events import FileCreatedEvent
                mock_event = FileCreatedEvent(str(file_path))
                watcher_handler.on_created(mock_event)
                processed.append(str(file_path))
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        return {
            "status": "completed",
            "directory": str(abs_watch_dir),
            "files_found": len(csv_files),
            "files_processed": len(processed),
            "processed_files": processed
        }
    except Exception as e:
        logger.error(f"Error scanning directory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/test-file-detection")
async def test_file_detection(file_path: Optional[str] = None):
    """Manually test file detection for a specific file or scan directory if no file specified"""
    global watcher_handler, watch_dir
    
    try:
        # If no file path provided, scan the watch directory
        if not file_path:
            if watcher_handler is None or watch_dir is None:
                raise HTTPException(status_code=400, detail="Watcher not initialized. Start watching first.")
            
            # Scan directory for files
            abs_watch_dir = watch_dir.resolve()
            logger.info(f"🔍 Scanning directory for files: {abs_watch_dir}")
            
            csv_files = []
            for ext in ['.csv', '.xlsx', '.xls']:
                csv_files.extend(abs_watch_dir.glob(f"*{ext}"))
                csv_files.extend(abs_watch_dir.glob(f"*{ext.upper()}"))
            
            logger.info(f"📊 Found {len(csv_files)} CSV/Excel files in directory")
            
            processed = []
            for test_path in csv_files[:10]:  # Process first 10 files to avoid overload
                try:
                    logger.info(f"🔄 Processing file: {test_path.name}")
                    from watchdog.events import FileCreatedEvent
                    mock_event = FileCreatedEvent(str(test_path))
                    watcher_handler.on_created(mock_event)
                    processed.append(str(test_path))
                except Exception as e:
                    logger.error(f"Error processing {test_path}: {e}")
            
            return {
                "status": "completed",
                "directory": str(abs_watch_dir),
                "files_found": len(csv_files),
                "files_processed": len(processed),
                "processed_files": processed[:10]  # Return first 10
            }
        
        # Test specific file
        test_path = Path(file_path)
        if not test_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        if not test_path.is_file():
            raise HTTPException(status_code=400, detail=f"Path is not a file: {file_path}")
        
        logger.info(f"🧪 Testing file detection for: {test_path}")
        
        # Create a mock event to test detection
        from watchdog.events import FileCreatedEvent
        mock_event = FileCreatedEvent(str(test_path))
        
        # Get the watcher handler
        if watcher_handler is None:
            raise HTTPException(status_code=400, detail="Watcher handler not initialized. Start watching first.")
        
        # Manually trigger the handler
        watcher_handler.on_created(mock_event)
        
        return {
            "status": "processed",
            "file": str(test_path),
            "message": "File detection test completed. Check logs for details."
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in test file detection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))




if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("WATCHER_PORT", 8000))
    host = os.environ.get("WATCHER_HOST", "0.0.0.0")
    
    logger.info(f"Starting POLARIS Watcher Server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
