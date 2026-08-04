from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    version: str = "1.3.0"


class SessionContextModel(BaseModel):
    experiment_id: Optional[int] = None
    stage: Optional[str] = None
    routing_mode: Optional[str] = None
    has_hypothesis: bool = False
    has_experimental_outputs: bool = False
    has_curve_fitting_results: bool = False
    has_analysis_results: bool = False
    has_gp_results: bool = False


class Experiment(BaseModel):
    id: int
    name: str
    stage: str = "initial"
    created_at: datetime
    updated_at: datetime


class ExperimentListResponse(BaseModel):
    items: List[Experiment]


class DashboardSummary(BaseModel):
    experiment_id: Optional[int] = None
    stage: Optional[str] = None
    active_workflow: bool = False
    last_hypothesis_preview: Optional[str] = None
    agent_counts: Dict[str, int] = Field(default_factory=dict)


class DashboardMetrics(BaseModel):
    cpu_percent: Optional[float] = None
    memory_percent: Optional[float] = None
    disk_percent: Optional[float] = None
    uptime_seconds: Optional[float] = None
    total_events: int = 0
    cpu_delta: Optional[float] = None
    memory_delta: Optional[float] = None
    uptime_display: Optional[str] = None
    uptime_delta_seconds: Optional[int] = None
    events_delta: Optional[int] = None


class DashboardDetailResponse(BaseModel):
    system_performance: Dict[str, Any] = Field(default_factory=dict)
    agent_usage: Dict[str, Any] = Field(default_factory=dict)
    watcher: Dict[str, Any] = Field(default_factory=dict)
    uploaded_files: List[Dict[str, Any]] = Field(default_factory=list)
    additional_analytics: Dict[str, Any] = Field(default_factory=dict)
    workflow: Dict[str, Any] = Field(default_factory=dict)
    session_statistics: Dict[str, Any] = Field(default_factory=dict)
    ml_analysis_activity: Dict[str, Any] = Field(default_factory=dict)
    stage: Optional[str] = None
    last_hypothesis_preview: Optional[str] = None


class HypothesisChatRequest(BaseModel):
    action: Literal["submit_question", "choose_option", "generate_hypothesis", "reset"]
    question: Optional[str] = None
    choice: Optional[str] = None
    experiment_id: Optional[int] = None
    reasoning_policy: Optional[str] = "auto"


class HypothesisChatBubble(BaseModel):
    role: Literal["user", "assistant"] = "assistant"
    title: Optional[str] = None
    content: str = ""


class AgentDocumentMeta(BaseModel):
    document_id: Optional[str] = None
    title: Optional[str] = None
    markdown: Optional[str] = None
    pdf_url: Optional[str] = None
    markdown_url: Optional[str] = None


class HypothesisChatResponse(BaseModel):
    stage: str
    messages: List[HypothesisChatBubble] = Field(default_factory=list)
    assistant_message: str = ""
    options: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    document_id: Optional[str] = None
    document_markdown: Optional[str] = None
    pdf_url: Optional[str] = None
    reasoning_metadata: Optional[Dict[str, Any]] = None


class WorkflowRunRequest(BaseModel):
    experiment_id: int
    mode: Literal["autonomous", "manual"] = "autonomous"
    steps: List[str] = Field(default_factory=list)


class WorkflowRunResponse(BaseModel):
    workflow_id: UUID
    status: Literal["queued", "running", "completed", "failed"] = "queued"


class WorkflowStatus(BaseModel):
    workflow_id: UUID
    status: str
    current_step: Optional[str] = None
    error: Optional[str] = None


class ExperimentSessionPatch(BaseModel):
    experiment_id: Optional[int] = None
    experimental_constraints: Optional[Dict[str, Any]] = None
    manual_inputs: Optional[Dict[str, Any]] = None


class AgentRunRequest(BaseModel):
    experiment_id: Optional[int] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    session_context: Optional[SessionContextModel] = None


class AgentRunResponse(BaseModel):
    agent: str
    status: Literal["success", "error", "skipped"]
    message: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    next_agent: Optional[str] = Field(default=None, alias="nextAgent")

    model_config = {"populate_by_name": True}


class WatcherStartRequest(BaseModel):
    directory: Optional[str] = None
    results_dir: Optional[str] = None
    port: int = 8765
    experiment_id: Optional[int] = None


class WatcherStatus(BaseModel):
    running: bool
    directory: Optional[str] = None
    port: Optional[int] = None
    pid: Optional[int] = None
    message: Optional[str] = None


class McpOrchestrateRequest(BaseModel):
    query: str
    experiment_id: Optional[int] = None
    require_hypothesis_gate: bool = True


class McpOrchestrateResponse(BaseModel):
    status: str = "ok"
    hypothesis_gate: Dict[str, Any] = Field(default_factory=dict)
    literature: Dict[str, Any] = Field(default_factory=dict)
    trace_id: str = Field(default_factory=lambda: str(uuid4()))


class JupyterConfigModel(BaseModel):
    server_url: Optional[str] = None
    token: Optional[str] = None
    upload_enabled: Optional[bool] = None
    notebook_path: Optional[str] = None


class AppSettings(BaseModel):
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    qwen_base_url: Optional[str] = None
    routing_mode: Optional[str] = None
    max_hypothesis_rounds: Optional[int] = None
    watcher_directory: Optional[str] = None
    watcher_results_dir: Optional[str] = None
    watcher_enabled: Optional[bool] = None
    experimental_mode: Optional[bool] = None
    jupyter_config: Optional[JupyterConfigModel] = None


class AppSettingsPatch(AppSettings):
    api_key: Optional[str] = Field(default=None, description="Provider API key (write-only)")


class AppSettingsResponse(AppSettings):
    api_key_configured: bool = False


class HistoryEntry(BaseModel):
    id: str
    timestamp: datetime
    event_type: str
    agent: Optional[str] = None
    component: Optional[str] = None
    role: Optional[str] = None
    summary: Optional[str] = None
    experiment_id: Optional[int] = None


class HistoryListResponse(BaseModel):
    items: List[HistoryEntry]
