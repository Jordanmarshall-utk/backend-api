from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from langchain_core.runnables import RunnableConfig

from app.graph.state import PolarisGraphState
from app.services.analysis_service import run_analysis_pipeline
from app.services.memory_service import get_memory_manager
from app.services.reasoning_integration import reasoning_context_for_stage
from app.tools.memory_adapter import MemoryAdapter

_logger = logging.getLogger(__name__)


async def _gather_context_node(state: PolarisGraphState, config: RunnableConfig) -> PolarisGraphState:
    memory = get_memory_manager()
    return MemoryAdapter.write(memory, state, current_agent="analysis", stage="analysis")


async def _run_analysis_node(state: PolarisGraphState, config: RunnableConfig) -> PolarisGraphState:
    memory = get_memory_manager()
    try:
        result = run_analysis_pipeline(
            memory,
            {
                "action": "run",
                "reasoning_context": reasoning_context_for_stage(memory, "analysis"),
            },
        )
        analysis = result.get("analysis_report") if isinstance(result, dict) else None
    except Exception as exc:
        _logger.warning("run_analysis_pipeline failed: %s", exc)
        analysis = None
    return MemoryAdapter.write(memory, state, analysis_results=analysis if analysis is not None else {})


async def _evaluate_outcome_node(state: PolarisGraphState, config: RunnableConfig) -> PolarisGraphState:
    memory = get_memory_manager()
    analysis_text = str(memory.get_var("analysis_results") or "")
    verdict = "complete"
    lower = analysis_text.lower()
    if any(kw in lower for kw in ("inconclusive", "insufficient", "retry", "repeat experiment", "more data")):
        verdict = "needs_more_experiments"
    memory.set_var("analysis_verdict", verdict)
    return {**state}


async def _decide_next_step_node(state: PolarisGraphState, config: RunnableConfig) -> PolarisGraphState:
    memory = get_memory_manager()
    verdict = memory.get_var("analysis_verdict") or "complete"
    return MemoryAdapter.write(
        memory, state,
        stage=verdict,
        interrupt_payload={
            "stage": "analysis",
            "verdict": verdict,
            "hint": "Analysis complete. Review before finalizing.",
        },
    )


async def _analysis_checkpoint_node(state: PolarisGraphState, config: RunnableConfig) -> PolarisGraphState:
    payload = state.get("interrupt_payload") or {}
    interrupt(payload)  # pause — frontend reviews; resumes via POST /pipeline/resume
    return {**state, "interrupt_payload": None}


def _route_after_decide(state: PolarisGraphState) -> str:
    """Conditional edge: route to experiment loop or END based on analysis verdict."""
    memory = get_memory_manager()
    verdict = memory.get_var("analysis_verdict") or "complete"
    return verdict  # "needs_more_experiments" | "complete"


def analysis_subgraph() -> CompiledStateGraph:
    g = StateGraph(PolarisGraphState)
    g.add_node("gather_context", _gather_context_node)
    g.add_node("run_analysis", _run_analysis_node)
    g.add_node("evaluate_outcome", _evaluate_outcome_node)
    g.add_node("decide_next_step", _decide_next_step_node)
    g.add_node("analysis_checkpoint", _analysis_checkpoint_node)
    g.set_entry_point("gather_context")
    g.add_edge("gather_context", "run_analysis")
    g.add_edge("run_analysis", "evaluate_outcome")
    g.add_edge("evaluate_outcome", "decide_next_step")
    g.add_edge("decide_next_step", "analysis_checkpoint")
    g.add_conditional_edges(
        "analysis_checkpoint",
        _route_after_decide,
        {"needs_more_experiments": END, "complete": END},
    )
    return g.compile()
