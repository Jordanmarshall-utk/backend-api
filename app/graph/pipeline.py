from __future__ import annotations

import logging
from typing import Optional

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

# INTERRUPT_NODES documents which sub-graph nodes pause — sub-graphs call interrupt() internally
# from app.graph.interrupts import INTERRUPT_NODES
from app.graph.nodes.analysis import analysis_subgraph
from app.graph.nodes.curve_fitting import curve_fitting_subgraph
from app.graph.nodes.experiment import experiment_subgraph
from app.graph.nodes.fallback import fallback_node
from app.graph.nodes.hypothesis import hypothesis_subgraph
from app.graph.nodes.ml_models import ml_models_node
from app.graph.nodes.watcher import watcher_node
from app.graph.state import PolarisGraphState
from app.services.memory_service import get_memory_manager
from app.services.reasoning_integration import plan_pipeline_reasoning

_logger = logging.getLogger(__name__)
_pipeline = None


def _stage_reasoning_node(domain_agent: str, stage: str, objective: str = ""):
    def node(state: PolarisGraphState) -> PolarisGraphState:
        memory = get_memory_manager()
        payload = {
            "action": "run",
            "research_goal": state.get("research_goal") or memory.get_var("research_goal") or "",
        }
        if objective:
            payload["objective"] = objective
        try:
            plan_pipeline_reasoning(
                memory,
                domain_agent,
                stage,
                payload=payload,
                objective=objective,
            )
        except Exception as exc:
            _logger.warning("Reasoning gate failed for %s: %s", stage, exc)
        return {**state}

    return node


def _route_after_analysis(state: PolarisGraphState) -> str:
    memory = get_memory_manager()
    verdict = memory.get_var("analysis_verdict") or "complete"
    if verdict == "needs_more_experiments":
        return "experiment"
    return END


def build_pipeline(checkpointer: Optional[BaseCheckpointSaver] = None) -> CompiledStateGraph:
    """Build and compile the top-level POLARIS pipeline StateGraph."""
    g = StateGraph(PolarisGraphState)

    # Embed compiled sub-graphs as single nodes
    g.add_node("hypothesis", hypothesis_subgraph())
    g.add_node("reason_experiment", _stage_reasoning_node("Experiment Agent", "experiment"))
    g.add_node("experiment", experiment_subgraph())
    g.add_node(
        "reason_curve_fitting",
        _stage_reasoning_node(
            "Curve Fitting Agent",
            "curve_fitting",
            "Calculate curve fits, preserve raw measurements, and export diagnostics.",
        ),
    )
    g.add_node("curve_fitting", curve_fitting_subgraph())
    g.add_node(
        "reason_ml_models",
        _stage_reasoning_node(
            "ML Models Agent",
            "ml_models",
            "Select and run predictive models while preserving validation and uncertainty constraints.",
        ),
    )
    g.add_node("ml_models", ml_models_node)
    g.add_node(
        "reason_analysis",
        _stage_reasoning_node(
            "Analysis Agent",
            "analysis",
            "Evaluate whether results support the hypothesis, separate observation from inference, and decide the next justified action.",
        ),
    )
    g.add_node("analysis", analysis_subgraph())
    g.add_node("fallback", fallback_node)
    g.add_node("watcher", watcher_node)

    g.set_entry_point("hypothesis")
    g.add_edge("hypothesis", "reason_experiment")
    g.add_edge("reason_experiment", "experiment")
    g.add_edge("experiment", "reason_curve_fitting")
    g.add_edge("reason_curve_fitting", "curve_fitting")
    g.add_edge("curve_fitting", "reason_ml_models")
    g.add_edge("reason_ml_models", "ml_models")
    g.add_edge("ml_models", "reason_analysis")
    g.add_edge("reason_analysis", "analysis")
    g.add_conditional_edges(
        "analysis",
        _route_after_analysis,
        {"experiment": "reason_experiment", END: END},
    )
    g.add_edge("fallback", END)
    g.add_edge("watcher", END)

    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
    # INTERRUPT_NODES are sub-graph internal node names; sub-graphs use interrupt() calls
    # internally so they pause regardless. We compile without top-level interrupt_before to
    # avoid validation errors since those node names don't exist at the top-level graph scope.
    return g.compile(checkpointer=checkpointer)


def init_pipeline(checkpointer: Optional[BaseCheckpointSaver] = None) -> CompiledStateGraph:
    """Build the pipeline singleton with the given checkpointer.

    Call this once from the FastAPI lifespan after init_checkpointer() has run.
    """
    global _pipeline
    _pipeline = build_pipeline(checkpointer=checkpointer)
    return _pipeline


def get_pipeline() -> CompiledStateGraph:
    """Singleton accessor for the production pipeline.

    Raises RuntimeError if called before init_pipeline().
    Falls back to building with InMemorySaver if checkpointer is not available
    (for test contexts that call get_pipeline() directly).
    """
    global _pipeline
    if _pipeline is None:
        from langgraph.checkpoint.memory import MemorySaver
        _logger.warning("get_pipeline() called before init_pipeline(); using MemorySaver fallback")
        _pipeline = build_pipeline(checkpointer=MemorySaver())
    return _pipeline
