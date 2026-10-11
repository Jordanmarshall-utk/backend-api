from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig

from app.graph.state import PolarisGraphState
from app.services.experiment_service import run_experiment_pipeline
from app.services.memory_service import get_memory_manager
from app.services.reasoning_integration import reasoning_context_for_stage
from app.tools.memory_adapter import MemoryAdapter


async def _build_context_node(state: PolarisGraphState, config: RunnableConfig) -> PolarisGraphState:
    memory = get_memory_manager()
    return MemoryAdapter.write(
        memory, state,
        current_agent="experiment",
        stage="experiment",
        research_goal=state.get("research_goal") or memory.get_var("research_goal") or "",
    )


async def _generate_plan_node(state: PolarisGraphState, config: RunnableConfig) -> PolarisGraphState:
    memory = get_memory_manager()
    return MemoryAdapter.write(
        memory,
        state,
        experimental_plan=memory.get_var("experimental_plan"),
    )


async def _generate_worklist_node(state: PolarisGraphState, config: RunnableConfig) -> PolarisGraphState:
    memory = get_memory_manager()
    # Delegate full pipeline execution to the existing service
    result = run_experiment_pipeline(
        memory,
        {
            "action": "run",
            "reasoning_context": reasoning_context_for_stage(memory, "experiment"),
        },
    )
    outputs = result if isinstance(result, dict) else {}
    return MemoryAdapter.write(
        memory, state,
        experimental_outputs=outputs or True,
        stage="experiment",
    )


def experiment_subgraph() -> CompiledStateGraph:
    """Build and compile the 3-node experiment sub-graph."""
    g = StateGraph(PolarisGraphState)
    g.add_node("build_context", _build_context_node)
    g.add_node("generate_plan", _generate_plan_node)
    g.add_node("generate_worklist", _generate_worklist_node)
    g.set_entry_point("build_context")
    g.add_edge("build_context", "generate_plan")
    g.add_edge("generate_plan", "generate_worklist")
    g.add_edge("generate_worklist", END)
    return g.compile()
