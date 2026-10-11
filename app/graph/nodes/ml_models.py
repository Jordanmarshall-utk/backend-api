from __future__ import annotations

import asyncio

from langchain_core.runnables import RunnableConfig

from app.graph.state import PolarisGraphState
from app.services.memory_service import get_memory_manager
from app.services.ml_session_service import run_ml_automation
from app.services.reasoning_integration import reasoning_context_for_stage
from app.tools.memory_adapter import MemoryAdapter


async def ml_models_node(state: PolarisGraphState, config: RunnableConfig) -> PolarisGraphState:
    memory = get_memory_manager()
    payload = {
        "action": "run",
        "reasoning_context": reasoning_context_for_stage(memory, "ml_models"),
    }
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, run_ml_automation, memory, payload)
    gp_results = memory.get_var("gp_results") or result.get("result")
    return MemoryAdapter.write(
        memory, state,
        gp_results=gp_results,
        current_agent="ml_models",
        stage="ml_models",
    )
