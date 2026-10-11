import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.graph.state import PolarisGraphState


def _state(**kwargs) -> PolarisGraphState:
    base: PolarisGraphState = {
        "stage": "initial",
        "has_hypothesis": False,
        "has_experimental_outputs": False,
        "has_curve_results": False,
        "has_ml_results": False,
        "has_analysis_results": False,
        "hypothesis_ready": False,
        "hypothesis_preview": None,
        "research_goal": "Test question",
        "experiment_id": None,
        "current_agent": None,
        "error": None,
        "interrupt_payload": None,
        "routing_mode": "autonomous",
        "manual_workflow": [],
        "workflow_index": 0,
    }
    return {**base, **kwargs}


@pytest.mark.asyncio
async def test_fallback_node_sets_stage_error():
    from app.graph.nodes.fallback import fallback_node
    memory = MagicMock()
    with patch("app.graph.nodes.fallback.get_memory_manager", return_value=memory):
        result = await fallback_node(_state(error="something broke"), {})
    assert result["stage"] == "error"
    assert result["current_agent"] == "fallback"
    memory.log_event.assert_called_once()


@pytest.mark.asyncio
async def test_watcher_node_sets_current_agent():
    from app.graph.nodes.watcher import watcher_node
    memory = MagicMock()
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value="curve_fitting_agent")
    with patch("app.graph.nodes.watcher.get_memory_manager", return_value=memory), \
         patch("app.graph.nodes.watcher.watcher_routing_chain", mock_chain):
        result = await watcher_node(_state(), {})
    assert result["current_agent"] == "curve_fitting_agent"


@pytest.mark.asyncio
async def test_ml_models_node_sets_has_ml_results():
    from app.graph.nodes.ml_models import ml_models_node
    memory = MagicMock()
    memory.get_var.return_value = {"gp_model": "fitted"}
    mock_run = MagicMock(
        return_value={"status": "success", "result": {"gp_model": "fitted"}}
    )
    with patch("app.graph.nodes.ml_models.get_memory_manager", return_value=memory), \
         patch("app.graph.nodes.ml_models.run_ml_automation", mock_run):
        result = await ml_models_node(_state(), {})
    assert result["has_ml_results"] is True
