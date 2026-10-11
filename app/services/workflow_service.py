"""Workflow builder/runner session — parity with Streamlit pages/workflow.py."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.services.workflow_followups import (
    WORKFLOW_STEP_PAGES,
    sync_workflow_from_steps,
)
from app.tools.paths import get_results_dir
from app.utils.json_safe import to_jsonable

AVAILABLE_STEPS: Dict[str, Dict[str, Any]] = {
    "Hypothesis Agent": {
        "name": "Hypothesis Agent",
        "description": "Generate and refine research hypotheses",
        "can_auto": False,
        "page": "/agents/hypothesis",
    },
    "Experiment Agent": {
        "name": "Experiment Agent",
        "description": "Design experiments and generate protocols",
        "can_auto": False,
        "page": "/agents/experiment",
    },
    "Curve Fitting": {
        "name": "Curve Fitting",
        "description": "Fit curves to spectral data",
        "can_auto": True,
        "page": "/agents/curve-fitting",
    },
    "ML Models": {
        "name": "ML Models",
        "description": "Run ML models for optimization",
        "can_auto": True,
        "page": "/agents/ml-models",
    },
    "Analysis Agent": {
        "name": "Analysis Agent",
        "description": "Analyze results and provide recommendations",
        "can_auto": True,
        "page": "/agents/analysis",
    },
}

ML_MODEL_OPTIONS = [
    "Single-objective GP (scikit-learn)",
    "Dual-objective GP (PyTorch)",
    "Monte Carlo Decision Tree (external)",
]


def _step_catalog() -> List[Dict[str, Any]]:
    return [
        {
            "name": info["name"],
            "description": info["description"],
            "can_auto": info["can_auto"],
            "page": info.get("page"),
        }
        for info in AVAILABLE_STEPS.values()
    ]


def get_workflow_session(memory: Any) -> Dict[str, Any]:
    workflows = memory._db.get_workflows()

    current = memory.get_var("current_workflow") or {"name": "My Workflow", "steps": []}
    steps = memory.get_var("workflow_steps") or current.get("steps") or []
    manual = memory.get_var("manual_workflow") or []
    auto_flags = memory.get_var("workflow_auto_flags") or {}

    step = memory.get_var("workflow_step") or "idle"
    return {
        "available_steps": _step_catalog(),
        "saved_workflows": [
            {
                "name": name,
                "steps": wf.get("steps", []),
                "ml_model_choice": wf.get("ml_model_choice"),
                "created_at": wf.get("created_at"),
            }
            for name, wf in workflows.items()
        ],
        "current_workflow_name": current.get("name") or memory.get_var("current_workflow_name"),
        "workflow_steps": steps,
        "manual_workflow": manual,
        "workflow_auto_flags": auto_flags,
        "workflow_ml_model_choice": memory.get_var("workflow_ml_model_choice")
        or memory.get_var("optimization_model_choice"),
        "ml_model_options": ML_MODEL_OPTIONS,
        "routing_mode": memory.get_var("routing_mode") or "Autonomous (LLM)",
        "workflow_active": bool(memory.get_var("workflow_active")),
        "workflow_step": step,
        "workflow_index": int(memory.get_var("workflow_index") or 0),
        "next_page": WORKFLOW_STEP_PAGES.get(str(step)) if step != "idle" else None,
        "demo_workflow_running": bool(memory.get_var("demo_workflow_running")),
        "auto_ml_after_curve_fitting": bool(memory.get_var("auto_ml_after_curve_fitting")),
        "auto_route_to_analysis": bool(memory.get_var("auto_route_to_analysis")),
    }


def patch_workflow_session(
    memory: Any,
    *,
    workflow_name: Optional[str] = None,
    workflow_steps: Optional[List[Dict[str, Any]]] = None,
    routing_mode: Optional[str] = None,
    workflow_ml_model_choice: Optional[str] = None,
    workflow_index: Optional[int] = None,
    workflow_step: Optional[str] = None,
    auto_ml_after_curve_fitting: Optional[bool] = None,
    auto_route_to_analysis: Optional[bool] = None,
) -> Dict[str, Any]:
    if workflow_name is not None:
        cw = memory.get_var("current_workflow") or {}
        cw["name"] = workflow_name
        memory.set_var("current_workflow", cw)
        memory.set_var("current_workflow_name", workflow_name)

    if workflow_steps is not None:
        memory.set_var("workflow_steps", workflow_steps)
        cw = memory.get_var("current_workflow") or {}
        cw["steps"] = workflow_steps
        memory.set_var("current_workflow", cw)
        sync_workflow_from_steps(memory, workflow_steps)

    if routing_mode is not None:
        memory.set_var("routing_mode", routing_mode)

    if workflow_ml_model_choice is not None:
        memory.set_var("workflow_ml_model_choice", workflow_ml_model_choice)
        memory.set_var("optimization_model_choice", workflow_ml_model_choice)

    if workflow_index is not None:
        memory.set_var("workflow_index", workflow_index)

    if workflow_step is not None:
        memory.set_var("workflow_step", workflow_step)

    if auto_ml_after_curve_fitting is not None:
        memory.set_var("auto_ml_after_curve_fitting", auto_ml_after_curve_fitting)

    if auto_route_to_analysis is not None:
        memory.set_var("auto_route_to_analysis", auto_route_to_analysis)

    return get_workflow_session(memory)


def save_named_workflow(memory: Any, name: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    sync_workflow_from_steps(memory, steps)
    ml_choice = None
    if any(s.get("name") == "ML Models" for s in steps):
        ml_choice = memory.get_var("workflow_ml_model_choice") or memory.get_var("optimization_model_choice")
    memory.save_workflow(name, steps, ml_choice)
    memory.set_var("workflow_created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    cw = memory.get_var("current_workflow") or {}
    cw["name"] = name
    cw["steps"] = steps
    memory.set_var("current_workflow", cw)
    memory.set_var("current_workflow_name", name)
    return {"status": "success", "message": f"Workflow '{name}' saved.", "name": name}


def apply_named_workflow(memory: Any, name: str) -> Dict[str, Any]:
    workflow = memory.load_workflow(name)
    if not workflow:
        return {"status": "error", "message": f"Workflow '{name}' not found."}

    step_names = [step["name"] for step in workflow.get("steps", [])]
    memory.set_var("manual_workflow", step_names)
    auto_flags = {
        step["name"]: True for step in workflow.get("steps", []) if step.get("automatic")
    }
    memory.set_var("workflow_auto_flags", auto_flags)
    memory.set_var("current_workflow_name", name)
    memory.set_var("current_workflow", workflow)
    memory.set_var("workflow_steps", workflow.get("steps", []))
    sync_workflow_from_steps(memory, workflow.get("steps", []))

    ml_model_choice = workflow.get("ml_model_choice")
    if ml_model_choice:
        memory.set_var("workflow_ml_model_choice", ml_model_choice)
        memory.set_var("optimization_model_choice", ml_model_choice)

    return {"status": "success", "message": f"Workflow '{name}' applied.", "manual_workflow": step_names}


def load_named_workflow(memory: Any, name: str) -> Dict[str, Any]:
    workflow = memory.load_workflow(name)
    if not workflow:
        return {"status": "error", "message": f"Workflow '{name}' not found."}
    memory.set_var("current_workflow", workflow)
    memory.set_var("workflow_steps", workflow.get("steps", []))
    if workflow.get("ml_model_choice"):
        memory.set_var("workflow_ml_model_choice", workflow["ml_model_choice"])
    return {"status": "success", "message": f"Loaded workflow '{name}'.", **get_workflow_session(memory)}


def delete_named_workflow(memory: Any, name: str) -> Dict[str, Any]:
    memory.delete_workflow(name)
    if memory.get_var("current_workflow_name") == name:
        memory.set_var("current_workflow", {"name": "Default Workflow", "steps": []})
        memory.set_var("workflow_steps", [])
    return {"status": "success", "message": f"Deleted workflow '{name}'."}


def start_workflow(memory: Any) -> Dict[str, Any]:
    steps = memory.get_var("workflow_steps") or []
    if steps and not memory.get_var("manual_workflow"):
        sync_workflow_from_steps(memory, steps)
    memory.set_var("workflow_active", True)
    memory.set_var("workflow_step", "hypothesis")
    memory.set_var("workflow_index", 0)
    memory.set_var("workflow_completed", False)
    memory.set_var("workflow_experiment_started", False)
    memory.set_var("workflow_experiment_completed", False)
    memory.set_var("workflow_experiment_outputs", None)
    memory.set_var("workflow_curve_fitting_completed", False)
    memory.set_var("experimental_outputs", None)
    memory.set_var("hypothesis_ready", False)
    memory.set_var("stop_hypothesis", False)
    memory.set_var("stage", "initial")
    return {
        "status": "success",
        "message": "Workflow started.",
        "next_page": WORKFLOW_STEP_PAGES["hypothesis"],
    }


def stop_workflow(memory: Any) -> Dict[str, Any]:
    memory.set_var("workflow_active", False)
    memory.set_var("workflow_step", "idle")
    return {"status": "success", "message": "Workflow stopped."}


def run_demo_workflow(memory: Any, *, auto_fit: bool = True) -> Dict[str, Any]:
    from app.tools.demo_data_generator import generate_demo_dataset, generate_demo_worklist

    results_dir = Path(get_results_dir())
    demo_dir = results_dir / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    output_dir = str(demo_dir)

    spectral_path, comp_path = generate_demo_dataset(n_wells=15, output_dir=output_dir)
    wells_demo = [
        "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "B1", "B2", "B3",
    ]
    memory.set_var("selected_wells", set(wells_demo))

    demo_hypothesis = (
        "Increasing PEA-rich, lower-dimensional character should strengthen phase-resolved emission "
        "near 619, 574, and 522 nm relative to the long-wavelength FAPbI3 feature near 815 nm. "
        "The dominant peak may switch between coexisting phases rather than shift linearly."
    )
    demo_clarified = (
        "How does the PEA2PbI4/FAPbI3 composition ratio affect phase-resolved "
        "photoluminescence emission?"
    )
    demo_socratic = (
        "Which bands represent coexisting lower-dimensional and 3D phases, and where does the "
        "dominant emitting phase change? Which replicate and processing controls test that result?"
    )
    memory.insert_interaction("assistant", demo_hypothesis, "hypothesis", "hypothesis")
    memory.insert_interaction("user", demo_clarified, "clarified_question", "hypothesis")
    memory.insert_interaction("assistant", demo_socratic, "socratic_pass", "hypothesis")
    memory.set_var("hypothesis_ready", True)
    memory.set_var("last_hypothesis", demo_hypothesis)

    max_vol = (
        memory.get_var("experimental_constraints", {})
        .get("liquid_handling", {})
        .get("max_volume_per_mixture", 50)
    )
    worklist_csv = generate_demo_worklist(comp_path, max_vol)
    demo_plan = (
        "Screen 15 PEA2PbI4/FAPbI3 compositions across A1-B3 with replicate and process controls. "
        "Collect PL spectra from 450-900 nm and compare fitted phase intensities across composition."
    )
    demo_outputs = {
        "plan": demo_plan,
        "worklist": worklist_csv,
        "layout": "A1-A12, B1-B3 (96-well plate)",
        "protocol": "Phase-resolved PL measurement protocol with composition-series controls.",
    }
    memory.set_var("experimental_outputs", demo_outputs)
    memory.set_var("workflow_experiment_outputs", demo_outputs)
    memory.insert_interaction(
        "assistant",
        "Generated a constrained 15-well composition series, plate layout, and phase-resolved PL protocol.",
        "experimental_plan",
        "experiment",
    )

    memory.set_var("demo_workflow_running", True)
    memory.set_var("auto_run_curve_fitting", True)
    memory.set_var("auto_run_data_file", spectral_path)
    memory.set_var("auto_run_comp_file", comp_path)
    memory.set_var(
        "auto_run_params",
        {
            "wells_to_analyze": wells_demo,
            "reads_to_analyze": "1",
            "read_type": "em_spectrum",
            "max_peaks": 4,
            "r2_target": 0.90,
            "max_attempts": 3,
            "api_delay_seconds": 0.5,
        },
    )
    memory.set_var(
        "manual_workflow",
        [
            "Hypothesis Agent",
            "Experiment Agent",
            "Curve Fitting",
            "ML Models",
            "Analysis Agent",
            "Experiment Agent",
        ],
    )
    memory.set_var(
        "workflow_auto_flags",
        {"Curve Fitting": False, "ML Models": False, "Analysis Agent": False},
    )
    memory.set_var("workflow_ml_model_choice", ML_MODEL_OPTIONS[0])
    memory.set_var("optimization_model_choice", ML_MODEL_OPTIONS[0])
    memory.set_var(
        "ml_model_config",
        {"target": "peak_1_wavelength", "beta": 2.0, "n_candidates": 20},
    )
    memory.set_var("auto_ml_after_curve_fitting", True)
    memory.set_var("auto_route_to_analysis", False)
    memory.set_var(
        "research_goal",
        "Optimize PL peak wavelength for target emission. "
        "Synthetic data has peaks varying 450–850 nm with composition.",
    )
    memory.set_var("workflow_active", True)
    memory.set_var("workflow_step", "curve_fitting")
    memory.set_var("ml_auto_composition_path", comp_path)
    memory.set_var("curve_fitting_last_composition_file", comp_path)
    sync_workflow_from_steps(
        memory,
        [
            {"name": "Hypothesis Agent", "automatic": False},
            {"name": "Experiment Agent", "automatic": False},
            {"name": "Curve Fitting", "automatic": False},
            {"name": "ML Models", "automatic": True},
            {"name": "Analysis Agent", "automatic": False},
        ],
    )

    out: Dict[str, Any] = {
        "status": "success",
        "message": "Demo workflow primed. Continue on Curve Fitting.",
        "spectral_path": spectral_path,
        "composition_path": comp_path,
        "next_page": WORKFLOW_STEP_PAGES["curve_fitting"],
    }

    if auto_fit and memory.get_var("auto_run_curve_fitting"):
        from app.services.curve_fitting_runner import run_curve_fitting_for_api
        from app.services.workflow_followups import (
            build_auto_curve_fitting_payload,
            consume_auto_curve_fitting_flag,
        )

        payload = build_auto_curve_fitting_payload(memory)
        if payload:
            consume_auto_curve_fitting_flag(memory)
            cf_result = run_curve_fitting_for_api(memory, payload)
            out["curve_fitting"] = cf_result
            if cf_result.get("status") == "success":
                from app.services.workflow_followups import on_curve_fitting_complete

                follow = on_curve_fitting_complete(memory)
                out.update(follow)
                out["message"] = (
                    "Demo workflow complete: curve fitting finished."
                    + (
                        " ML automation ran."
                        if follow.get("ml_automation")
                        else " Configure ML on the ML Models page."
                    )
                )
                if follow.get("next_page"):
                    out["next_page"] = follow["next_page"]
            else:
                out["message"] = (
                    f"Demo data ready but curve fitting failed: {cf_result.get('message', 'unknown error')}. "
                    "Open Curve Fitting to retry."
                )

    return to_jsonable(out)


def export_workflow_json(memory: Any) -> str:
    session = get_workflow_session(memory)
    payload = {
        "name": session.get("current_workflow_name") or "My Workflow",
        "steps": session.get("workflow_steps") or [],
        "ml_model_choice": session.get("workflow_ml_model_choice"),
    }
    return json.dumps(payload, indent=2)
