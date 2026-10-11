"""Headless API execution for non-hypothesis agents (no Streamlit)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.services.experiment_service import (
    generate_gp_worklist,
    get_experiment_session,
    patch_experiment_session,
    run_experiment_pipeline,
    upload_experiment_artifact_to_jupyter,
)


def _hypothesis_preview(memory: Any) -> Optional[str]:
    hyp = memory.view_component("hypothesis") or memory.get_var("last_hypothesis")
    if hyp and str(hyp).strip():
        text = str(hyp).strip()
        return text[:500] + ("…" if len(text) > 500 else "")
    return None


def run_hypothesis_agent(memory: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    stage = str(memory.get_var("stage") or "initial")
    last_doc = memory.get_var("last_document") or memory.get_var("document_hypothesis")
    out: Dict[str, Any] = {
        "ready": True,
        "stage": stage,
        "message": "Use POST /api/v1/agents/hypothesis/chat or /chat/stream for the interactive flow.",
        "hypothesis_preview": _hypothesis_preview(memory),
        "use_chat_endpoint": True,
        "payload": payload,
    }
    if isinstance(last_doc, dict) and last_doc.get("document_id"):
        out.update(
            {
                "document_id": last_doc.get("document_id"),
                "document_markdown": last_doc.get("markdown"),
                "pdf_url": last_doc.get("pdf_url"),
            }
        )
    return out


def run_experiment_agent(memory: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    action = (payload.get("action") or "").strip().lower()

    if action == "save_session":
        patched = patch_experiment_session(
            memory,
            experimental_constraints=payload.get("experimental_constraints"),
            manual_inputs=payload.get("manual_inputs"),
        )
        return {
            "status": "success",
            "message": "Experiment session saved.",
            "ready": patched["readiness"]["ready_to_run"],
            **patched,
        }

    if action == "generate_gp_worklist":
        return {
            "ready": True,
            **generate_gp_worklist(
                memory,
                upload_to_jupyter=bool(payload.get("upload_to_jupyter")),
            ),
        }

    if action == "upload_jupyter":
        artifact = str(payload.get("artifact") or "worklist").strip().lower()
        return {"ready": True, **upload_experiment_artifact_to_jupyter(memory, artifact=artifact)}

    if action in ("generate_plan", "run", "execute"):
        if payload.get("experimental_constraints") is not None or payload.get("manual_inputs") is not None:
            patch_experiment_session(
                memory,
                experimental_constraints=payload.get("experimental_constraints"),
                manual_inputs=payload.get("manual_inputs"),
            )
        return {"ready": True, **run_experiment_pipeline(memory, payload)}

    session = get_experiment_session(memory)
    last_doc = memory.get_var("document_experiment") or memory.get_var("last_document")
    out: Dict[str, Any] = {
        "ready": session["readiness"]["ready_to_run"],
        "message": (
            "Experiment agent ready. Configure constraints below, then run."
            if session["readiness"]["ready_to_run"]
            else "Complete Hypothesis agent or provide manual clarified question + Socratic questions."
        ),
        "hint_action": "run",
        **session,
    }
    if isinstance(last_doc, dict) and last_doc.get("document_id"):
        out.update(
            {
                "document_id": last_doc.get("document_id"),
                "document_markdown": last_doc.get("markdown"),
                "pdf_url": last_doc.get("pdf_url"),
            }
        )
    return out


def run_analysis_agent(memory: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    from app.services.analysis_service import get_analysis_session, run_analysis_pipeline

    session = get_analysis_session(memory)
    action = (payload.get("action") or "").strip().lower()

    if action not in ("analyze", "run", "execute"):
        return {
            "ready": session.get("ready", False),
            "message": (
                "Analysis agent can run on hypothesis, curve fitting, and ML results."
                if session.get("ready")
                else "Generate a hypothesis first (Hypothesis agent)."
            ),
            "has_analysis": bool(session.get("analysis_full_report")),
            "hypothesis_preview": session.get("hypothesis_preview"),
            "hint_action": "analyze",
            "session": session,
        }

    result = run_analysis_pipeline(memory, payload)
    return {
        "ready": session.get("ready", False),
        "status": result.get("status", "error"),
        "message": result.get("message", ""),
        "analysis_preview": (result.get("analysis_report") or "")[:1500],
        "analysis_report": result.get("analysis_report"),
        "parsed": result.get("parsed"),
        "quality": result.get("quality"),
        "session": result.get("session"),
    }


def run_ml_agent(memory: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    from app.services.ml_session_service import get_ml_session, run_ml_automation

    session = get_ml_session(memory)
    if (payload.get("action") or "").strip().lower() not in ("run", "execute", "train"):
        return {
            "ready": session.get("ready", False),
            "message": (
                "ML agent ready — select curve fitting results and run optimization."
                if session.get("ready")
                else "Run curve fitting first or select result files on the ML Models page."
            ),
            "json_path": session.get("json_path"),
            "csv_path": session.get("csv_path"),
            "hint_action": "run",
            "session": session,
        }

    result = run_ml_automation(memory, payload)
    return {
        "ready": session.get("ready", False),
        "status": result.get("status", "error"),
        "message": result.get("message", ""),
        "result": result.get("result"),
        "session": result.get("session"),
    }


def run_curve_fitting_agent(memory: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    from app.services.curve_fitting_runner import run_curve_fitting_for_api
    from app.services.curve_fitting_service import get_curve_fitting_results

    data_file = payload.get("data_file") or payload.get("trigger_file") or memory.get_var("watcher_last_file")
    ready = bool(data_file and str(data_file).strip())

    if (payload.get("action") or "").strip().lower() not in ("run", "execute", "fit"):
        session = get_curve_fitting_results(memory)
        return {
            "ready": ready,
            "message": (
                f"Curve fitting can run on: {data_file}"
                if ready
                else "Provide data_file in payload or trigger the watcher with a spectral data file."
            ),
            "data_file": data_file,
            "has_curve_results": session.get("has_results"),
            "hint_action": "run",
            **{k: v for k, v in session.items() if k not in ("has_results",)},
        }

    return run_curve_fitting_for_api(memory, payload)


def run_agent_headless(agent: Any, memory: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    name = (getattr(agent, "name", "") or "").lower()
    if "hypothesis" in name:
        return run_hypothesis_agent(memory, payload)
    if "experiment" in name:
        return run_experiment_agent(memory, payload)
    if "analysis" in name:
        return run_analysis_agent(memory, payload)
    if "ml" in name:
        return run_ml_agent(memory, payload)
    if "curve" in name:
        return run_curve_fitting_agent(memory, payload)
    if "watcher" in name:
        return {
            "ready": True,
            "message": "Use POST /api/v1/watcher/start for the file watcher.",
        }
    return {"ready": False, "message": f"No headless handler for agent: {getattr(agent, 'name', agent)}"}
