"""Headless curve fitting execution for the REST API (no Streamlit UI)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.services.composition_fallback import composition_provenance, ensure_composition_csv
from app.services.curve_fitting_exports import sync_ml_paths_from_curve_fitting
from app.services.curve_fitting_service import serialize_curve_fitting_results


def run_curve_fitting_for_api(memory: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the full curve fitting pipeline and return an API-friendly dict.
    Raises no exceptions — failures are returned as status=error payloads.
    """
    data_file = payload.get("data_file") or payload.get("trigger_file") or memory.get_var("watcher_last_file")
    if not data_file or not str(data_file).strip():
        return {
            "ready": False,
            "status": "error",
            "message": "A data file is required (upload CSV or provide data_file_path).",
            "has_results": False,
        }

    data_file = str(data_file)
    memory.set_var("curve_fitting_last_error", None)

    try:
        from app.agents.curve_fitting_agent import CurveFittingAgent

        agent = CurveFittingAgent()
        agent.memory = memory

        comp_file = ensure_composition_csv(data_file, payload.get("composition_file"))
        comp_provenance = composition_provenance(comp_file)
        memory.set_var("curve_fitting_last_composition_file", comp_file)
        memory.set_var("curve_fitting_composition_provenance", comp_provenance)

        params = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}

        results = agent.run_curve_fitting(
            data_csv_path=data_file,
            composition_csv_path=comp_file,
            wells_to_analyze=payload.get("wells_to_analyze") or params.get("wells_to_analyze"),
            reads_to_analyze=payload.get("reads_to_analyze")
            or params.get("read_selection")
            or "auto",
            read_type=str(payload.get("read_type") or params.get("read_type") or "em_spectrum"),
            max_peaks=int(payload.get("max_peaks") or params.get("max_peaks") or 4),
            r2_target=float(payload.get("r2_target") or params.get("r2_target") or 0.90),
            max_attempts=int(payload.get("max_attempts") or params.get("max_attempts") or 3),
            save_plots=True,
            api_delay_seconds=float(
                payload.get("api_delay_seconds") or params.get("api_delay_seconds") or 0.5
            ),
            use_llm=bool(payload.get("use_llm") or params.get("use_llm") or False),
            auto_trigger=True,
        )

        memory.set_var("curve_fitting_last_data_file", data_file)

        if not results or not isinstance(results, dict):
            msg = "Curve fitting returned no results."
            memory.set_var("curve_fitting_last_error", msg)
            return {
                "ready": True,
                "status": "error",
                "message": msg,
                "data_file": data_file,
                "composition_file": comp_file,
                "has_results": False,
            }

        serialized = serialize_curve_fitting_results(results)
        if serialized:
            memory.set_var("curve_fitting_results", serialized)

        files = results.get("files") if isinstance(results.get("files"), dict) else {}
        if files.get("json_results"):
            memory.set_var("curve_fitting_last_json", files["json_results"])
        if files.get("csv_export"):
            memory.set_var("curve_fitting_last_csv", files["csv_export"])

        if results.get("success") is not False and (results.get("results") or []):
            sync_ml_paths_from_curve_fitting(memory)
            workflow_followup = None
            if memory.get_var("workflow_active") or memory.get_var("demo_workflow_running"):
                from app.services.workflow_followups import on_curve_fitting_complete

                workflow_followup = on_curve_fitting_complete(memory)
        else:
            workflow_followup = None

        if results.get("success") is False:
            msg = str(results.get("error") or "Curve fitting failed.")
            memory.set_var("curve_fitting_last_error", msg)
            return {
                "ready": True,
                "status": "error",
                "message": msg,
                "data_file": data_file,
                "composition_file": comp_file,
                "has_results": False,
                "results": serialized,
            }

        wells = results.get("results") or []
        if not wells:
            msg = str(results.get("error") or "No wells were successfully analyzed.")
            memory.set_var("curve_fitting_last_error", msg)
            return {
                "ready": True,
                "status": "error",
                "message": msg,
                "data_file": data_file,
                "composition_file": comp_file,
                "has_results": False,
                "results": serialized,
            }

        response = {
            "ready": True,
            "status": "success",
            "message": "Curve fitting completed (Gaussian / multi-peak model).",
            "has_results": True,
            "results": serialized,
            "data_file": data_file,
            "composition_file": comp_file,
            "composition_provenance": comp_provenance,
        }
        if workflow_followup:
            response["workflow"] = workflow_followup
        return response
    except Exception as exc:
        memory.set_var("curve_fitting_last_error", str(exc))
        memory.set_var("curve_fitting_results", None)
        return {
            "ready": True,
            "status": "error",
            "message": str(exc),
            "data_file": data_file,
            "has_results": False,
        }
