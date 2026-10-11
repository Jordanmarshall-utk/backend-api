"""Analysis Agent page session — parity with Streamlit pages/analysis.py."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional

from app.services.curve_fitting_service import serialize_curve_fitting_results
from app.services.llm_runtime import require_api_key
from app.services.reasoning_integration import reasoning_context_for_stage
from app.services.report_quality_control import run_report_quality_cycle
from app.tools.paths import get_results_dir
from app.utils.json_safe import to_jsonable


def _hypothesis_text(memory: Any) -> str:
    hyp = memory.view_component("hypothesis") or memory.get_var("last_hypothesis")
    if hyp and str(hyp).strip():
        return str(hyp).strip()
    return ""


def _hypothesis_fingerprint(hypothesis: str) -> str:
    normalized = " ".join(str(hypothesis or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


def _hypothesis_preview(memory: Any) -> Optional[str]:
    text = _hypothesis_text(memory)
    if not text:
        return None
    return text[:500] + ("..." if len(text) > 500 else "")


def _load_curve_fitting_results(memory: Any) -> Optional[Dict[str, Any]]:
    raw = memory.get_var("curve_fitting_results")
    if isinstance(raw, dict):
        if raw.get("wells"):
            return raw
        if raw.get("results"):
            return serialize_curve_fitting_results(raw) or raw

    results_dir = get_results_dir()
    if not os.path.isdir(results_dir):
        return None
    json_files = [f for f in os.listdir(results_dir) if f.endswith("_peak_fit_results.json")]
    if not json_files:
        return None
    latest = max(json_files, key=lambda f: os.path.getmtime(os.path.join(results_dir, f)))
    path = os.path.join(results_dir, latest)
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _summarize_curve_fitting(results: Dict[str, Any]) -> str:
    if not isinstance(results, dict):
        return "No curve fitting results available."

    parts: list[str] = []

    if "wells" in results and isinstance(results["wells"], list):
        wells = results["wells"]
        successful = sum(1 for w in wells if (w.get("fit") or {}).get("success"))
        parts.append(f"**Total Wells Analyzed:** {len(wells)}")
        parts.append(f"**Successful Fits:** {successful}/{len(wells)}")
        rows: list[str] = []
        all_centers: list[float] = []
        r2_values: list[float] = []
        for w in wells:
            fit = w.get("fit") or {}
            if fit.get("success"):
                peaks = fit.get("peaks") or []
                centers = [p.get("center") for p in peaks[:3] if p.get("center") is not None]
                all_centers.extend(float(center) for center in centers)
                r2 = fit.get("r2")
                if isinstance(r2, (int, float)):
                    r2_values.append(float(r2))
                label = f"{w.get('well_name', '?')}"
                if w.get("read"):
                    label += f" (Read {w['read']})"
                peak_str = ", ".join(f"{c:.1f}nm" for c in centers) if centers else "n/a"
                rows.append(f"  - {label}: R-squared={r2:.5f}, peaks at {peak_str}" if r2 is not None else f"  - {label}: peaks at {peak_str}")
        if all_centers:
            parts.append(f"**Fitted Peak-Center Range:** {min(all_centers):.2f}-{max(all_centers):.2f} nm")
        if r2_values:
            parts.append(f"**R-squared Range:** {min(r2_values):.5f}-{max(r2_values):.5f}")
        parts.extend(rows if len(rows) <= 24 else [*rows[:12], *rows[-12:]])
        return "\n".join(parts)

    if "wells" in results and isinstance(results["wells"], dict):
        wells_dict = results["wells"]
        parts.append(f"**Total Wells:** {len(wells_dict)}")
        successful = 0
        rows: list[str] = []
        centers: list[float] = []
        r2_values: list[float] = []
        for name, data in wells_dict.items():
            fr = (data or {}).get("fitting_results") or {}
            qm = fr.get("quality_metrics") or {}
            if fr.get("success"):
                successful += 1
            r2 = qm.get("r_squared")
            if isinstance(r2, (int, float)):
                r2_values.append(float(r2))
            well_centers = [
                float(peak["center_nm"])
                for peak in fr.get("quality_peaks") or []
                if isinstance(peak, dict) and isinstance(peak.get("center_nm"), (int, float))
            ]
            centers.extend(well_centers)
            center_text = ", ".join(f"{value:.2f} nm" for value in well_centers) or "n/a"
            r2_text = f"{float(r2):.5f}" if isinstance(r2, (int, float)) else "n/a"
            rows.append(f"  - {name}: R-squared={r2_text}; fitted centers={center_text}")
        parts.append(f"**Successful Fits:** {successful}/{len(wells_dict)}")
        if centers:
            parts.append(f"**Fitted Peak-Center Range:** {min(centers):.2f}-{max(centers):.2f} nm")
        if r2_values:
            parts.append(f"**R-squared Range:** {min(r2_values):.5f}-{max(r2_values):.5f}")
        parts.extend(rows if len(rows) <= 24 else [*rows[:12], *rows[-12:]])
        return "\n".join(parts)

    summary = results.get("summary")
    if isinstance(summary, dict):
        parts.append(
            f"**Summary:** {summary.get('successful_fits', 0)}/{summary.get('total_wells', 0)} successful fits"
        )
    return "\n".join(parts) if parts else "Curve fitting results loaded (see JSON for details)."


def get_analysis_session(memory: Any) -> Dict[str, Any]:
    hypothesis = _hypothesis_text(memory)
    hypothesis_preview = _hypothesis_preview(memory)
    hypothesis_fingerprint = _hypothesis_fingerprint(hypothesis)
    hypothesis_ctx = ""
    if hypothesis:
        hypothesis_ctx += f"**Hypothesis:**\n{hypothesis}\n"
        clarified = memory.view_component("clarified_question")
        if clarified:
            hypothesis_ctx += f"**Research Question:**\n{clarified}\n"
        socratic = memory.view_component("socratic_pass")
        if socratic:
            hypothesis_ctx += f"**Socratic Analysis:**\n{socratic}\n"
    else:
        hypothesis_ctx = (
            "No finalized hypothesis is available. Generate and select a hypothesis before "
            "running Analysis. Research-question context is not treated as a hypothesis."
        )

    exp_out = memory.get_var("experimental_outputs") or {}
    exp_ctx = ""
    if isinstance(exp_out, dict):
        if exp_out.get("plan"):
            exp_ctx += f"**Experimental Plan:**\n{exp_out['plan']}\n"
        if exp_out.get("worklist"):
            wl = str(exp_out["worklist"])
            exp_ctx += f"**Worklist:**\n{wl[:500]}{'…' if len(wl) > 500 else ''}\n"

    curve = _load_curve_fitting_results(memory)
    gp_results = memory.get_var("gp_results")
    mc_results = memory.get_var("monte_carlo_results")
    stored_report = memory.get_var("analysis_full_report") or memory.view_component("analysis_rubric")
    stored_fingerprint = str(memory.get_var("analysis_hypothesis_fingerprint") or "")
    analysis_matches_hypothesis = bool(
        stored_report
        and hypothesis_fingerprint
        and stored_fingerprint == hypothesis_fingerprint
    )
    full_report = stored_report if analysis_matches_hypothesis else None
    recommendations = (
        (memory.get_var("analysis_recommendations") or [])
        if analysis_matches_hypothesis
        else []
    )

    return to_jsonable(
        {
            "research_goal": memory.get_var("research_goal") or "",
            "hypothesis_context": hypothesis_ctx.strip(),
            "experimental_context": exp_ctx.strip() or "No experimental data available.",
            "has_curve_fitting": curve is not None,
            "curve_fitting_summary": _summarize_curve_fitting(curve) if curve else None,
            "has_gp_results": bool(gp_results),
            "gp_results": gp_results if isinstance(gp_results, dict) else None,
            "monte_carlo_results": mc_results if isinstance(mc_results, dict) else None,
            "has_hypothesis": bool(hypothesis),
            "hypothesis_preview": hypothesis_preview,
            "analysis_full_report": str(full_report) if full_report else None,
            "analysis_recommendations": recommendations,
            "analysis_matches_hypothesis": analysis_matches_hypothesis,
            "stale_analysis_available": bool(stored_report and not analysis_matches_hypothesis),
            "ready": bool(hypothesis),
        }
    )


def patch_analysis_session(memory: Any, *, research_goal: Optional[str] = None) -> Dict[str, Any]:
    if research_goal is not None:
        memory.set_var("research_goal", research_goal)
    return get_analysis_session(memory)


def _guard_analysis_report(
    report: str,
    *,
    synthetic_data: bool,
    literature_grounded: bool,
) -> str:
    import re

    text = str(report or "").strip()
    text = re.sub(
        r"(?i)\bthe hypothesis is confirmed\b",
        "the hypothesis is supported by the current dataset but is not proven",
        text,
    )
    text = re.sub(
        r"(?i)\bthe hypothesis appears to be confirmed\b",
        "the current dataset is consistent with the hypothesis but does not prove it",
        text,
    )

    if synthetic_data:
        synthetic_evaluation = (
            "## Goal / Hypothesis Evaluation\n\n"
            "This run used synthetic demo data whose composition-to-peak relationship was programmed "
            "into the generator. The result validates data ingestion, curve fitting, and predictive-model "
            "plumbing; it does not test, support, reject, or confirm the physical materials hypothesis.\n\n"
        )
        text = re.sub(
            r"(?ims)^## Goal / Hypothesis Evaluation\s*.*?(?=^## Results Analysis)",
            synthetic_evaluation,
            text,
        )
        text = re.sub(r"(?i)\bconfirming\b", "reproducing", text)
        text = re.sub(r"(?i)\bconfirms\b", "reproduces", text)
        text = re.sub(
            r"(?i)\baligns? (?:well )?with the hypothesis\b",
            "matches the programmed synthetic relationship only",
            text,
        )
        text = re.sub(
            r"(?i)\bsupports the hypothesis\b",
            "matches the programmed demo relationship but does not support the physical hypothesis",
            text,
        )
        text = re.sub(
            r"(?i)\bsupporting the hypothesis\b",
            "matching the programmed demo relationship without supporting the physical hypothesis",
            text,
        )
        text = re.sub(
            r"(?i)\bsuggests that the electronic properties .*?composition\.",
            "does not establish physical electronic-property tuning.",
            text,
        )

    if not literature_grounded:
        literature_section = (
            "## Literature Grounding\n\n"
            "No traceable LiteratureAgent evidence packet was supplied for this analysis. Literature "
            "agreement, novelty, and published mechanisms were therefore not evaluated.\n\n"
        )
        text = re.sub(
            r"(?ims)^## Literature (?:Context|Grounding)\s*.*?(?=^## |\Z)",
            literature_section,
            text,
        )

    prefix = ""
    if synthetic_data:
        prefix = (
            "**Data provenance:** Synthetic POLARIS demo data. Scientific conclusions about the "
            "material system are out of scope for this run.\n\n"
        )
    return f"{prefix}{text}".strip()


def run_analysis_pipeline(memory: Any, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = payload or {}
    if payload.get("research_goal"):
        memory.set_var("research_goal", payload["research_goal"])

    session = get_analysis_session(memory)
    if not session.get("has_hypothesis"):
        return {"status": "error", "message": "Generate a hypothesis first (Hypothesis agent).", "session": session}

    curve = _load_curve_fitting_results(memory)
    if not curve and not payload.get("skip_curve_check"):
        return {
            "status": "error",
            "message": "No curve fitting results found. Run Curve Fitting first or upload results JSON.",
            "session": session,
        }

    try:
        require_api_key(memory)
    except ValueError as exc:
        return {"status": "error", "message": str(exc), "session": session}

    from app.agents.analysis_agent import AnalysisAgent

    agent = AnalysisAgent()
    agent.memory = memory

    hypothesis_context = agent._get_hypothesis_context()
    experimental_context = agent._get_experimental_context()
    synthetic_data = bool(memory.get_var("demo_workflow_running"))
    literature_context = str(payload.get("literature_context") or "").strip()
    literature_grounded = bool(literature_context)
    provenance_context = (
        "DATA PROVENANCE: Synthetic POLARIS demo data. Evaluate software-path behavior only; "
        "do not treat the programmed trend as physical evidence."
        if synthetic_data
        else "DATA PROVENANCE: Treat values as measured only when their source metadata identifies them as measured."
    )
    literature_rule = (
        f"TRACEABLE LITERATURE EVIDENCE:\n{literature_context}"
        if literature_grounded
        else "LITERATURE GATE: No traceable literature packet was supplied. Do not make claims about published findings or literature agreement."
    )
    experimental_context = f"{experimental_context}\n\n{provenance_context}\n{literature_rule}"
    reasoning_context = (
        str(payload.get("reasoning_context") or "")
        if "reasoning_context" in payload
        else reasoning_context_for_stage(memory, "analysis")
    )
    if reasoning_context:
        experimental_context = (
            f"{experimental_context}\n\n{reasoning_context}\n"
            "Apply the council guidance as epistemic checks only. Preserve measured values and "
            "distinguish observations, model outputs, inference, and unresolved uncertainty. Do not "
            "mention the council, philosophy schools, or internal reasoning instructions in the report."
        )
    curve_summary = _summarize_curve_fitting(curve) if curve else "No curve fitting data."
    gp_results = agent._get_gp_results()
    monte_carlo_results = memory.get_var("monte_carlo_results")

    gp_summary = agent._summarize_gp_results(gp_results) if gp_results else ""
    mc_summary = ""
    if monte_carlo_results:
        mc_summary = agent._summarize_monte_carlo_results(monte_carlo_results)

    combined_summary = curve_summary
    if gp_summary:
        combined_summary += f"\n\n## ML (GP)\n{gp_summary}"
    if mc_summary:
        combined_summary += f"\n\n## ML (Monte Carlo)\n{mc_summary}"

    result = agent._analyze_results_with_llm(
        hypothesis_context,
        experimental_context,
        combined_summary,
        curve,
        gp_results,
        monte_carlo_results,
    )

    if not result.get("success"):
        return {
            "status": "error",
            "message": result.get("error") or "Analysis failed.",
            "session": get_analysis_session(memory),
        }

    raw_analysis_text = result.get("analysis") or ""
    analysis_text = _guard_analysis_report(
        raw_analysis_text,
        synthetic_data=synthetic_data,
        literature_grounded=literature_grounded,
    )
    live_auditor_enabled = bool(memory.get_var("pipeline_live_auditor_enabled", True)) and bool(
        memory.get_var("analysis_live_auditor_enabled", True)
    )
    quality_trace = None
    if live_auditor_enabled:
        from app.tools import socratic

        scientific_context = (
            f"HYPOTHESIS CONTEXT\n{hypothesis_context}\n\n"
            f"EXPERIMENTAL AND PROVENANCE CONTEXT\n{experimental_context}\n\n"
            f"CURVE-FITTING AND MODEL SUMMARY\n{combined_summary}\n\n"
            f"TRACEABLE LITERATURE PACKET\n{literature_context or 'None supplied.'}"
        )
        analysis_text, quality_trace = run_report_quality_cycle(
            analysis_text,
            scientific_context,
            stage="analysis",
            synthetic_data=synthetic_data,
            literature_grounded=literature_grounded,
            auditor_llm=socratic.generate_text_with_llm,
            editor_llm=socratic.generate_text_with_llm,
            memory=memory,
        )
        analysis_text = _guard_analysis_report(
            analysis_text,
            synthetic_data=synthetic_data,
            literature_grounded=literature_grounded,
        )
    parsed = agent._parse_analysis_response(analysis_text)
    if synthetic_data:
        parsed["hypothesis_status"] = "not_tested"
    quality = {
        "synthetic_data": synthetic_data,
        "literature_grounded": literature_grounded,
        "report_guard_applied": analysis_text != raw_analysis_text,
        "scientific_scope": "software_validation_only" if synthetic_data else "scientific_analysis",
        "live_auditor_enabled": live_auditor_enabled,
        "audit_verdict": (quality_trace or {}).get("audit", {}).get("verdict"),
        "audit_finding_count": len((quality_trace or {}).get("audit", {}).get("findings") or []),
        "report_edited": bool((quality_trace or {}).get("edit", {}).get("changed")),
        "release_blocked": bool((quality_trace or {}).get("release_blocked")),
        "trace_path": (quality_trace or {}).get("trace_path"),
        "source_changes_applied": False,
    }
    memory.insert_interaction("assistant", analysis_text, "analysis_rubric", "analysis")
    memory.set_var("analysis_full_report", analysis_text)
    memory.set_var(
        "analysis_results",
        {"preview": analysis_text[:2000], "parsed": parsed, "quality": quality},
    )
    memory.set_var("analysis_recommendations", parsed.get("experiment_recommendations"))
    memory.set_var(
        "analysis_hypothesis_fingerprint",
        _hypothesis_fingerprint(_hypothesis_text(memory)),
    )
    memory.set_var("analysis_quality", quality)
    memory.set_var("analysis_ready", True)

    return to_jsonable(
        {
            "status": "success",
            "message": (
                "Analysis report generated; automated scientific review retained it as an internal draft."
                if quality["release_blocked"]
                else "Analysis report generated and checked by the live report auditor."
            ),
            "analysis_report": analysis_text,
            "parsed": parsed,
            "quality": quality,
            "session": get_analysis_session(memory),
        }
    )
