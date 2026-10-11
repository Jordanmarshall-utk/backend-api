"""Live audit, bounded report editing, and trace persistence."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from app.agents.report_auditor_agent import ReportAuditorAgent, deterministic_report_audit
from app.agents.report_editor_agent import ReportEditorAgent


logger = logging.getLogger(__name__)
LLMCallable = Callable[[str], str]


def _default_trace_root(stage: str) -> Path:
    project_root = Path(__file__).resolve().parents[3]
    safe_stage = "".join(char if char.isalnum() or char in "-_" else "_" for char in stage)
    return project_root / "runtime" / "self_improvement" / (safe_stage or "report")


def _atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _memory_set(memory: Any, key: str, value: Any) -> None:
    if memory is None:
        return
    try:
        memory.set_var(key, value)
    except Exception:
        logger.debug("Unable to persist report quality key %s", key, exc_info=True)


def _memory_event(memory: Any, stage: str, payload: dict) -> None:
    if memory is None:
        return
    method = getattr(memory, "log_event", None)
    if not callable(method):
        return
    try:
        method("pipeline_report_quality", {"stage": stage, **payload}, stage)
    except TypeError:
        try:
            method("pipeline_report_quality", {"stage": stage, **payload})
        except Exception:
            logger.debug("Unable to log report quality event", exc_info=True)
    except Exception:
        logger.debug("Unable to log report quality event", exc_info=True)


def run_report_quality_cycle(
    report: str,
    scientific_context: str = "",
    *,
    stage: str,
    synthetic_data: bool = False,
    literature_grounded: bool = True,
    auditor_llm: Optional[LLMCallable] = None,
    editor_llm: Optional[LLMCallable] = None,
    memory: Any = None,
    trace_root: Optional[Path | str] = None,
) -> tuple[str, dict]:
    """Audit once, edit at most once, re-audit deterministically, and persist a trace."""
    root = Path(trace_root) if trace_root is not None else _default_trace_root(stage)
    timestamp = datetime.now(timezone.utc)
    stamp = timestamp.strftime("%Y%m%dT%H%M%S.%fZ")

    auditor = ReportAuditorAgent(auditor_llm)
    audit = auditor.audit(
        report,
        scientific_context,
        stage=stage,
        synthetic_data=synthetic_data,
        literature_grounded=literature_grounded,
    )
    editor = ReportEditorAgent(editor_llm)
    edit = editor.revise(
        report,
        audit,
        scientific_context,
        stage=stage,
        synthetic_data=synthetic_data,
        literature_grounded=literature_grounded,
    )
    final_report = edit.report
    post_audit = deterministic_report_audit(
        final_report,
        scientific_context,
        stage=stage,
        synthetic_data=synthetic_data,
        literature_grounded=literature_grounded,
    )
    edit.unresolved_finding_ids = [item.id for item in post_audit.findings]
    blocked = any(item.severity in {"critical", "high"} for item in post_audit.findings)

    if blocked:
        final_report = (
            "**Scientific quality gate:** Internal draft. Automated review found unresolved "
            "high-severity scientific or provenance issues. Do not treat this report as released "
            "until those findings are resolved.\n\n"
            f"{final_report}"
        )
        edit.report = final_report

    trace = {
        "schema_version": "1.0",
        "created_at": timestamp.isoformat(),
        "stage": stage,
        "scope": "generated_report_only",
        "max_edit_passes": 1,
        "edit_pass_count": 1 if audit.verdict != "pass" else 0,
        "release_blocked": blocked,
        "source_changes_applied": False,
        "original_report": report,
        "scientific_context": scientific_context,
        "audit": audit.model_dump(mode="json"),
        "edit": edit.model_dump(mode="json"),
        "post_edit_deterministic_audit": post_audit.model_dump(mode="json"),
        "final_report": final_report,
    }
    trace_path = root / "audits" / f"{stamp}_{audit.audit_id}.json"
    trace["trace_path"] = str(trace_path)
    _atomic_json_write(trace_path, trace)

    proposal_path = None
    if audit.improvement_proposals:
        proposal = {
            "schema_version": "1.0",
            "created_at": timestamp.isoformat(),
            "audit_id": audit.audit_id,
            "stage": stage,
            "status": "awaiting_human_approval",
            "applied": False,
            "proposals": audit.improvement_proposals,
        }
        proposal_path = root / "proposals" / f"{stamp}_{audit.audit_id}.json"
        _atomic_json_write(proposal_path, proposal)
        trace["proposal_path"] = str(proposal_path)
        _atomic_json_write(trace_path, trace)

    key = "".join(char if char.isalnum() else "_" for char in stage.lower()).strip("_")
    _memory_set(memory, f"last_{key}_quality_audit", trace["audit"])
    _memory_set(memory, f"last_{key}_quality_edit", trace["edit"])
    _memory_set(memory, f"last_{key}_quality_trace", trace)
    _memory_event(
        memory,
        stage,
        {
            "audit_id": audit.audit_id,
            "verdict": audit.verdict,
            "finding_count": len(audit.findings),
            "edited": edit.changed,
            "post_edit_verdict": post_audit.verdict,
            "release_blocked": blocked,
            "trace_path": str(trace_path),
            "proposal_path": str(proposal_path) if proposal_path else None,
        },
    )
    logger.info(
        "pipeline_report_quality stage=%s audit_id=%s verdict=%s findings=%s edited=%s post_edit=%s blocked=%s",
        stage,
        audit.audit_id,
        audit.verdict,
        len(audit.findings),
        edit.changed,
        post_audit.verdict,
        blocked,
    )
    return final_report, trace


__all__ = ["run_report_quality_cycle"]
