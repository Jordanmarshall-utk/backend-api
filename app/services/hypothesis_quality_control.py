"""Bounded audit and revision orchestration for final hypothesis reports."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from app.agents.hypothesis_auditor_agent import (
    HypothesisAuditorAgent,
    deterministic_hypothesis_audit,
)
from app.agents.hypothesis_editor_agent import HypothesisReportEditorAgent
from app.tools.hypothesis_scientific_guard import guard_hypothesis_analysis


logger = logging.getLogger(__name__)
LLMCallable = Callable[[str], str]


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _default_trace_root() -> Path:
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "runtime" / "self_improvement" / "hypothesis"


def _atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".tmp", dir=str(path.parent)
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
        logger.debug("Unable to persist hypothesis quality memory key %s", key, exc_info=True)


def _memory_event(memory: Any, event: dict) -> None:
    if memory is None:
        return
    for method_name in ("log_event", "add_event", "record_event"):
        method = getattr(memory, method_name, None)
        if not callable(method):
            continue
        try:
            try:
                method("hypothesis_live_quality", event)
            except TypeError:
                method({"type": "hypothesis_live_quality", **event})
            return
        except Exception:
            logger.debug("Unable to log hypothesis quality event", exc_info=True)
            return


def run_hypothesis_quality_cycle(
    report: str,
    hypothesis_context: str = "",
    auditor_llm: Optional[LLMCallable] = None,
    editor_llm: Optional[LLMCallable] = None,
    memory: Any = None,
    trace_root: Optional[Path | str] = None,
) -> tuple[str, dict]:
    """Audit a report, revise it at most twice, and persist the full trace."""
    root = Path(trace_root) if trace_root is not None else _default_trace_root()
    timestamp = datetime.now(timezone.utc)
    stamp = timestamp.strftime("%Y%m%dT%H%M%S.%fZ")

    auditor = HypothesisAuditorAgent(auditor_llm)
    audit = auditor.audit(report, hypothesis_context)
    editor = HypothesisReportEditorAgent(editor_llm)
    edit = editor.revise(report, audit, hypothesis_context)
    raw_edit_report = edit.report
    final_report = guard_hypothesis_analysis(raw_edit_report, hypothesis_context)
    edit.changed = bool(edit.changed or final_report != raw_edit_report)
    edit.report = final_report
    post_audit = deterministic_hypothesis_audit(final_report, hypothesis_context)
    edit.unresolved_finding_ids = [item.id for item in post_audit.findings]
    edit_passes = [_dump(edit)]

    if post_audit.findings:
        second_edit = editor.revise(final_report, post_audit, hypothesis_context)
        raw_second_report = second_edit.report
        final_report = guard_hypothesis_analysis(raw_second_report, hypothesis_context)
        second_edit.changed = bool(second_edit.changed or final_report != raw_second_report)
        second_edit.report = final_report
        post_audit = deterministic_hypothesis_audit(final_report, hypothesis_context)
        second_edit.unresolved_finding_ids = [item.id for item in post_audit.findings]
        edit = second_edit
        edit_passes.append(_dump(second_edit))

    blocked = bool(post_audit.findings)
    if blocked and "**Quality-control release gate:**" not in final_report:
        final_report = (
            "**Quality-control release gate:** Blocked draft. The automated audit still found "
            f"{len(post_audit.findings)} unresolved scientific issue(s) after two bounded edit passes. "
            "Treat the document below as an internal draft, not a released evaluation.\n\n"
            f"{final_report}"
        )
        edit.report = final_report

    if audit.findings or edit.changed or post_audit.findings:
        unresolved_count = len(post_audit.findings)
        action = "Revised" if edit.changed else "Reviewed"
        footer = (
            f"**Automated quality control:** {action} {len(audit.findings)} "
            f"full-document finding(s) across {len(edit_passes)} bounded edit pass(es). "
            f"Post-edit full-document audit: `{post_audit.verdict}` with "
            f"{unresolved_count} unresolved finding(s). Trace: `{audit.audit_id}`."
        )
        if "**Automated quality control:**" not in final_report:
            final_report = f"{final_report.rstrip()}\n\n{footer}"
        edit.report = final_report

    audit_payload = _dump(audit)
    edit_payload = _dump(edit)
    post_payload = _dump(post_audit)
    trace = {
        "schema_version": "1.0",
        "created_at": timestamp.isoformat(),
        "scope": "hypothesis_and_analysis_document",
        "max_edit_passes": 2,
        "edit_pass_count": len(edit_passes),
        "release_blocked": blocked,
        "source_changes_applied": False,
        "original_report": report,
        "hypothesis_context": hypothesis_context,
        "audited_document": f"{hypothesis_context}\n\n{report}".strip(),
        "audit": audit_payload,
        "edit": edit_payload,
        "edit_passes": edit_passes,
        "post_edit_deterministic_audit": post_payload,
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
            "status": "awaiting_human_approval",
            "applied": False,
            "proposals": audit.improvement_proposals,
        }
        proposal_path = root / "proposals" / f"{stamp}_{audit.audit_id}.json"
        _atomic_json_write(proposal_path, proposal)
        trace["proposal_path"] = str(proposal_path)
        _atomic_json_write(trace_path, trace)

    _memory_set(memory, "last_hypothesis_quality_audit", audit_payload)
    _memory_set(memory, "last_hypothesis_quality_edit", edit_payload)
    _memory_set(memory, "last_hypothesis_quality_trace", trace)
    _memory_event(
        memory,
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
        "hypothesis_live_quality audit_id=%s verdict=%s findings=%s edited=%s post_edit=%s blocked=%s passes=%s",
        audit.audit_id,
        audit.verdict,
        len(audit.findings),
        edit.changed,
        post_audit.verdict,
        blocked,
        len(edit_passes),
    )
    return final_report, trace


def get_latest_hypothesis_quality(memory: Any = None) -> dict:
    result = {
        "enabled": True,
        "latest_audit": None,
        "latest_edit": None,
        "latest_trace": None,
    }
    if memory is not None:
        try:
            result["enabled"] = bool(
                memory.get_var("hypothesis_live_auditor_enabled", True)
            )
            result["latest_audit"] = memory.get_var(
                "last_hypothesis_quality_audit", None
            )
            result["latest_edit"] = memory.get_var(
                "last_hypothesis_quality_edit", None
            )
            result["latest_trace"] = memory.get_var(
                "last_hypothesis_quality_trace", None
            )
        except Exception:
            logger.debug("Unable to read hypothesis quality memory", exc_info=True)

    if result["latest_trace"] is None:
        audit_dir = _default_trace_root() / "audits"
        if audit_dir.exists():
            candidates = sorted(audit_dir.glob("*.json"), reverse=True)
            if candidates:
                try:
                    result["latest_trace"] = json.loads(
                        candidates[0].read_text(encoding="utf-8")
                    )
                    result["latest_audit"] = result["latest_trace"].get("audit")
                    result["latest_edit"] = result["latest_trace"].get("edit")
                except (OSError, json.JSONDecodeError):
                    logger.debug("Unable to load latest hypothesis quality trace", exc_info=True)
    return result


__all__ = ["get_latest_hypothesis_quality", "run_hypothesis_quality_cycle"]
