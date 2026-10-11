"""Bounded report revision for audited hypothesis evaluations."""

from __future__ import annotations

import json
import re
import uuid
from typing import Callable, Optional

from pydantic import BaseModel, Field

from app.agents.hypothesis_auditor_agent import HypothesisAudit
from app.tools.hypothesis_scientific_guard import guard_hypothesis_analysis


LLMCallable = Callable[[str], str]


class HypothesisEditResult(BaseModel):
    edit_id: str = Field(default_factory=lambda: f"edit-{uuid.uuid4().hex[:12]}")
    changed: bool = False
    report: str
    applied_finding_ids: list[str] = Field(default_factory=list)
    unresolved_finding_ids: list[str] = Field(default_factory=list)
    source: str = "none"


def _model_dump(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _clean_editor_output(value: str) -> str:
    text = (value or "").strip()
    text = re.sub(r"^```(?:markdown|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            payload = json.loads(text)
            text = str(payload.get("report") or payload.get("revised_report") or "").strip()
        except json.JSONDecodeError:
            pass
    return text


def _editor_prompt(report: str, audit: HypothesisAudit, hypothesis_context: str) -> str:
    audit_json = json.dumps(_model_dump(audit), ensure_ascii=True, indent=2)
    return f"""You are the report editor for a scientific hypothesis agent.

Revise the hypothesis evaluation report below while accounting for findings in both the report and supplied hypothesis context. Preserve the novel scientific idea and its testability. Correct every bounded report finding, but do not claim a context finding is resolved unless the report states the corrected mechanistic or measurement boundary explicitly. Do not invent evidence, citations, chemical identities, formulas, data, or numerical effects. Clearly separate retrieved evidence, analogy, inference, and untested prediction. Require clarification for ambiguous chemical identities rather than deleting a novel candidate. Align each proposed measurement with the claim it can actually support. Calibrate the conclusion to the evidence.

This is one report-edit pass. Do not discuss source code, prompts, the audit process, philosophy agents, the philosophy manager, or named reasoning constraints such as Kantian or Humean constraints. Do not include a compliance note about how the report was generated. Return only the complete revised scientific report in Markdown. Do not return JSON or a code fence.

HYPOTHESIS AND EVIDENCE CONTEXT
{hypothesis_context[-18000:]}

STRUCTURED AUDIT
{audit_json}

ORIGINAL REPORT
{report[-18000:]}
"""


class HypothesisReportEditorAgent:
    """Applies at most one evidence-preserving revision to a final report."""

    def __init__(self, llm_callable: Optional[LLMCallable] = None):
        self.llm_callable = llm_callable

    def revise(
        self,
        report: str,
        audit: HypothesisAudit,
        hypothesis_context: str = "",
    ) -> HypothesisEditResult:
        original = report or ""
        guarded_original = guard_hypothesis_analysis(original, hypothesis_context)
        if audit.verdict == "pass":
            return HypothesisEditResult(
                changed=guarded_original != original,
                report=guarded_original,
                applied_finding_ids=[],
                unresolved_finding_ids=[],
                source="deterministic_guard" if guarded_original != original else "none",
            )

        candidate = ""
        source = "deterministic_guard_fallback"
        if self.llm_callable is not None:
            try:
                candidate = _clean_editor_output(
                    str(self.llm_callable(_editor_prompt(original, audit, hypothesis_context)))
                )
                if len(candidate) < 120:
                    candidate = ""
                else:
                    source = "scientific_editor_llm+deterministic_guard"
            except Exception:
                candidate = ""

        revised = guard_hypothesis_analysis(candidate or guarded_original, hypothesis_context)
        changed = revised.strip() != original.strip()
        applied = [item.id for item in audit.findings] if changed else []
        unresolved = [] if changed else [item.id for item in audit.findings]
        return HypothesisEditResult(
            changed=changed,
            report=revised,
            applied_finding_ids=applied,
            unresolved_finding_ids=unresolved,
            source=source,
        )


__all__ = ["HypothesisEditResult", "HypothesisReportEditorAgent"]
