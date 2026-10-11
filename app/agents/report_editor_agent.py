"""Evidence-preserving revision for audited POLARIS reports."""

from __future__ import annotations

import json
import re
import uuid
from typing import Callable, Optional

from pydantic import BaseModel, Field

from app.agents.report_auditor_agent import ReportAudit, doi_tokens, scientific_tokens


LLMCallable = Callable[[str], str]


class ReportEditResult(BaseModel):
    edit_id: str = Field(default_factory=lambda: f"report-edit-{uuid.uuid4().hex[:12]}")
    changed: bool = False
    report: str
    applied_finding_ids: list[str] = Field(default_factory=list)
    unresolved_finding_ids: list[str] = Field(default_factory=list)
    integrity_rejections: list[str] = Field(default_factory=list)
    source: str = "none"


def _clean_editor_output(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^```(?:markdown|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            payload = json.loads(text)
            text = str(payload.get("report") or payload.get("revised_report") or "").strip()
        except json.JSONDecodeError:
            pass
    return text


def _deterministic_edit(
    report: str,
    audit: ReportAudit,
    *,
    synthetic_data: bool,
    literature_grounded: bool,
) -> str:
    text = str(report or "").strip()
    text = re.sub(
        r"(?i)\bthe hypothesis (?:is|was) confirmed\b",
        "the current evidence is consistent with the hypothesis but does not prove it",
        text,
    )
    text = re.sub(
        r"(?i)\bthe hypothesis appears to be confirmed\b",
        "the current evidence is consistent with the hypothesis but does not prove it",
        text,
    )
    text = re.sub(
        r"(?i)\b(?:these|the) (?:data|results) prove(?:s)? that\b",
        "the current data are consistent with the interpretation that",
        text,
    )

    if not literature_grounded:
        replacement = (
            "## Literature Grounding\n\n"
            "No traceable LiteratureAgent evidence packet was supplied for this analysis. "
            "Literature agreement, novelty, and published mechanisms were not evaluated.\n\n"
        )
        text = re.sub(
            r"(?ims)^## Literature (?:Context|Grounding)\s*.*?(?=^## |\Z)",
            replacement,
            text,
        )

    if any(item.category == "provenance.internal_reasoning_exposed" for item in audit.findings):
        kept = []
        for line in text.splitlines():
            if re.search(
                r"(?i)(reasoning council|philosophy manager|primary school|supporting schools|"
                r"humean constraint|kantian constraint|socratic constraint)",
                line,
            ):
                continue
            kept.append(line)
        text = "\n".join(kept).strip()

    if synthetic_data and "synthetic polaris demo data" not in text.lower():
        text = (
            "**Data provenance:** Synthetic POLARIS demo data. The run evaluates software-path "
            "behavior and does not test the physical materials hypothesis.\n\n"
            f"{text}"
        )

    if any(item.category == "calibration.missing_uncertainty" for item in audit.findings):
        text = (
            f"{text.rstrip()}\n\n## Limitations and uncertainty\n\n"
            "Interpretation is limited to the supplied measurements, fitted/model outputs, and "
            "traceable evidence context. Associations and model attributions do not by themselves "
            "establish a causal mechanism; unresolved alternatives require discriminating tests."
        )
    return text.strip()


def _editor_prompt(report: str, audit: ReportAudit, scientific_context: str, stage: str) -> str:
    audit_json = json.dumps(audit.model_dump(mode="json"), ensure_ascii=True, indent=2)
    return f"""You are the bounded scientific report editor for the POLARIS {stage} agent.

Revise only the interpretation language needed to address the structured audit. Preserve every context-grounded measured value, fitted parameter, model output, DOI, and comparison exactly. Do not add any number, unit-bearing quantity, citation, material identity, mechanism, or result absent from the report and context. Separate observations, model outputs, inference, and uncertainty. Preserve useful novelty and recommendations, but calibrate them to the evidence.

Do not discuss the audit, source code, prompts, philosophy agents, or internal reasoning. Return only the complete revised report in Markdown, with no JSON and no code fence.

SCIENTIFIC CONTEXT
{scientific_context[-24000:]}

STRUCTURED AUDIT
{audit_json}

ORIGINAL REPORT
{report[-18000:]}
"""


def _integrity_rejections(original: str, candidate: str, context: str) -> list[str]:
    allowed_numbers = scientific_tokens(original) | scientific_tokens(context)
    candidate_numbers = scientific_tokens(candidate)
    anchored_original = scientific_tokens(original) & scientific_tokens(context)
    allowed_dois = doi_tokens(original) | doi_tokens(context)
    candidate_dois = doi_tokens(candidate)
    anchored_dois = doi_tokens(original) & doi_tokens(context)
    rejected: list[str] = []
    novel_numbers = sorted(candidate_numbers - allowed_numbers)
    missing_anchored_numbers = sorted(anchored_original - candidate_numbers)
    novel_dois = sorted(candidate_dois - allowed_dois)
    missing_anchored_dois = sorted(anchored_dois - candidate_dois)
    if novel_numbers:
        rejected.append(f"new scientific quantities: {', '.join(novel_numbers[:8])}")
    if missing_anchored_numbers:
        rejected.append(f"removed anchored quantities: {', '.join(missing_anchored_numbers[:8])}")
    if novel_dois:
        rejected.append(f"new DOI citations: {', '.join(novel_dois[:5])}")
    if missing_anchored_dois:
        rejected.append(f"removed anchored DOI citations: {', '.join(missing_anchored_dois[:5])}")
    return rejected


class ReportEditorAgent:
    """Applies at most one interpretation-only edit with numerical integrity checks."""

    def __init__(self, llm_callable: Optional[LLMCallable] = None):
        self.llm_callable = llm_callable

    def revise(
        self,
        report: str,
        audit: ReportAudit,
        scientific_context: str = "",
        *,
        stage: str = "analysis",
        synthetic_data: bool = False,
        literature_grounded: bool = True,
    ) -> ReportEditResult:
        original = str(report or "")
        if audit.verdict == "pass":
            return ReportEditResult(report=original)

        deterministic = _deterministic_edit(
            original,
            audit,
            synthetic_data=synthetic_data,
            literature_grounded=literature_grounded,
        )
        candidate = ""
        integrity_rejections: list[str] = []
        source = "deterministic"
        if self.llm_callable is not None:
            try:
                candidate = _clean_editor_output(
                    self.llm_callable(_editor_prompt(original, audit, scientific_context, stage))
                )
            except Exception:
                candidate = ""
            if candidate:
                integrity_rejections = _integrity_rejections(
                    original,
                    candidate,
                    scientific_context,
                )
                if integrity_rejections:
                    candidate = ""
                    source = "deterministic_integrity_fallback"
                elif len(candidate) >= 200:
                    source = "llm+integrity_guard"
                else:
                    candidate = ""
                    source = "deterministic_incomplete_fallback"
            else:
                candidate = ""

        revised = candidate or deterministic
        changed = revised.strip() != original.strip()
        return ReportEditResult(
            changed=changed,
            report=revised,
            applied_finding_ids=[item.id for item in audit.findings] if changed else [],
            unresolved_finding_ids=[] if changed else [item.id for item in audit.findings],
            integrity_rejections=integrity_rejections,
            source=source,
        )


__all__ = ["ReportEditResult", "ReportEditorAgent"]
