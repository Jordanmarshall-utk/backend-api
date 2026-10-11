"""Bounded scientific audit for report-producing POLARIS agents."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, Field


LLMCallable = Callable[[str], str]
Severity = Literal["critical", "high", "medium", "low"]


class ReportFinding(BaseModel):
    id: str = ""
    severity: Severity = "medium"
    category: str
    message: str
    required_change: str
    evidence: str = ""
    source: str = "deterministic"


class ReportAudit(BaseModel):
    audit_id: str = Field(default_factory=lambda: f"report-audit-{uuid.uuid4().hex[:12]}")
    stage: str
    verdict: Literal["pass", "revise", "block"] = "pass"
    findings: list[ReportFinding] = Field(default_factory=list)
    improvement_proposals: list[dict[str, Any]] = Field(default_factory=list)
    source: str = "deterministic"


_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
_QUANTITY_RE = re.compile(
    r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*"
    r"(?:nm|um|mm|cm|mV|V|meV|eV|ns|us|ms|s|min|h|K|C|mol|mM|uM|M|uL|mL|L)\b",
    re.IGNORECASE,
)
_DECIMAL_RE = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+\.\d+(?:[eE][-+]?\d+)?")


def scientific_tokens(text: str) -> set[str]:
    rendered = str(text or "")
    tokens = {
        re.sub(r"\s+", "", match.group(0)).lower()
        for match in _QUANTITY_RE.finditer(rendered)
    }
    tokens.update(match.group(0).lower() for match in _DECIMAL_RE.finditer(rendered))
    return tokens


def doi_tokens(text: str) -> set[str]:
    return {match.group(0).rstrip(".,;)").lower() for match in _DOI_RE.finditer(str(text or ""))}


def _finding(
    category: str,
    message: str,
    required_change: str,
    *,
    severity: Severity = "medium",
    evidence: str = "",
    source: str = "deterministic",
) -> ReportFinding:
    return ReportFinding(
        category=category,
        severity=severity,
        message=message,
        required_change=required_change,
        evidence=evidence[:500],
        source=source,
    )


def _assign_ids(stage: str, findings: list[ReportFinding]) -> list[ReportFinding]:
    prefix = re.sub(r"[^a-z0-9]+", "-", str(stage or "report").lower()).strip("-")
    for index, item in enumerate(findings, start=1):
        item.id = f"{prefix}-finding-{index:02d}"
    return findings


def _verdict(findings: list[ReportFinding]) -> Literal["pass", "revise", "block"]:
    severities = {item.severity for item in findings}
    if "critical" in severities:
        return "block"
    if findings:
        return "revise"
    return "pass"


def deterministic_report_audit(
    report: str,
    scientific_context: str = "",
    *,
    stage: str = "analysis",
    synthetic_data: bool = False,
    literature_grounded: bool = True,
) -> ReportAudit:
    text = str(report or "").strip()
    lower = text.lower()
    findings: list[ReportFinding] = []

    if len(text) < 200:
        findings.append(
            _finding(
                "completeness.incomplete_report",
                "The report is too short to provide a complete scientific evaluation.",
                "Provide the observations, interpretation boundaries, limitations, and next action.",
                severity="critical",
                evidence=text,
            )
        )

    internal_match = re.search(
        r"(?i)(reasoning council|philosophy manager|primary school|supporting schools|"
        r"humean constraint|kantian constraint|socratic constraint)",
        text,
    )
    if internal_match:
        findings.append(
            _finding(
                "provenance.internal_reasoning_exposed",
                "Internal orchestration language appears in the scientific report.",
                "Remove internal agent and reasoning-policy language from the released report.",
                severity="high",
                evidence=internal_match.group(0),
            )
        )

    overclaim_match = re.search(
        r"(?i)(the hypothesis (?:is|was|appears to be) confirmed|"
        r"(?:these|the) (?:data|results) prove(?:s)? that|"
        r"this (?:proves|confirms) the (?:hypothesis|mechanism))",
        text,
    )
    if overclaim_match:
        findings.append(
            _finding(
                "calibration.overclaim",
                "The report converts limited evidence into proof or confirmation.",
                "Use calibrated support language and state what remains unresolved or untested.",
                severity="high",
                evidence=overclaim_match.group(0),
            )
        )

    if synthetic_data:
        synthetic_match = re.search(
            r"(?i)(supports? the (?:physical )?hypothesis|rejects? the (?:physical )?hypothesis|"
            r"confirms? the (?:physical )?hypothesis|establishes? the (?:physical )?mechanism)",
            text,
        )
        if synthetic_match:
            findings.append(
                _finding(
                    "provenance.synthetic_physical_claim",
                    "Synthetic demo output is being interpreted as evidence about the physical material system.",
                    "Limit the conclusion to software-path validation and withhold physical support or rejection.",
                    severity="critical",
                    evidence=synthetic_match.group(0),
                )
            )

    if not literature_grounded:
        literature_match = re.search(
            r"(?i)(published studies (?:show|report|support|demonstrate)|"
            r"the literature (?:shows|reports|supports|demonstrates|confirms)|"
            r"prior work (?:shows|reports|supports|demonstrates|confirms))",
            text,
        )
        if literature_match:
            findings.append(
                _finding(
                    "evidence.ungrounded_literature",
                    "The report makes a literature claim without a traceable evidence packet.",
                    "Remove the literature assertion or attach a traceable source and evidence excerpt.",
                    severity="high",
                    evidence=literature_match.group(0),
                )
            )

    context_dois = doi_tokens(scientific_context)
    novel_dois = sorted(doi_tokens(text) - context_dois)
    if novel_dois:
        findings.append(
            _finding(
                "evidence.unanchored_citation",
                "The report contains DOI citations absent from the supplied evidence context.",
                "Remove or independently retrieve and verify each unanchored citation.",
                severity="high",
                evidence=", ".join(novel_dois[:5]),
            )
        )

    context_numbers = scientific_tokens(scientific_context)
    novel_numbers = sorted(scientific_tokens(text) - context_numbers)
    if novel_numbers:
        findings.append(
            _finding(
                "evidence.unanchored_quantity",
                "The report contains scientific quantities not found in the supplied data context.",
                "Remove the values or identify the measured/model source that supports each quantity.",
                severity="high",
                evidence=", ".join(novel_numbers[:8]),
            )
        )

    if not re.search(r"(?i)\b(limitations?|uncertaint(?:y|ies)|caveat|confidence interval|error|variance)\b", text):
        findings.append(
            _finding(
                "calibration.missing_uncertainty",
                "The report does not state its principal limitations or uncertainty.",
                "Add a concise limitations and uncertainty section tied to the available evidence.",
                severity="medium",
            )
        )

    if not (
        re.search(r"(?i)\b(observed|measured|measurement)\b", text)
        and re.search(r"(?i)\b(inference|interpretation|suggests?|consistent with|model(?:ed| output)?)\b", text)
    ):
        findings.append(
            _finding(
                "epistemic.observation_inference_boundary",
                "The report does not clearly distinguish observations or model outputs from interpretation.",
                "Label measured observations, fitted/model outputs, inference, and unresolved questions separately.",
                severity="medium",
            )
        )

    findings = _assign_ids(stage, findings)
    return ReportAudit(
        stage=stage,
        verdict=_verdict(findings),
        findings=findings,
        source="deterministic",
    )


def _clean_json(value: str) -> dict[str, Any]:
    text = str(value or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _auditor_prompt(report: str, scientific_context: str, stage: str) -> str:
    return f"""You are the bounded live scientific report auditor for POLARIS.

Audit the {stage} report against only the supplied scientific context. Check evidence provenance, numerical fidelity, observation-versus-inference boundaries, causal overclaiming, limitations, and whether recommendations follow from the available results. Preserve novelty, but do not reward unsupported specificity. Do not propose edits to measured data, fitted parameters, model outputs, or citations. Any source-code, prompt, or schema improvement must be a proposal requiring human approval.

Return only JSON with this schema:
{{
  "findings": [
    {{
      "severity": "critical|high|medium|low",
      "category": "short.category",
      "message": "specific problem",
      "required_change": "bounded report correction",
      "evidence": "short exact excerpt from the report"
    }}
  ],
  "improvement_proposals": [
    {{"scope": "prompt|code|schema", "description": "proposal", "requires_human_approval": true}}
  ]
}}

Do not invent findings, evidence, citations, values, or proposals. Return an empty list when no supported issue exists.

SCIENTIFIC CONTEXT
{scientific_context[-24000:]}

REPORT TO AUDIT
{report[-18000:]}
"""


class ReportAuditorAgent:
    """Combines deterministic checks with a grounded optional LLM audit."""

    def __init__(self, llm_callable: Optional[LLMCallable] = None):
        self.llm_callable = llm_callable

    def audit(
        self,
        report: str,
        scientific_context: str = "",
        *,
        stage: str = "analysis",
        synthetic_data: bool = False,
        literature_grounded: bool = True,
    ) -> ReportAudit:
        base = deterministic_report_audit(
            report,
            scientific_context,
            stage=stage,
            synthetic_data=synthetic_data,
            literature_grounded=literature_grounded,
        )
        if self.llm_callable is None:
            return base

        try:
            payload = _clean_json(self.llm_callable(_auditor_prompt(report, scientific_context, stage)))
        except Exception:
            return base

        llm_findings: list[ReportFinding] = []
        report_lower = str(report or "").lower()
        context_lower = str(scientific_context or "").lower()
        for raw in payload.get("findings") or []:
            if not isinstance(raw, dict):
                continue
            evidence = str(raw.get("evidence") or "").strip()
            category = str(raw.get("category") or "review.other").strip()
            grounded = bool(
                not evidence
                or evidence.lower() in report_lower
                or evidence.lower() in context_lower
                or category.startswith("structure.")
                or category.startswith("calibration.")
            )
            if not grounded:
                continue
            try:
                item = ReportFinding(
                    severity=str(raw.get("severity") or "medium").lower(),
                    category=category,
                    message=str(raw.get("message") or "").strip(),
                    required_change=str(raw.get("required_change") or "").strip(),
                    evidence=evidence,
                    source="llm",
                )
            except Exception:
                continue
            if item.message and item.required_change:
                llm_findings.append(item)

        merged = list(base.findings)
        existing = {(item.category, item.evidence.lower()) for item in merged}
        for item in llm_findings:
            key = (item.category, item.evidence.lower())
            if key not in existing:
                merged.append(item)
                existing.add(key)
        merged = _assign_ids(stage, merged[:16])

        proposals = []
        for raw in payload.get("improvement_proposals") or []:
            if not isinstance(raw, dict) or not str(raw.get("description") or "").strip():
                continue
            proposals.append(
                {
                    "scope": str(raw.get("scope") or "prompt"),
                    "description": str(raw["description"]).strip(),
                    "requires_human_approval": True,
                }
            )

        return base.model_copy(
            update={
                "verdict": _verdict(merged),
                "findings": merged,
                "improvement_proposals": proposals[:8],
                "source": "deterministic+llm",
            }
        )


__all__ = [
    "ReportAudit",
    "ReportAuditorAgent",
    "ReportFinding",
    "deterministic_report_audit",
    "doi_tokens",
    "scientific_tokens",
]
