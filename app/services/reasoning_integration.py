"""Adapters between POLARIS domain agents and the shared ReasoningService."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Dict, List, Mapping, Optional

from app.agents.reasoning.reasoning_service import ReasoningService, normalize_reasoning_mode
from app.agents.reasoning.schemas import ReasoningPlan, TaskPacket


@dataclass(frozen=True)
class HypothesisReasoningDecision:
    plan: ReasoningPlan
    effective_policy: str


@dataclass(frozen=True)
class PipelineReasoningDecision:
    plan: ReasoningPlan
    effective_policy: str
    context: str


_PIPELINE_EXECUTION_ACTIONS = {
    "experiment": {"generate_plan", "run", "execute"},
    "curve_fitting": {"fit", "run", "execute"},
    "ml_models": {"train", "run", "execute"},
    "analysis": {"analyze", "run", "execute"},
}


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [f"{key}: {json.dumps(item, default=str)}" for key, item in value.items()]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def build_hypothesis_task_packet(memory: Any, question: str, stage: str) -> TaskPacket:
    uploaded = memory.get_var("uploaded_files", []) or []
    evidence = []
    for item in uploaded:
        if isinstance(item, dict):
            evidence.append(str(item.get("filename") or item.get("name") or item.get("path") or item))
        else:
            evidence.append(str(item))

    constraints = _string_list(memory.get_var("experimental_constraints", {}))
    assumptions = _string_list(memory.get_var("reasoning_assumptions", []))
    contradictions = _string_list(memory.get_var("reasoning_contradictions", []))
    current_stage = str(memory.get_var("stage") or "initial")
    last_hypothesis = memory.get_var("last_hypothesis")
    if last_hypothesis and current_stage != "initial":
        assumptions.append(f"Prior hypothesis exists: {str(last_hypothesis)[:300]}")

    session_context: Dict[str, Any] = {
        "experimental_mode": bool(memory.get_var("experimental_mode", False)),
        "research_goal": str(memory.get_var("research_goal", "") or ""),
        "current_hypothesis_stage": current_stage,
        "hypothesis_round_count": int(memory.get_var("hypothesis_round_count", 0) or 0),
        "has_uploaded_files": bool(uploaded),
        "has_negative_hypotheses": bool(memory.get_negative_hypotheses(limit=1)),
    }
    return TaskPacket(
        domain_agent="Hypothesis Agent",
        objective=question,
        stage=stage,
        evidence=evidence,
        assumptions=assumptions,
        constraints=constraints,
        contradictions=contradictions,
        requested_output="A scientifically testable hypothesis and reasoning path",
        session_context=session_context,
    )


def effective_reasoning_policy(plan: ReasoningPlan, requested_policy: str) -> str:
    requested = requested_policy or "auto"
    if plan.mode != "Active" or not plan.crew_success or plan.fallback_used:
        return requested
    try:
        from app.tools.scientific_reasoning_policy_layer import resolve_policy_id

        return resolve_policy_id(plan.primary_school) or plan.primary_school
    except Exception:
        return requested


def plan_hypothesis_reasoning(
    memory: Any,
    question: str,
    stage: str,
    requested_policy: str = "auto",
) -> HypothesisReasoningDecision:
    mode = normalize_reasoning_mode(memory.get_var("reasoning_mode", "Shadow"))
    packet = build_hypothesis_task_packet(memory, question, stage)
    plan = ReasoningService(memory=memory).plan(packet, mode=mode)
    policy = effective_reasoning_policy(plan, requested_policy)
    memory.set_var("reasoning_policy", policy)
    return HypothesisReasoningDecision(plan=plan, effective_policy=policy)


def reasoning_context_for_synthesis(memory: Any) -> str:
    mode = normalize_reasoning_mode(memory.get_var("reasoning_mode", "Shadow"))
    raw = memory.get_var("last_reasoning_plan")
    if mode != "Active" or not isinstance(raw, dict):
        return ""
    try:
        plan = ReasoningPlan.model_validate(raw)
    except Exception:
        return ""
    if not plan.crew_success or plan.fallback_used or not plan.reasoning_required:
        return ""
    assignments = "\n".join(
        f"- {school}: {assignment}" for school, assignment in plan.assignments.items()
    )
    return (
        "POLARIS Reasoning Council guidance (strategy only):\n"
        f"Primary school: {plan.primary_school}\n"
        f"Supporting schools: {', '.join(plan.supporting_schools) or 'none'}\n"
        f"Rationale: {plan.rationale}\n"
        f"Assignments:\n{assignments}"
    )


def _memory_value(memory: Any, name: str, default: Any = None) -> Any:
    try:
        return memory.get_var(name, default)
    except Exception:
        return default


def _compact_value(value: Any, limit: int = 500) -> str:
    try:
        rendered = json.dumps(value, default=str, ensure_ascii=True)
    except Exception:
        rendered = str(value)
    return rendered[:limit]


def pipeline_stage_for_agent(domain_agent: str) -> str:
    name = str(domain_agent or "").lower()
    if "curve" in name:
        return "curve_fitting"
    if "ml" in name:
        return "ml_models"
    if "analysis" in name:
        return "analysis"
    if "experiment" in name:
        return "experiment"
    if "watcher" in name:
        return "watcher"
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_") or "unknown"


def should_plan_pipeline_reasoning(
    domain_agent: str,
    payload: Optional[Mapping[str, Any]] = None,
    *,
    stage: str = "",
) -> bool:
    resolved_stage = stage or pipeline_stage_for_agent(domain_agent)
    action = str((payload or {}).get("action") or "").strip().lower()
    return action in _PIPELINE_EXECUTION_ACTIONS.get(resolved_stage, set())


def _pipeline_objective(
    memory: Any,
    domain_agent: str,
    stage: str,
    payload: Mapping[str, Any],
    objective: str,
) -> str:
    name = f"{domain_agent} {stage}".lower()
    if "curve" in name:
        source = payload.get("data_file") or payload.get("source") or "the supplied spectra"
        return f"Calculate curve fits for {source}, preserve raw measurements, and export results."
    explicit = objective or str(
        payload.get("objective")
        or payload.get("research_goal")
        or payload.get("question")
        or ""
    ).strip()
    research_goal = str(_memory_value(memory, "research_goal", "") or "").strip()
    hypothesis = str(_memory_value(memory, "last_hypothesis", "") or "").strip()
    if explicit:
        return explicit
    if "analysis" in name:
        return (
            "Evaluate whether the experimental and model evidence supports the current hypothesis, "
            "separate observation from inference, and determine the next justified action."
        )
    if "ml" in name:
        return (
            f"Select and run a predictive model for {research_goal or hypothesis or 'the current data'}, "
            "preserving validation and uncertainty constraints."
        )
    if "experiment" in name:
        return f"Design discriminating experiments for {hypothesis or research_goal or 'the current research goal'}."
    return f"Run {domain_agent} for {research_goal or hypothesis or stage}."


def build_pipeline_task_packet(
    memory: Any,
    domain_agent: str,
    stage: str,
    *,
    payload: Optional[Mapping[str, Any]] = None,
    objective: str = "",
) -> TaskPacket:
    data = dict(payload or {})
    uploaded = _memory_value(memory, "uploaded_files", []) or []
    evidence = _string_list(uploaded)
    evidence_values = (
        ("Current hypothesis", _memory_value(memory, "last_hypothesis")),
        ("Experimental outputs", _memory_value(memory, "experimental_outputs")),
        ("Curve-fitting results", _memory_value(memory, "curve_fitting_results")),
        ("Gaussian-process results", _memory_value(memory, "gp_results")),
        ("Monte Carlo results", _memory_value(memory, "monte_carlo_results")),
        ("Prior analysis", _memory_value(memory, "analysis_results")),
    )
    for label, value in evidence_values:
        if value:
            evidence.append(f"{label}: {_compact_value(value)}")

    constraints = _string_list(_memory_value(memory, "experimental_constraints", {}))
    for key in (
        "model_choice",
        "optimization_model_choice",
        "plate_format",
        "action",
        "target",
        "kernel_type",
        "acquisition_method",
    ):
        if data.get(key) is not None:
            constraints.append(f"{key}: {_compact_value(data[key])}")

    assumptions = _string_list(_memory_value(memory, "reasoning_assumptions", []))
    contradictions = _string_list(_memory_value(memory, "reasoning_contradictions", []))
    requested_outputs = {
        "experiment": "A constraint-compliant, discriminating experimental plan",
        "curve_fitting": "Deterministic fitted parameters and diagnostics without altered measurements",
        "ml_models": "Validated predictive outputs with uncertainty and model limitations",
        "analysis": "An evidence-calibrated interpretation and justified next action",
    }
    stage_key = re.sub(r"[^a-z0-9]+", "_", stage.lower()).strip("_")
    return TaskPacket(
        domain_agent=domain_agent,
        objective=_pipeline_objective(memory, domain_agent, stage, data, objective),
        stage=stage,
        evidence=evidence,
        assumptions=assumptions,
        constraints=constraints,
        contradictions=contradictions,
        requested_output=requested_outputs.get(stage_key, f"A validated {stage} result"),
        session_context={
            "action": str(data.get("action") or "run"),
            "has_hypothesis": bool(_memory_value(memory, "last_hypothesis")),
            "has_curve_results": bool(_memory_value(memory, "curve_fitting_results")),
            "has_ml_results": bool(_memory_value(memory, "gp_results")),
            "has_analysis_results": bool(_memory_value(memory, "analysis_results")),
            "explicit_model_choice": bool(data.get("model_choice")),
        },
    )


def reasoning_context_from_plan(plan: ReasoningPlan) -> str:
    if (
        plan.mode != "Active"
        or not plan.reasoning_required
        or not plan.crew_success
        or plan.fallback_used
    ):
        return ""
    assignments = "\n".join(
        f"- {school}: {assignment}" for school, assignment in plan.assignments.items()
    )
    return (
        "POLARIS Reasoning Council guidance (strategy and epistemic checks only):\n"
        f"Primary school: {plan.primary_school}\n"
        f"Supporting schools: {', '.join(plan.supporting_schools) or 'none'}\n"
        f"Rationale: {plan.rationale}\n"
        f"Assignments:\n{assignments}\n"
        "Preserve measured values, deterministic calculations, explicit constraints, and user choices. "
        "Use this guidance internally; do not mention the council, philosophy schools, or internal "
        "reasoning instructions in the scientific output."
    )


def _pipeline_policy(plan: ReasoningPlan, requested_policy: str) -> str:
    if not plan.reasoning_required:
        return "deterministic_gate"
    if plan.mode != "Active" or not plan.crew_success or plan.fallback_used:
        return requested_policy or "auto"
    try:
        from app.tools.scientific_reasoning_policy_layer import resolve_policy_id

        return resolve_policy_id(plan.primary_school) or plan.primary_school
    except Exception:
        return plan.primary_school


def plan_pipeline_reasoning(
    memory: Any,
    domain_agent: str,
    stage: str,
    *,
    payload: Optional[Mapping[str, Any]] = None,
    objective: str = "",
    requested_policy: str = "auto",
    service: Optional[ReasoningService] = None,
) -> PipelineReasoningDecision:
    packet = build_pipeline_task_packet(
        memory,
        domain_agent,
        stage,
        payload=payload,
        objective=objective,
    )
    mode = normalize_reasoning_mode(_memory_value(memory, "reasoning_mode", "Shadow"))
    plan = (service or ReasoningService(memory=memory)).plan(packet, mode=mode)
    policy = _pipeline_policy(plan, requested_policy)
    context = reasoning_context_from_plan(plan)
    key = re.sub(r"[^a-z0-9]+", "_", stage.lower()).strip("_")
    memory.set_var(f"reasoning_plan_{key}", plan.model_dump(mode="json"))
    memory.set_var(f"reasoning_policy_{key}", policy)
    memory.set_var(f"reasoning_context_{key}", context)
    memory.set_var("reasoning_policy", policy)
    memory.set_var("active_reasoning_context", context)
    return PipelineReasoningDecision(plan=plan, effective_policy=policy, context=context)


def reasoning_context_for_stage(memory: Any, stage: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", stage.lower()).strip("_")
    return str(_memory_value(memory, f"reasoning_context_{key}", "") or "")


def prepare_pipeline_reasoning(
    memory: Any,
    domain_agent: str,
    *,
    payload: Optional[Mapping[str, Any]] = None,
    stage: str = "",
    objective: str = "",
    requested_policy: str = "auto",
    service: Optional[ReasoningService] = None,
) -> tuple[Dict[str, Any], Optional[PipelineReasoningDecision]]:
    body = dict(payload or {})
    resolved_stage = stage or pipeline_stage_for_agent(domain_agent)
    if not should_plan_pipeline_reasoning(domain_agent, body, stage=resolved_stage):
        return body, None
    decision = plan_pipeline_reasoning(
        memory,
        domain_agent,
        resolved_stage,
        payload=body,
        objective=objective,
        requested_policy=requested_policy,
        service=service,
    )
    body["reasoning_policy"] = decision.effective_policy
    body["reasoning_plan"] = decision.plan.model_dump(mode="json")
    body["reasoning_context"] = decision.context
    body["reasoning_stage"] = resolved_stage
    return body, decision
