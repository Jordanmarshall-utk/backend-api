"""Stable, failure-safe adapter between POLARIS and CrewAI."""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional

from app.agents.reasoning.philosophy_definitions import PHILOSOPHY_DEFINITIONS
from app.agents.reasoning.schemas import Philosophy, ReasoningMode, ReasoningPlan, TaskPacket

logger = logging.getLogger(__name__)
CrewRunner = Callable[[TaskPacket], Any]


def normalize_reasoning_mode(value: Any) -> ReasoningMode:
    normalized = str(value or "Shadow").strip().lower()
    return {"legacy": "Legacy", "active": "Active"}.get(normalized, "Shadow")


class ReasoningService:
    def __init__(self, memory: Any = None, crew_runner: Optional[CrewRunner] = None):
        self.memory = memory
        self.crew_runner = crew_runner

    @staticmethod
    def needs_reasoning(packet: TaskPacket) -> bool:
        text = f"{packet.domain_agent} {packet.stage} {packet.objective}".lower()
        if "hypothesis" in text:
            return True
        deterministic = (
            "calculate",
            "convert units",
            "sort rows",
            "export csv",
            "plot values",
            "curve fitting",
            "curve_fitting",
            "fit spectrum",
            "peak fitting",
        )
        return not any(marker in text for marker in deterministic)

    @staticmethod
    def _fallback_selection(packet: TaskPacket) -> tuple[Philosophy, List[Philosophy]]:
        text = " ".join(
            [
                packet.objective,
                packet.stage,
                " ".join(packet.evidence),
                " ".join(packet.assumptions),
                " ".join(packet.constraints),
                " ".join(packet.contradictions),
            ]
        ).lower()
        if packet.contradictions or any(x in text for x in ("conflict", "discrepancy", "contradict")):
            return "hegelian", ["humean", "socratic"]
        if packet.constraints or any(x in text for x in ("constraint", "compliance", "safety", "satisfies")):
            return "kantian", ["cartesian"]
        if any(x in text for x in ("next three experiments", "next experiments", "decompose", "sequence", "steps needed")):
            return "cartesian", ["humean", "kantian"]
        if any(x in text for x in ("transferable", "generaliz", "invariant", "shared principle")):
            return "platonic", ["humean", "aristotelian"]
        if any(x in text for x in ("evidence", "empirical", "supports the claim", "causal claim", "falsif")):
            return "humean", ["socratic"]
        if any(x in text for x in ("classif", "mechanism", "properties", "relationship")):
            return "aristotelian", ["humean"]
        return "socratic", []

    def deterministic_plan(
        self,
        packet: TaskPacket,
        *,
        mode: ReasoningMode,
        error: Optional[str] = None,
        source: str = "deterministic_fallback",
        reasoning_required: bool = True,
    ) -> ReasoningPlan:
        primary, supporting = self._fallback_selection(packet)
        selected = [primary, *supporting]
        assignments: Dict[str, str] = {
            school: PHILOSOPHY_DEFINITIONS[school]["goal"] for school in selected
        }
        return ReasoningPlan(
            primary_school=primary,
            supporting_schools=supporting,
            assignments=assignments,
            rationale=(
                "Deterministic safety selection based on task structure; the existing POLARIS "
                "reasoning path remains authoritative."
            ),
            confidence=0.65 if error else 0.72,
            reasoning_required=reasoning_required,
            task_summary=packet.objective[:500],
            reasoning_sequence=selected,
            mode=mode,
            selection_source=source,
            crew_success=False,
            fallback_used=bool(error),
            error=error,
            warnings=["CrewAI fallback used; scientific output was not changed."] if error else [],
        )

    @staticmethod
    def _grounded_items(items: List[str], packet: TaskPacket) -> List[str]:
        reference = " ".join(
            [
                packet.objective,
                *packet.evidence,
                *packet.assumptions,
                *packet.constraints,
                *packet.contradictions,
            ]
        ).lower()
        stopwords = {
            "about", "after", "before", "could", "from", "have", "into", "might",
            "other", "should", "their", "there", "these", "this", "those", "what",
            "when", "where", "which", "with", "would", "hypothesis", "evidence",
            "experiment", "experiments", "scientific", "task",
        }
        reference_tokens = {
            token for token in re.findall(r"[a-z0-9]+", reference)
            if len(token) >= 4 and token not in stopwords
        }
        grounded: List[str] = []
        for item in items:
            item_tokens = {
                token for token in re.findall(r"[a-z0-9]+", str(item).lower())
                if len(token) >= 4 and token not in stopwords
            }
            if len(reference_tokens.intersection(item_tokens)) >= 2:
                grounded.append(str(item))
        return grounded

    @classmethod
    def _ground_crewai_plan(cls, packet: TaskPacket, plan: ReasoningPlan) -> ReasoningPlan:
        selected = [plan.primary_school, *plan.supporting_schools]
        assignments = {
            school: (
                str(plan.assignments.get(school) or "").strip()
                or f"{PHILOSOPHY_DEFINITIONS[school]['goal']} Apply this to: {packet.objective}"
            )
            for school in selected
        }
        unresolved = cls._grounded_items(plan.unresolved_questions, packet)
        evidence = cls._grounded_items(plan.evidence_requirements, packet)
        warnings = list(plan.warnings)
        removed_count = (
            len(plan.unresolved_questions) - len(unresolved)
            + len(plan.evidence_requirements) - len(evidence)
        )
        if removed_count:
            warnings.append(
                f"Removed {removed_count} council detail(s) not grounded in the current TaskPacket."
            )
        return plan.model_copy(
            update={
                "task_summary": packet.objective,
                "assignments": assignments,
                "reasoning_sequence": selected,
                "unresolved_questions": unresolved,
                "evidence_requirements": evidence,
                "warnings": warnings,
            }
        )

    def _run_crewai(self, packet: TaskPacket) -> ReasoningPlan:
        from app.agents.reasoning.llm_adapter import build_crewai_llm
        from app.agents.reasoning.philosophy_crew import PhilosophyCrew

        return PhilosophyCrew(build_crewai_llm(self.memory)).run(packet)

    def _store(self, plan: ReasoningPlan, packet: TaskPacket) -> None:
        if self.memory is None:
            return
        payload = plan.model_dump(mode="json")
        try:
            self.memory.set_var("reasoning_mode", plan.mode)
            self.memory.set_var("last_reasoning_plan", payload)
            self.memory.log_event(
                "reasoning_plan",
                {
                    "domain_agent": packet.domain_agent,
                    "stage": packet.stage,
                    "objective": packet.objective[:500],
                    "primary_school": plan.primary_school,
                    "supporting_schools": plan.supporting_schools,
                    "confidence": plan.confidence,
                    "assignments": plan.assignments,
                    "mode": plan.mode,
                    "crew_success": plan.crew_success,
                    "fallback_used": plan.fallback_used,
                    "error": plan.error,
                },
                mode="reasoning",
            )
        except Exception:
            logger.exception("Unable to persist POLARIS reasoning plan")

    def plan(self, packet: TaskPacket, mode: Any = None) -> ReasoningPlan:
        configured = mode
        if configured is None and self.memory is not None:
            try:
                configured = self.memory.get_var("reasoning_mode", "Shadow")
            except Exception:
                configured = "Shadow"
        resolved_mode = normalize_reasoning_mode(configured)

        if resolved_mode == "Legacy":
            plan = self.deterministic_plan(
                packet, mode=resolved_mode, source="legacy", reasoning_required=False
            )
            self._store(plan, packet)
            return plan
        if not self.needs_reasoning(packet):
            plan = self.deterministic_plan(
                packet, mode=resolved_mode, source="gate", reasoning_required=False
            )
            self._store(plan, packet)
            return plan

        try:
            raw = (self.crew_runner or self._run_crewai)(packet)
            grounded = self._ground_crewai_plan(packet, ReasoningPlan.model_validate(raw))
            plan = grounded.model_copy(
                update={
                    "mode": resolved_mode,
                    "selection_source": "crewai",
                    "crew_success": True,
                    "fallback_used": False,
                    "error": None,
                }
            )
        except Exception as exc:
            logger.warning("CrewAI Reasoning Council failed; using legacy-safe fallback: %s", exc)
            plan = self.deterministic_plan(
                packet, mode=resolved_mode, error=f"{type(exc).__name__}: {exc}"
            )
        self._store(plan, packet)
        return plan
