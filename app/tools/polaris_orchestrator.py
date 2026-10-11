"""Small first-class agent registry/orchestrator for the current POLARIS codebase."""

from __future__ import annotations

import time
from typing import Any, Protocol

from app.tools.agent_contract import AgentResult, AgentTask
from app.tools.literature_agent_service import LiteratureAgentService


class PolarisAgent(Protocol):
    def execute_task(self, task: AgentTask) -> AgentResult: ...


class PolarisOrchestrator:
    def __init__(self):
        self.agents: dict[str, PolarisAgent] = {}
        self.register("literature_agent", LiteratureAgentService())

    def register(self, agent_id: str, agent: PolarisAgent) -> None:
        self.agents[agent_id] = agent

    def dispatch(self, task: AgentTask | dict[str, Any]) -> AgentResult:
        task = task if isinstance(task, AgentTask) else AgentTask(**task)
        agent = self.agents.get(task.agent_id)
        if not agent:
            return AgentResult(
                task_id=task.task_id,
                agent_id=task.agent_id,
                action=task.action,
                status="failed",
                error=f"Unknown POLARIS agent: {task.agent_id}",
            )
        try:
            return agent.execute_task(task)
        except Exception as exc:
            return AgentResult(
                task_id=task.task_id,
                agent_id=task.agent_id,
                action=task.action,
                status="failed",
                error=repr(exc),
            )

    def literature_evidence(
        self,
        query: str,
        limit: int = 5,
        max_chars: int = 6000,
        auto_expand_on_miss: bool = True,
        min_items: int = 1,
        retrieval_max_papers: int = 3,
        wait_for_retrieval: bool = True,
        retrieval_timeout_seconds: int = 1800,
        retrieval_attempts: int = 3,
        require_strong_evidence: bool = False,
    ) -> dict[str, Any]:
        result = self.dispatch(
            AgentTask("literature_agent", "evidence_packet", {"query": query, "limit": limit, "max_chars": max_chars})
        )
        packet = result.data if result.status == "completed" else {"formatted_context": result.error or "Evidence unavailable."}
        if not isinstance(packet, dict):
            packet = {"formatted_context": "Evidence unavailable."}

        provenance = packet.setdefault("provenance", {})
        strong_count = int(provenance.get("strong_evidence_count") or 0)
        contextual_count = int(provenance.get("claim_count") or provenance.get("relevant_evidence_count") or 0)
        if contextual_count == 0:
            contextual_count = int(provenance.get("paper_count") or len(packet.get("papers") or []))
            contextual_count += int(provenance.get("relationship_count") or len(packet.get("relationships") or []))
        item_count = strong_count or contextual_count
        gate_count = strong_count if require_strong_evidence else item_count
        minimum_items = max(1, min_items)
        fresh_retrieval_required = bool(
            auto_expand_on_miss and gate_count < minimum_items
        )
        provenance["evidence_items_used_for_gate"] = gate_count
        provenance["evidence_gate_level"] = "strong" if strong_count else "contextual" if contextual_count else "none"
        provenance["evidence_gate_requirement"] = (
            "strong" if require_strong_evidence else "strong_or_contextual"
        )
        provenance["fresh_retrieval_required"] = fresh_retrieval_required
        if gate_count >= minimum_items and not provenance.get("grounding_status"):
            provenance["grounding_status"] = "grounded" if strong_count else "partially_grounded"

        if fresh_retrieval_required:
            max_attempts = max(1, int(retrieval_attempts))
            retrieval_history: list[dict[str, Any]] = []
            per_attempt_timeout = max(30, int(retrieval_timeout_seconds) // max_attempts)

            for attempt in range(1, max_attempts + 1):
                papers_this_attempt = min(10, max(1, int(retrieval_max_papers)) * attempt)
                retrieval = self.dispatch(
                    AgentTask(
                        "literature_agent",
                        "retrieve_for_hypothesis",
                        {
                            "query": query,
                            "max_papers": papers_this_attempt,
                            "attempt": attempt,
                        },
                    )
                )
                if retrieval.status != "completed" or not isinstance(retrieval.data, dict):
                    retrieval_history.append({
                        "attempt": attempt,
                        "status": "failed_to_start",
                        "error": retrieval.error or "LiteratureAgent retrieval did not start.",
                        "max_papers": papers_this_attempt,
                    })
                    continue

                retrieval_info = {
                    "attempt": attempt,
                    "job_id": retrieval.data.get("job_id"),
                    "status": retrieval.data.get("status"),
                    "already_running": retrieval.data.get("already_running", False),
                    "max_papers": papers_this_attempt,
                }
                retrieval_history.append(retrieval_info)
                provenance["background_retrieval_job"] = retrieval_info
                job_id = retrieval_info.get("job_id")

                if not wait_for_retrieval or not job_id:
                    note = (
                        "\n\nLocal evidence was thin, so POLARIS started a small "
                        f"LiteratureAgent hypothesis-support retrieval job: {job_id}"
                    )
                    context = str(packet.get("formatted_context") or "Evidence unavailable.")
                    context_budget = max(0, max_chars - len(note))
                    packet["formatted_context"] = context[:context_budget] + note
                    break

                deadline = time.monotonic() + per_attempt_timeout
                while time.monotonic() < deadline:
                    status_result = self.dispatch(
                        AgentTask("literature_agent", "job_status", {"job_id": job_id})
                    )
                    if status_result.status != "completed" or not isinstance(status_result.data, dict):
                        retrieval_info["status"] = "status_unavailable"
                        retrieval_info["error"] = status_result.error or f"Could not read job {job_id}"
                        break
                    final_status = status_result.data
                    status = str(final_status.get("status") or "unknown")
                    retrieval_info["status"] = status
                    if status == "completed":
                        break
                    if status in {"failed", "cancelled"}:
                        retrieval_info["error"] = str(final_status.get("log_tail") or "").strip()
                        break
                    time.sleep(2)
                else:
                    retrieval_info["status"] = "timed_out"
                    retrieval_info["error"] = f"Attempt exceeded {per_attempt_timeout} seconds."

                if retrieval_info.get("status") != "completed":
                    continue

                refreshed = self.dispatch(
                    AgentTask(
                        "literature_agent",
                        "evidence_packet",
                        {"query": query, "limit": limit, "max_chars": max_chars},
                    )
                )
                if refreshed.status != "completed" or not isinstance(refreshed.data, dict):
                    retrieval_info["error"] = refreshed.error or "No refreshed evidence packet returned."
                    continue

                packet = refreshed.data
                provenance = packet.setdefault("provenance", {})
                provenance["background_retrieval_job"] = retrieval_info
                refreshed_strong_count = int(provenance.get("strong_evidence_count") or 0)
                refreshed_contextual_count = int(
                    provenance.get("claim_count") or provenance.get("relevant_evidence_count") or 0
                )
                refreshed_count = refreshed_strong_count or refreshed_contextual_count
                refreshed_gate_count = (
                    refreshed_strong_count
                    if require_strong_evidence
                    else refreshed_count
                )
                provenance["evidence_items_used_for_gate"] = refreshed_gate_count
                if refreshed_gate_count >= minimum_items:
                    provenance["evidence_gate_level"] = "strong" if refreshed_strong_count else "contextual"
                    provenance["grounding_status"] = "grounded" if refreshed_strong_count else "partially_grounded"
                    break
            else:
                provenance = packet.setdefault("provenance", {})
                warning = (
                    f"No strongly relevant LiteratureAgent evidence was found after {max_attempts} targeted "
                    "retrieval attempts. Continue as an unverified scientific exploration: contextual B# "
                    "items may motivate the question but must not be presented as direct support, and citations "
                    "must not be invented."
                )
                provenance["grounding_status"] = "unavailable_after_retries"
                provenance["grounding_warning"] = warning
                provenance["background_retrieval_error"] = warning
                existing_context = str(packet.get("formatted_context") or "").strip()
                if existing_context:
                    separator = "\n\nDirect-support boundary: "
                    context_budget = max(0, max_chars - len(separator) - len(warning))
                    packet["formatted_context"] = (
                        existing_context[:context_budget].rstrip() + separator + warning
                    )[:max_chars]
                else:
                    packet["formatted_context"] = f"No strong direct local support. {warning}"[:max_chars]

            provenance = packet.setdefault("provenance", {})
            provenance["retrieval_attempts"] = retrieval_history
            provenance["retrieval_attempt_count"] = len(retrieval_history)
            provenance["fresh_retrieval_performed"] = bool(retrieval_history)

        provenance = packet.setdefault("provenance", {})
        provenance["evidence_gate_requirement"] = (
            "strong" if require_strong_evidence else "strong_or_contextual"
        )
        provenance["fresh_retrieval_required"] = fresh_retrieval_required
        provenance.setdefault("fresh_retrieval_performed", False)

        if packet.get("formatted_context"):
            packet["formatted_context"] = str(packet["formatted_context"])[:max_chars]
        return packet
