"""Controlled CrewAI execution for independent bids and manager selection."""

from __future__ import annotations

import json
from typing import Any, List

from app.agents.reasoning.philosophy_definitions import PHILOSOPHY_DEFINITIONS, PHILOSOPHY_ORDER
from app.agents.reasoning.schemas import ReasoningBid, ReasoningPlan, TaskPacket


class PhilosophyCrew:
    def __init__(self, llm: Any):
        self.llm = llm

    def _agent(self, school: str):
        from crewai import Agent

        definition = PHILOSOPHY_DEFINITIONS[school]
        return Agent(
            role=definition["role"],
            goal=definition["goal"],
            backstory=definition["backstory"],
            llm=self.llm,
            allow_delegation=False,
            verbose=False,
            max_iter=1,
        )

    @staticmethod
    def _structured_value(result: Any, task: Any) -> Any:
        task_output = getattr(task, "output", None)
        for candidate in (
            getattr(task_output, "pydantic", None),
            getattr(result, "pydantic", None),
            getattr(task_output, "json_dict", None),
            getattr(result, "json_dict", None),
        ):
            if candidate is not None:
                return candidate
        raw = getattr(task_output, "raw", None) or getattr(result, "raw", None) or str(result)
        return json.loads(raw)

    def _run_bid(self, packet: TaskPacket, school: str) -> ReasoningBid:
        from crewai import Crew, Process, Task

        agent = self._agent(school)
        task = Task(
            description=(
                f"Independently evaluate this TaskPacket only from the {school} epistemic "
                "perspective. Do not solve the scientific task, delegate, or select other schools. "
                f"Return a calibrated bid with the school fixed to '{school}'.\n\n"
                "TaskPacket:\n{task_packet}"
            ),
            expected_output="A validated ReasoningBid for the assigned school.",
            agent=agent,
            output_pydantic=ReasoningBid,
        )
        result = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            memory=False,
            cache=False,
            verbose=False,
            share_crew=False,
        ).kickoff(inputs={"task_packet": packet.model_dump_json(indent=2)})
        bid = ReasoningBid.model_validate(self._structured_value(result, task))
        if bid.school != school:
            bid = bid.model_copy(update={"school": school})
        return bid

    def _run_manager(self, packet: TaskPacket, bids: List[ReasoningBid]) -> ReasoningPlan:
        from crewai import Agent, Crew, Process, Task

        manager = Agent(
            role="POLARIS Epistemic Manager",
            goal=(
                "Choose exactly one primary philosophical reasoning school and at most two useful "
                "supporting schools for the scientific task."
            ),
            backstory=(
                "You route epistemic work, not scientific domain work. Select the minimum useful "
                "set from calibrated specialist bids and assign a precise subtask to each selection."
            ),
            llm=self.llm,
            allow_delegation=False,
            verbose=False,
            max_iter=1,
        )
        task = Task(
            description=(
                "Review the TaskPacket and seven independent ReasoningBids. Do not perform the "
                "experiment, literature review, curve fitting, or ML task. Choose exactly one "
                "primary school, zero to two supporting schools, and the minimum useful set. "
                "Treat the TaskPacket objective as the current scientific task and do not import "
                "entities, materials, or mechanisms absent from that packet. Set task_summary to "
                "the objective, include a nonempty assignment for every selected school, and list "
                "selected school identifiers in reasoning_sequence. Explicit contradictions "
                "normally require Hegelian reasoning, while disputed empirical evidence normally "
                "requires Humean reasoning. Assignments may reference selected schools only.\n\n"
                "TaskPacket:\n{task_packet}\n\nReasoningBids:\n{reasoning_bids}"
            ),
            expected_output="A validated, concise ReasoningPlan.",
            agent=manager,
            output_pydantic=ReasoningPlan,
        )
        result = Crew(
            agents=[manager],
            tasks=[task],
            process=Process.sequential,
            memory=False,
            cache=False,
            verbose=False,
            share_crew=False,
        ).kickoff(
            inputs={
                "task_packet": packet.model_dump_json(indent=2),
                "reasoning_bids": json.dumps(
                    [bid.model_dump(mode="json") for bid in bids], indent=2
                ),
            }
        )
        return ReasoningPlan.model_validate(self._structured_value(result, task))

    def run(self, packet: TaskPacket) -> ReasoningPlan:
        bids = [self._run_bid(packet, school) for school in PHILOSOPHY_ORDER]
        return self._run_manager(packet, bids)
