"""SSE streaming for progressive hypothesis chat updates."""

from __future__ import annotations

import json
from typing import Any, Dict, Generator, Iterator, List, Optional

from app.services.hypothesis_chat import (
    _insert_handoff_bubble,
    _literature_handoff_bubble,
    choose_option,
    generate_hypothesis,
    reset_session,
)
from app.services.hypothesis_messages import (
    build_submit_bubbles,
    join_bubbles_for_legacy,
)
from app.services.hypothesis_perf import (
    fast_submit_enabled,
    skip_socratic_answers,
)
from app.services.llm_runtime import require_api_key
from app.services.reasoning_integration import (
    plan_hypothesis_reasoning,
)
from app.tools import socratic


def format_sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _progress(step: str, messages: List[Dict[str, str]], **extra: Any) -> Dict[str, Any]:
    return {
        "type": "progress",
        "step": step,
        "messages": messages,
        **extra,
    }


def _complete(result: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "complete", **result}


def iter_submit_question(memory: Any, question: str, reasoning_policy: str = "auto") -> Iterator[Dict[str, Any]]:
    require_api_key(memory)
    q = question.strip()
    if not q:
        raise ValueError("Question is required.")

    policy = reasoning_policy or "auto"
    yield _progress(
        "reasoning_council",
        [],
        label="Consulting the POLARIS Reasoning Council...",
    )
    decision = plan_hypothesis_reasoning(
        memory,
        q,
        "hypothesis_generation",
        requested_policy=policy,
    )
    policy = decision.effective_policy
    yield _progress(
        "reasoning_council",
        [],
        label=f"Reasoning plan: {decision.plan.primary_school}",
        reasoning_policy=policy,
        reasoning_plan=decision.plan.model_dump(mode="json"),
    )

    clarified = socratic.clarify_question(q, reasoning_policy=policy)
    if not clarified or not str(clarified).strip():
        raise ValueError("Could not generate clarified question. Check your API key in Settings.")

    yield _progress(
        "clarify",
        build_submit_bubbles(clarified, "", None, [])[:1],
        reasoning_policy=policy,
    )

    soc_answers: Optional[str] = None
    soc_pass: Optional[str] = None
    thoughts: Optional[List[str]] = None

    if fast_submit_enabled():
        yield _progress(
            "literature_retrieval",
            [],
            label="Waiting for LiteratureAgent evidence packet before generating suggestions…",
        )
        soc_pass, thoughts = socratic.socratic_pass_and_tot(
            clarified, reasoning_policy=policy, evidence_query=q
        )
        if soc_pass and str(soc_pass).strip():
            yield _progress(
                "socratic_pass",
                [
                    {
                        "role": "assistant",
                        "title": "Socratic pass (probing questions)",
                        "content": str(soc_pass).strip(),
                    }
                ],
            )
    elif skip_socratic_answers():
        yield _progress("socratic_pass", [], label="Generating Socratic pass…")
        soc_pass = socratic.socratic_pass(clarified, reasoning_policy=policy)
        if not soc_pass or not str(soc_pass).strip():
            raise ValueError("Could not generate socratic questions.")
        soc_answers = ""
        yield _progress(
            "socratic_pass",
            [{"role": "assistant", "title": "Socratic pass (probing questions)", "content": str(soc_pass)}],
        )
        yield _progress("thoughts", [], label="Generating three exploration paths…")
        thoughts = socratic.tot_generation(soc_pass, clarified, None, reasoning_policy=policy)
    else:
        yield _progress("socratic_pass", [], label="Generating Socratic pass…")
        soc_pass = socratic.socratic_pass(clarified, reasoning_policy=policy)
        if not soc_pass or not str(soc_pass).strip():
            raise ValueError("Could not generate socratic questions.")
        yield _progress(
            "socratic_pass",
            [{"role": "assistant", "title": "Socratic pass (probing questions)", "content": str(soc_pass)}],
        )
        yield _progress("socratic_answers", [], label="Generating Socratic reasoning…")
        soc_answers = socratic.socratic_answer_questions(clarified, soc_pass, reasoning_policy=policy)
        if soc_answers and str(soc_answers).strip():
            yield _progress(
                "socratic_answers",
                [{"role": "assistant", "title": "Socratic reasoning", "content": str(soc_answers)}],
            )
        yield _progress("thoughts", [], label="Generating three exploration paths…")
        thoughts = socratic.tot_generation(soc_pass, clarified, soc_answers, reasoning_policy=policy)

    if not soc_pass or not str(soc_pass).strip():
        raise ValueError("Could not generate socratic questions.")
    if not thoughts:
        raise ValueError("Could not generate exploration options.")

    thoughts = list(thoughts)[:3]
    while len(thoughts) < 3:
        thoughts.append(f"Option {len(thoughts) + 1}: Continue exploring")

    handoff_bubble = _literature_handoff_bubble(memory)
    if handoff_bubble:
        yield _progress("literature_handoff", [handoff_bubble])

    thought_bubbles = []
    for i, thought in enumerate(thoughts[:3], 1):
        if thought and str(thought).strip():
            thought_bubbles.append(
                {"role": "assistant", "title": f"Generated thought {i}", "content": str(thought).strip()}
            )
    if thought_bubbles:
        yield _progress("thoughts", thought_bubbles)

    memory.insert_interaction("user", q, "initial_question", "hypothesis")
    memory.insert_interaction("assistant", clarified, "clarified_question", "hypothesis")
    memory.insert_interaction("assistant", soc_pass, "socratic_pass", "hypothesis")
    if soc_answers and str(soc_answers).strip():
        memory.insert_interaction("assistant", soc_answers, "socratic_answers", "hypothesis")
    memory.insert_interaction("assistant", thoughts[0], "first_thought_1", "hypothesis")
    memory.insert_interaction("assistant", thoughts[1], "second_thought_1", "hypothesis")
    memory.insert_interaction("assistant", thoughts[2], "third_thought_1", "hypothesis")
    if handoff_bubble:
        memory.insert_interaction(
            "assistant", handoff_bubble["content"], "literature_agent_handoff", "hypothesis"
        )

    memory.set_var("hypothesis_round_count", 0)
    memory.set_var("stage", "refine")

    from app.services.hypothesis_chat import _bump_usage

    _bump_usage(memory)

    bubbles = build_submit_bubbles(clarified, soc_pass, soc_answers, thoughts)
    _insert_handoff_bubble(bubbles, handoff_bubble)
    yield _complete(
        {
            "stage": "refine",
            "messages": bubbles,
            "assistant_message": join_bubbles_for_legacy(bubbles),
            "options": thoughts[:3],
            "reasoning_plan": decision.plan.model_dump(mode="json"),
        }
    )


def iter_generate_hypothesis(memory: Any) -> Iterator[Dict[str, Any]]:
    # Keep streaming and non-streaming evidence handling on one implementation.
    yield _progress("synthesize", [], label="Synthesizing hypothesis...")
    result = generate_hypothesis(memory)
    for bubble in result.get("messages") or []:
        title = str(bubble.get("title") or "").lower()
        if "literatureagent" in title:
            step = "literature_handoff"
        elif "analysis" in title:
            step = "analysis"
        else:
            step = "hypothesis"
        yield _progress(step, [bubble])
    yield _complete(result)
    return


def iter_handle_chat_stream(
    memory: Any,
    *,
    action: str,
    question: Optional[str] = None,
    choice: Optional[str] = None,
    reasoning_policy: str = "auto",
) -> Iterator[Dict[str, Any]]:
    if action == "reset":
        yield _complete(reset_session(memory))
        return
    if action == "submit_question":
        if not question:
            raise ValueError("question is required for submit_question")
        yield from iter_submit_question(memory, question, reasoning_policy=reasoning_policy)
        return
    if action == "generate_hypothesis":
        yield from iter_generate_hypothesis(memory)
        return
    if action == "choose_option":
        if not choice:
            raise ValueError("choice is required for choose_option")
        stage = memory.get_var("stage")
        label = "Generating continuation options…" if stage == "hypothesis" else "Deepening selected path…"
        yield _progress("choose", [], label=label)
        result = choose_option(memory, choice)
        yield _complete(result)
        return
    raise ValueError(f"Unknown action: {action}")


def iter_sse_events(
    memory: Any,
    *,
    action: str,
    question: Optional[str] = None,
    choice: Optional[str] = None,
    reasoning_policy: str = "auto",
) -> Generator[str, None, None]:
    try:
        for event in iter_handle_chat_stream(memory, action=action, question=question, choice=choice, reasoning_policy=reasoning_policy):
            if event.get("type") == "progress":
                yield format_sse(
                    "progress",
                    {
                        "step": event.get("step"),
                        "messages": event.get("messages", []),
                        "label": event.get("label"),
                        "reasoning_policy": event.get("reasoning_policy"),
                        "reasoning_plan": event.get("reasoning_plan"),
                    },
                )
            elif event.get("type") == "complete":
                payload = {k: v for k, v in event.items() if k != "type"}
                yield format_sse("complete", payload)
        yield format_sse("done", {})
    except ValueError as exc:
        yield format_sse("error", {"error": str(exc)})
    except Exception as exc:
        yield format_sse("error", {"error": str(exc) or "Hypothesis agent failed"})
