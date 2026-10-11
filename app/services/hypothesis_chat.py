"""Headless hypothesis agent flow for REST clients (no Streamlit)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.agents.hypothesis_agent import HypothesisAgent
from app.services.hypothesis_messages import (
    build_choose_bubbles,
    build_continue_bubbles,
    build_hypothesis_result_bubbles,
    build_submit_bubbles,
    join_bubbles_for_legacy,
)
from app.services.hypothesis_perf import (
    fast_submit_enabled,
    run_in_background,
    skip_analysis_on_generate,
    skip_readiness_check,
    skip_socratic_answers,
    trim_context,
)
from app.services.document_export import export_agent_document
from app.services.llm_runtime import require_api_key
from app.services.reasoning_integration import (
    plan_hypothesis_reasoning,
    reasoning_context_for_synthesis,
)
from app.tools import socratic
from app.tools.instruct import HYPOTHESIS_READINESS_CHECK
from app.tools.mcp_orchestrator_bridge import sync_hypothesis_proposal


def _ensure_api_key(memory: Any) -> None:
    require_api_key(memory)


def _bump_usage(memory: Any) -> None:
    counts = memory.get_var("agent_usage_counts") or {}
    if not isinstance(counts, dict):
        counts = {}
    counts["hypothesis"] = int(counts.get("hypothesis", 0)) + 1
    memory.set_var("agent_usage_counts", counts)


def _agent(memory: Any) -> HypothesisAgent:
    agent = HypothesisAgent("Hypothesis Agent", "API", "")
    agent.memory = memory
    return agent


def _next_step_options(memory: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    return (
        memory.view_component("next_step_option_1"),
        memory.view_component("next_step_option_2"),
        memory.view_component("next_step_option_3"),
    )


def _normalize_options(options: Any) -> List[str]:
    if not isinstance(options, list):
        options = [str(o) for o in options] if options else []
    options = [
        opt
        for opt in options
        if opt and str(opt).strip() and str(opt).strip().lower() != "none"
    ]
    while len(options) < 3:
        options.append(f"Option {len(options) + 1}: Continue exploring")
    return options[:3]


def _resolve_choice(memory: Any, choice: str, stage: Optional[str] = None) -> str:
    """Resolve a numeric choice or an exact displayed option without silent fallback."""
    selected = str(choice or "").strip()
    if selected in ("1", "2", "3"):
        return selected

    current_stage = stage or str(memory.get_var("stage") or "")
    if current_stage == "refine":
        options = (
            memory.view_component("first_thought_1"),
            memory.view_component("second_thought_1"),
            memory.view_component("third_thought_1"),
        )
    elif current_stage == "hypothesis":
        options = _next_step_options(memory)
    else:
        raise ValueError(f"Cannot choose option in stage '{current_stage}'.")

    exact_matches = [
        str(index)
        for index, option in enumerate(options, 1)
        if option is not None and str(option).strip() == selected
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    raise ValueError("Choice must be 1, 2, 3, or exactly match one displayed option.")


def _available_packet_reference_note(handoff: Dict[str, Any]) -> str:
    claim_ids = [str(item) for item in handoff.get("claim_ids") or [] if str(item).strip()]
    source_ids = [str(item) for item in handoff.get("source_ids") or [] if str(item).strip()]
    if not claim_ids and not source_ids:
        return " No packet source identifier was detected in the generated response."
    parts: List[str] = []
    if claim_ids:
        parts.append("claims " + ", ".join(f"[{item}]" for item in claim_ids[:8]))
    if source_ids:
        parts.append("sources " + ", ".join(f"`{item}`" for item in source_ids[:5]))
    return (
        " The generated options did not cite a packet source and therefore remain unsupported. "
        "Available packet references for manual review (not candidate-specific validation): "
        + "; ".join(parts)
        + "."
    )


def _literature_handoff_bubble(
    memory: Any = None,
    selected_scope: str = "",
) -> Optional[Dict[str, str]]:
    handoff = socratic.get_last_literature_handoff()
    if not handoff:
        return None
    if selected_scope:
        handoff["selected_scope"] = str(selected_scope)
        socratic.set_last_literature_handoff(handoff)
    if memory is not None:
        memory.set_var("last_literature_handoff", handoff)
    handoff_status = str(handoff.get("status") or "")
    retrieval = handoff.get("background_retrieval_job")
    grounding_status = str(handoff.get("grounding_status") or "")
    retrieval_query = str(handoff.get("retrieval_query") or handoff.get("query") or "").strip()
    scope_note = (
        "\n\n**Evidence scope:** " + retrieval_query[:320]
        if retrieval_query else ""
    )
    if handoff_status == "deferred_until_branch_selection":
        return {
            "role": "assistant",
            "title": "LiteratureAgent evidence status",
            "content": (
                "Literature retrieval has not run for these thoughts yet. Treat every "
                "option as an unverified proposal until a branch-specific evidence "
                "packet is retrieved."
            ),
        }
    if grounding_status == "unavailable_after_retries":
        attempts = int(handoff.get("retrieval_attempt_count") or 0)
        background_count = int(handoff.get("background_claim_count") or 0)
        background_sources = list(handoff.get("source_ids") or [])
        background_note = ""
        if background_count:
            background_note = (
                f" LiteratureAgent retained {background_count} related background claim(s); these can motivate "
                "the question but do not directly validate the selected proposal."
            )
            if background_sources:
                background_note += " Background source(s): " + ", ".join(
                    f"`{item}`" for item in background_sources[:5]
                ) + "."
        return {
            "role": "assistant",
            "title": "LiteratureAgent grounding unavailable",
            "content": (
                f"LiteratureAgent completed {attempts} targeted retrieval attempt(s), but no strongly relevant "
                "evidence packet was found. The exploration options were still generated using the original "
                "Hypothesis Agent workflow. Treat them as unverified proposals, not literature-supported claims."
                + background_note
                + scope_note
            ),
        }
    if grounding_status == "partially_grounded":
        used_sources = list(handoff.get("used_source_ids") or [])
        return {
            "role": "assistant",
            "title": "LiteratureAgent partial grounding",
            "content": (
                "LiteratureAgent found evidence relevant to part of the question, but not direct proof of the full "
                "proposed mechanism. The exploration options distinguish that evidence from inference."
                + (
                    "\n\nEvidence cited in the response: "
                    + ", ".join(f"`{item}`" for item in used_sources)
                    + "."
                    if used_sources else
                    "\n\n" + _available_packet_reference_note(handoff).lstrip()
                )
                + scope_note
            ),
        }
    if isinstance(retrieval, dict) and retrieval.get("job_id"):
        status = retrieval.get("status") or "queued"
        if status == "completed":
            used_sources = list(handoff.get("used_source_ids") or [])
            usage_note = (
                "\n\n**Evidence used:** " + ", ".join(f"`{item}`" for item in used_sources)
                if used_sources else
                "\n\n" + _available_packet_reference_note(handoff).lstrip()
            )
            content = (
                "The original local packet did not contain enough direct support. The Hypothesis Agent "
                "paused while LiteratureAgent completed a targeted search and rebuilt the evidence packet. "
                "The exploration options below were generated only after that packet was available.\n\n"
                f"**Job:** `{retrieval['job_id']}`  \n"
                "**Status:** `completed`"
                + usage_note
                + scope_note
            )
        else:
            content = (
                "The Hypothesis Agent is waiting for LiteratureAgent to finish a targeted search before "
                "generating exploration options.\n\n"
                f"**Job:** `{retrieval['job_id']}`  \n"
                f"**Status:** `{status}`"
                + scope_note
            )
        return {"role": "assistant", "title": "LiteratureAgent handoff", "content": content}
    strong_count = int(handoff.get("strong_evidence_count") or 0)
    if strong_count > 0:
        used_sources = list(handoff.get("used_source_ids") or [])
        return {
            "role": "assistant",
            "title": "LiteratureAgent grounding",
            "content": (
                f"Found {strong_count} directly relevant local evidence item(s); no new search was required. "
                + (
                    "Evidence cited in the response: " + ", ".join(f"`{item}`" for item in used_sources) + "."
                    if used_sources else
                    _available_packet_reference_note(handoff).lstrip()
                )
                + scope_note
            ),
        }
    if handoff.get("background_retrieval_error") or handoff.get("error"):
        return {
            "role": "assistant",
            "title": "LiteratureAgent handoff",
            "content": "Direct local evidence was insufficient, but the targeted literature search could not be started. Check the Literature Agent page for details.",
        }
    return {
        "role": "assistant",
        "title": "LiteratureAgent evidence status",
        "content": (
            "No directly supporting literature was identified for the current evidence scope. "
            "Treat the generated thoughts as exploratory proposals rather than literature-supported claims."
            + scope_note
        ),
    }


def _insert_handoff_bubble(bubbles: List[Dict[str, str]], bubble: Optional[Dict[str, str]]) -> None:
    if not bubble:
        return
    insert_at = 1 if bubbles else 0
    bubbles.insert(insert_at, bubble)


def _full_context(agent: HypothesisAgent) -> str:
    context = agent.build_conversation_context()
    negative_ctx = agent._build_negative_hypotheses_context()
    if negative_ctx:
        merged = negative_ctx + "\n\n--- Current conversation ---\n\n" + context
        return trim_context(merged)
    return trim_context(context)


def submit_question(memory: Any, question: str, reasoning_policy: str = "auto") -> Dict[str, Any]:
    _ensure_api_key(memory)
    q = question.strip()
    if not q:
        raise ValueError("Question is required.")

    policy = reasoning_policy or "auto"
    decision = plan_hypothesis_reasoning(
        memory,
        q,
        "hypothesis_generation",
        requested_policy=policy,
    )
    policy = decision.effective_policy

    clarified = socratic.clarify_question(q, reasoning_policy=policy)
    if not clarified or not str(clarified).strip():
        raise ValueError("Could not generate clarified question. Check your API key in Settings.")

    soc_answers: Optional[str] = None
    if fast_submit_enabled():
        soc_pass, thoughts = socratic.socratic_pass_and_tot(
            clarified,
            reasoning_policy=policy,
            evidence_query=q,
        )
    elif skip_socratic_answers():
        soc_pass = socratic.socratic_pass(clarified, reasoning_policy=policy)
        if not soc_pass or not str(soc_pass).strip():
            raise ValueError("Could not generate socratic questions.")
        soc_answers = ""
        thoughts = socratic.tot_generation(soc_pass, clarified, None, reasoning_policy=policy)
    else:
        soc_pass = socratic.socratic_pass(clarified, reasoning_policy=policy)
        if not soc_pass or not str(soc_pass).strip():
            raise ValueError("Could not generate socratic questions.")
        soc_answers = socratic.socratic_answer_questions(
            clarified, soc_pass, reasoning_policy=policy
        )
        thoughts = socratic.tot_generation(soc_pass, clarified, soc_answers, reasoning_policy=policy)

    if not soc_pass or not str(soc_pass).strip():
        raise ValueError("Could not generate socratic questions.")
    if not thoughts:
        raise ValueError("Could not generate exploration options.")

    thoughts = list(thoughts)[:3]
    while len(thoughts) < 3:
        thoughts.append(f"Option {len(thoughts) + 1}: Continue exploring")

    memory.insert_interaction("user", q, "initial_question", "hypothesis")
    memory.insert_interaction("assistant", clarified, "clarified_question", "hypothesis")
    memory.insert_interaction("assistant", soc_pass, "socratic_pass", "hypothesis")
    if soc_answers and str(soc_answers).strip():
        memory.insert_interaction("assistant", soc_answers, "socratic_answers", "hypothesis")
    memory.insert_interaction("assistant", thoughts[0], "first_thought_1", "hypothesis")
    memory.insert_interaction("assistant", thoughts[1], "second_thought_1", "hypothesis")
    memory.insert_interaction("assistant", thoughts[2], "third_thought_1", "hypothesis")

    memory.set_var("hypothesis_round_count", 0)
    memory.set_var("stage", "refine")
    _bump_usage(memory)

    handoff_bubble = _literature_handoff_bubble(memory)
    if handoff_bubble:
        memory.insert_interaction(
            "assistant", handoff_bubble["content"], "literature_agent_handoff", "hypothesis"
        )
    bubbles = build_submit_bubbles(clarified, soc_pass, soc_answers, thoughts)
    _insert_handoff_bubble(bubbles, handoff_bubble)
    return {
        "stage": "refine",
        "messages": bubbles,
        "assistant_message": join_bubbles_for_legacy(bubbles),
        "options": thoughts[:3],
        "reasoning_plan": decision.plan.model_dump(mode="json"),
    }


def _choose_refine_option(memory: Any, choice: str) -> Dict[str, Any]:
    if choice == "1":
        picked = memory.view_component("first_thought_1")
        prev1 = memory.view_component("second_thought_1")
        prev2 = memory.view_component("third_thought_1")
    elif choice == "2":
        picked = memory.view_component("second_thought_1")
        prev1 = memory.view_component("first_thought_1")
        prev2 = memory.view_component("third_thought_1")
    else:
        picked = memory.view_component("third_thought_1")
        prev1 = memory.view_component("first_thought_1")
        prev2 = memory.view_component("second_thought_1")

    clarified = memory.view_component("clarified_question") or memory.view_component("initial_question")
    context = trim_context(_agent(memory).build_conversation_context())
    reasoning_policy = str(memory.get_var("reasoning_policy") or "auto")
    result = socratic.retry_thinking_deepen_thoughts(picked, prev1, prev2, clarified, context, reasoning_policy=reasoning_policy)

    if result is None or len(result) != 2:
        soc_q = "How can we continue exploring this hypothesis?"
        options = [
            "Why is this approach theoretically sound?",
            "What are the mechanistic advantages?",
            "How does this compare to alternatives?",
        ]
    else:
        soc_q, options = result
        options = _normalize_options(options)

    memory.insert_interaction("user", choice, "tot_choice", "hypothesis")
    memory.insert_interaction(
        "assistant",
        soc_q if soc_q and str(soc_q).strip().lower() != "none" else "How can we continue exploring this hypothesis?",
        "retry_thinking_question",
        "hypothesis",
    )
    for i, opt in enumerate(options[:3], 1):
        memory.insert_interaction("assistant", opt, f"next_step_option_{i}", "hypothesis")

    memory.set_var("hypothesis_round_count", 0)
    memory.set_var("stage", "hypothesis")

    soc_q_display = (
        soc_q
        if soc_q and str(soc_q).strip().lower() != "none"
        else "How can we continue exploring this hypothesis?"
    )
    bubbles = build_choose_bubbles(str(picked), soc_q_display, options[:3])
    handoff_bubble = _literature_handoff_bubble(memory, str(picked))
    _insert_handoff_bubble(bubbles, handoff_bubble)
    if handoff_bubble:
        memory.insert_interaction(
            "assistant", handoff_bubble["content"], "literature_agent_continuation_handoff", "hypothesis"
        )
    return {
        "stage": "hypothesis",
        "messages": bubbles,
        "assistant_message": join_bubbles_for_legacy(bubbles),
        "options": options[:3],
    }


def _check_hypothesis_ready(memory: Any, *, picked: str, prev1: str, prev2: str, round_count: int) -> bool:
    if skip_readiness_check():
        max_rounds = int(memory.get_var("max_hypothesis_rounds", 5) or 5)
        return round_count + 1 >= max_rounds
    max_rounds = int(memory.get_var("max_hypothesis_rounds", 5) or 5)
    if round_count + 1 >= max_rounds:
        return True
    clarified_q = memory.view_component("clarified_question") or ""
    socratic_q = memory.view_component("socratic_pass") or ""
    previous_opts = f"{prev1}; {prev2}"
    readiness_prompt = HYPOTHESIS_READINESS_CHECK.format(
        clarified_question=str(clarified_q)[:500],
        socratic_questions=str(socratic_q)[:500],
        round_count=round_count + 1,
        selected_option=str(picked)[:300],
        previous_options=previous_opts[:300],
    )
    try:
        readiness_response = socratic.generate_text_with_llm(readiness_prompt).strip().upper()
        return "READY" in readiness_response
    except Exception:
        return (round_count + 1) >= max_rounds


def _choose_hypothesis_next_step(memory: Any, choice: str) -> Dict[str, Any]:
    opt1, opt2, opt3 = _next_step_options(memory)
    options_map = {"1": opt1, "2": opt2, "3": opt3}
    picked = options_map[choice]
    if not picked or not str(picked).strip():
        raise ValueError(f"Option {choice} is not available.")

    prev = [opt1, opt2, opt3]
    prev = [p for p in prev if p != picked]
    while len(prev) < 2:
        prev.append(f"Previous option {len(prev) + 1}")

    round_count = int(memory.get_var("hypothesis_round_count", 0) or 0)
    max_rounds = int(memory.get_var("max_hypothesis_rounds", 5) or 5)
    if round_count >= max_rounds:
        return generate_hypothesis(memory)

    memory.insert_interaction("user", choice, "option_choice", "hypothesis")
    memory.insert_interaction("assistant", picked, "last_selected_option", "hypothesis")
    memory.insert_interaction("assistant", prev[0], "last_prev1", "hypothesis")
    memory.insert_interaction("assistant", prev[1], "last_prev2", "hypothesis")

    if _check_hypothesis_ready(
        memory, picked=str(picked), prev1=str(prev[0]), prev2=str(prev[1]), round_count=round_count
    ):
        return generate_hypothesis(memory)

    agent = _agent(memory)
    context = trim_context(agent.build_conversation_context())
    clarified = memory.view_component("clarified_question")
    reasoning_policy = str(memory.get_var("reasoning_policy") or "auto")
    result = socratic.retry_thinking_deepen_thoughts(
        picked, prev[0], prev[1], clarified, context, reasoning_policy=reasoning_policy
    )

    if result is None or len(result) != 2:
        new_opts = _normalize_options([])
    else:
        _, new_opts = result
        new_opts = _normalize_options(new_opts)

    for i, opt in enumerate(new_opts[:3], 1):
        memory.insert_interaction("assistant", opt, f"next_step_option_{i}", "hypothesis")

    memory.set_var("hypothesis_round_count", round_count + 1)
    memory.set_var("stage", "hypothesis")

    bubbles = build_continue_bubbles(str(picked), new_opts[:3])
    handoff_bubble = _literature_handoff_bubble(memory, str(picked))
    _insert_handoff_bubble(bubbles, handoff_bubble)
    if handoff_bubble:
        memory.insert_interaction(
            "assistant", handoff_bubble["content"], "literature_agent_continuation_handoff", "hypothesis"
        )
    return {
        "stage": "hypothesis",
        "messages": bubbles,
        "assistant_message": join_bubbles_for_legacy(bubbles),
        "options": new_opts[:3],
    }


def choose_option(memory: Any, choice: str) -> Dict[str, Any]:
    _ensure_api_key(memory)
    stage = memory.get_var("stage")
    choice = _resolve_choice(memory, choice, str(stage or ""))
    if stage == "refine":
        return _choose_refine_option(memory, choice)
    if stage == "hypothesis":
        return _choose_hypothesis_next_step(memory, choice)
    raise ValueError(f"Cannot choose option in stage '{stage}'.")


def _hypothesis_document_fields(memory: Any, hypothesis: str, analysis_rubric: Optional[str]) -> Dict[str, str]:
    parts = [f"# Hypothesis\n\n{str(hypothesis).strip()}"]
    if analysis_rubric and str(analysis_rubric).strip():
        parts.append(f"\n\n# Analysis Report\n\n{str(analysis_rubric).strip()}")
    markdown_body = "\n".join(parts)
    doc = export_agent_document(
        title="Hypothesis Report",
        markdown_body=markdown_body,
        agent="hypothesis",
        memory=memory,
    )
    return {
        "document_id": doc["document_id"],
        "document_markdown": doc["markdown"],
        "pdf_url": doc["pdf_url"],
    }


def generate_hypothesis(memory: Any) -> Dict[str, Any]:
    _ensure_api_key(memory)
    agent = _agent(memory)
    persisted_handoff = memory.get_var("last_literature_handoff")
    if isinstance(persisted_handoff, dict):
        socratic.set_last_literature_handoff(persisted_handoff)
    opt1, opt2, opt3 = _next_step_options(memory)

    reasoning_policy = str(memory.get_var("reasoning_policy") or "auto")

    soc_q = (
        memory.view_component("retry_thinking_question")
        or memory.view_component("socratic_pass")
        or memory.view_component("clarified_question")
        or "How can we continue exploring this hypothesis?"
    )
    selected = (
        memory.view_component("last_selected_option")
        or opt1
        or memory.view_component("first_thought_1")
        or "Selected option"
    )
    prev1 = (
        memory.view_component("last_prev1")
        or opt2
        or memory.view_component("second_thought_1")
        or "Previous option 1"
    )
    prev2 = (
        memory.view_component("last_prev2")
        or opt3
        or memory.view_component("third_thought_1")
        or "Previous option 2"
    )

    if not soc_q or not selected:
        raise ValueError(
            "Insufficient context to generate hypothesis. Complete at least one exploration step first."
        )

    selected_packet_context = ""
    if isinstance(persisted_handoff, dict):
        selected_packet_context = str(
            persisted_handoff.get("formatted_context") or ""
        ).strip()

    context = trim_context(_full_context(agent))
    council_context = reasoning_context_for_synthesis(memory)
    if council_context:
        context = trim_context(f"{context}\n\n{council_context}")
    hypothesis = socratic.hypothesis_synthesis(
        soc_q,
        selected,
        prev1,
        prev2,
        context,
        reasoning_policy=reasoning_policy,
        literature_context_override=selected_packet_context or None,
    )
    if hypothesis is None or not str(hypothesis).strip():
        raise ValueError("Error generating hypothesis. Check your API key in Settings.")
    synthesis_handoff = socratic.get_last_literature_handoff()
    if isinstance(synthesis_handoff, dict):
        synthesis_handoff["used_for_hypothesis_synthesis"] = True
        memory.set_var("last_literature_handoff", synthesis_handoff)
        selected_packet_context = str(
            synthesis_handoff.get("formatted_context") or selected_packet_context
        ).strip()

    reasoning_meta: Optional[Dict] = None
    try:
        from app.tools.scientific_reasoning_policy_layer import select_reasoning_policy
        sel = select_reasoning_policy(
            task=f"Synthesize hypothesis for: {soc_q[:200]}",
            agent_name="Hypothesis Agent",
            task_type="hypothesis",
            requested_policy=reasoning_policy,
        )
        reasoning_meta = sel.to_dict()
    except Exception:
        pass

    analysis_rubric: Optional[str] = None
    if not skip_analysis_on_generate():
        persisted_synthesis_handoff = memory.get_var("last_literature_handoff")
        if isinstance(persisted_synthesis_handoff, dict):
            socratic.set_last_literature_handoff(persisted_synthesis_handoff)
        socratic_question_for_analysis = (
            memory.view_component("retry_thinking_question")
            or memory.view_component("socratic_pass")
            or soc_q
        )
        analysis_rubric = socratic.local_hypothesis_analysis_fallback(
            hypothesis,
            socratic_question_for_analysis,
            literature_context_override=selected_packet_context or None,
        )

    memory.insert_interaction("assistant", hypothesis, "hypothesis", "hypothesis")
    if analysis_rubric and str(analysis_rubric).strip():
        memory.insert_interaction("assistant", analysis_rubric, "analysis_rubric", "hypothesis")

    memory.set_var("last_hypothesis", hypothesis)
    run_in_background(
        lambda: sync_hypothesis_proposal(memory, hypothesis, source="hypothesis_agent_api"),
        label="sync-hypothesis-proposal",
    )
    memory.set_var("hypothesis_ready", True)
    memory.set_var("stop_hypothesis", False)
    memory.set_var("stage", "analysis")

    bubbles = build_hypothesis_result_bubbles(
        str(hypothesis),
        str(analysis_rubric) if analysis_rubric else None,
    )
    handoff_bubble = _literature_handoff_bubble(memory)
    _insert_handoff_bubble(bubbles, handoff_bubble)
    return {
        "stage": "analysis",
        "messages": bubbles,
        "assistant_message": join_bubbles_for_legacy(bubbles),
        "options": [],
        "reasoning_plan": memory.get_var("last_reasoning_plan"),
        **({"reasoning_metadata": reasoning_meta} if reasoning_meta else {}),
        **_hypothesis_document_fields(memory, str(hypothesis), analysis_rubric),
    }


def reset_session(memory: Any) -> Dict[str, Any]:
    memory.set_var("stage", "initial")
    memory.set_var("hypothesis_round_count", 0)
    memory.set_var("stop_hypothesis", False)
    memory.set_var("hypothesis_ready", False)
    memory.set_var("last_hypothesis", None)
    memory.set_var("last_reasoning_plan", None)
    memory.set_var("reasoning_policy", "auto")
    memory.set_var("reasoning_assumptions", [])
    memory.set_var("reasoning_contradictions", [])
    memory.set_var("interactions", [])
    memory.set_var("current_prompt_session_id", None)
    memory.set_var("last_literature_handoff", None)
    socratic.set_last_literature_handoff(None)
    return {"stage": "initial", "messages": [], "assistant_message": "", "options": []}


def handle_chat(
    memory: Any,
    *,
    action: str,
    question: Optional[str] = None,
    choice: Optional[str] = None,
    reasoning_policy: str = "auto",
) -> Dict[str, Any]:
    if action == "reset":
        return reset_session(memory)
    if action == "submit_question":
        if not question:
            raise ValueError("question is required for submit_question")
        return submit_question(memory, question, reasoning_policy=reasoning_policy)
    if action == "choose_option":
        if not choice:
            raise ValueError("choice is required for choose_option")
        return choose_option(memory, choice)
    if action == "generate_hypothesis":
        return generate_hypothesis(memory)
    raise ValueError(f"Unknown action: {action}")


from app.services.hypothesis_branch_grounding import (
    install_unicode_choice_resolution as _install_unicode_choice_resolution,
)

_install_unicode_choice_resolution(globals())
del _install_unicode_choice_resolution
