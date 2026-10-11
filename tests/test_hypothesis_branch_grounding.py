import app.services.hypothesis_branch_grounding as branch_grounding
from app.services.hypothesis_branch_grounding import (
    annotate_branch_payload,
    install_hypothesis_grounding,
    install_unicode_choice_resolution,
    normalize_choice_text,
)


def test_branch_audit_preserves_novel_identifier_but_marks_identity_boundary():
    source = "Test whether 3-APB+ improves a 2D perovskite interface."
    result = annotate_branch_payload(
        "3-APB+ should reduce defects by 20% according to AFM defect density.", source
    )
    assert "3-APB+ should reduce defects" in result
    assert "invented identity is not" in result
    assert "exploratory predictions" in result
    assert "AFM morphology is not" in result


def test_choice_normalization_handles_unicode_dashes_spaces_and_plus():
    assert normalize_choice_text(" 3\u2011APB\u207a ") == "3-apb+"


def test_choice_installer_maps_normalized_text_to_displayed_option():
    class Session:
        @staticmethod
        def _resolve_choice(choice, options):
            if choice not in options:
                raise ValueError("invalid")
            return choice

    Session.__module__ = "test_choice_module"
    namespace = {"__name__": "test_choice_module", "Session": Session}
    install_unicode_choice_resolution(namespace)
    displayed = "Branch \u2013 3-APB\u207a"
    resolved = Session._resolve_choice(" Branch - 3-APB+ ", [displayed])
    assert resolved == displayed


def test_grounding_reaches_retry_generator_and_audits_returned_branch(monkeypatch):
    observed = {}

    def retry_thinking_deepen_thoughts(
        line_of_thought,
        previous_thought_1,
        previous_thought_2,
        initial_clarified_question,
        conversation_context="",
        reasoning_policy="auto",
    ):
        observed["question"] = initial_clarified_question
        return (
            "Socratic question remains unchanged",
            ["XPA+ is assigned a new structure and should reduce migration by 20%."],
        )

    monkeypatch.setattr(
        branch_grounding,
        "build_hypothesis_kg_context",
        lambda query: "LITERATURE_KG_CONTEXT snapshot=test",
    )
    namespace = {"retry_thinking_deepen_thoughts": retry_thinking_deepen_thoughts}
    install_hypothesis_grounding(namespace)
    result = namespace["retry_thinking_deepen_thoughts"](
        "line",
        "previous one",
        "previous two",
        "Could undefined XPA+ improve stability?",
    )
    assert "LITERATURE_KG_CONTEXT snapshot=test" in observed["question"]
    assert result[0] == "Socratic question remains unchanged"
    assert "Branch audit:" in result[1][0]
    assert "invented identity is not" in result[1][0]


def test_invalid_identity_substitutions_retry_once_then_use_safe_fallback(monkeypatch):
    calls = []

    def socratic_pass_and_tot(clarified_question, reasoning_policy, evidence_query):
        calls.append(clarified_question)
        return (
            "Choose a mechanism",
            [
                "4-FPEA+ could suppress migration through its fluorine substituent.",
                "NMEA+ could preserve extraction through its hydroxyl group.",
                "DAP2+ could stabilize the interface through its dicationic structure.",
            ],
        )

    monkeypatch.setattr(
        branch_grounding,
        "build_hypothesis_kg_context",
        lambda query: "LITERATURE_KG_CONTEXT snapshot=test",
    )
    namespace = {"socratic_pass_and_tot": socratic_pass_and_tot}
    install_hypothesis_grounding(namespace)
    result = namespace["socratic_pass_and_tot"](
        "Could chemically undefined XPA+ suppress ion migration?",
        "auto",
        "perovskite interface",
    )

    assert len(calls) == 2
    assert "IDENTITY_RETRY_BOUNDARY" in calls[1]
    assert len(result[1]) == 3
    assert all("XPA+" in option for option in result[1])
    assert all("Branch audit:" in option for option in result[1])
    assert not any(
        name in option for option in result[1] for name in ("4-FPEA", "NMEA", "DAP2")
    )


def test_valid_identity_preserving_regeneration_is_returned(monkeypatch):
    calls = []

    def socratic_pass_and_tot(clarified_question, reasoning_policy, evidence_query):
        calls.append(clarified_question)
        if len(calls) == 1:
            return "Choose", ["4-FPEA+ is a replacement candidate."]
        return "Choose", ["XPA+ remains undefined while an interfacial packing mechanism is tested."]

    monkeypatch.setattr(
        branch_grounding,
        "build_hypothesis_kg_context",
        lambda query: "LITERATURE_KG_CONTEXT snapshot=test",
    )
    namespace = {"socratic_pass_and_tot": socratic_pass_and_tot}
    install_hypothesis_grounding(namespace)
    result = namespace["socratic_pass_and_tot"](
        "Could undefined XPA+ improve stability?", "auto", "interface"
    )

    assert len(calls) == 2
    assert "interfacial packing mechanism" in result[1][0]
    assert "Branch audit:" in result[1][0]


def test_structure_dependent_or_incoherent_mechanisms_use_safe_fallback(monkeypatch):
    calls = []

    def socratic_pass_and_tot(clarified_question, reasoning_policy, evidence_query):
        calls.append(clarified_question)
        return (
            "Choose",
            [
                "XPA+ suppresses migration through pi-pi stacking with the perovskite lattice.",
                "XPA+ uses its electron-rich aromatic ring to align with the conduction band.",
                "XPA+ produces steric hindrance because it is bulky.",
            ],
        )

    monkeypatch.setattr(
        branch_grounding,
        "build_hypothesis_kg_context",
        lambda query: "LITERATURE_KG_CONTEXT snapshot=test",
    )
    namespace = {"socratic_pass_and_tot": socratic_pass_and_tot}
    install_hypothesis_grounding(namespace)
    result = namespace["socratic_pass_and_tot"](
        "Could a chemically undefined aromatic ammonium cation, XPA+, improve stability?",
        "auto",
        "perovskite interface",
    )

    assert len(calls) == 2
    assert "IDENTITY_RETRY_BOUNDARY" in calls[1]
    assert all("XPA+" in option for option in result[1])
    assert not any(
        phrase in option.casefold()
        for option in result[1]
        for phrase in ("pi-pi stacking", "electron-rich", "steric hindrance")
    )
    assert any("effective activation barrier" in option for option in result[1])
