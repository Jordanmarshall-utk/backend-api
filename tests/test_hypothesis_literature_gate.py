import unittest
from pathlib import Path
from unittest.mock import patch

from app.tools.agent_contract import AgentResult
from app.tools.polaris_orchestrator import PolarisOrchestrator
from app.tools.literature_agent_service import build_literature_query_plan, query_plan_matches_text
from app.services.hypothesis_chat import _literature_handoff_bubble
from app.tools import socratic


def _result(action, data):
    return AgentResult("task", "literature_agent", action, "completed", data=data)


class HypothesisLiteratureGateTests(unittest.TestCase):
    def test_mojibake_charge_marker_does_not_create_an_xrd_pbi2_query(self):
        plan = build_literature_query_plan(
            "N-methyl-2-aminoethanol (NMEAâº) interfacial dipole and band alignment",
            Path(__file__).resolve().parents[1]
            / "literature_agent"
            / "config"
            / "perovskite_ontology_library_v19.json",
        )

        concepts = {group["concept"] for group in plan["required_groups"]}
        self.assertNotIn("xrd_pattern", concepts)
        self.assertFalse(any("PbI2" in query for query in plan["search_variants"]))

    def test_ontology_query_plan_disambiguates_pea(self):
        plan = build_literature_query_plan(
            "How does phenylethylammonium (PEA) chain length affect alpha-FAPbI3 stability?",
            Path(__file__).resolve().parents[1]
            / "literature_agent"
            / "config"
            / "perovskite_ontology_library_v19.json",
        )
        concepts = {group["concept"] for group in plan["required_groups"]}
        self.assertIn("PEA", concepts)
        self.assertIn("FAPbI3", concepts)
        self.assertEqual(plan["minimum_required_groups"], 2)
        self.assertIn("phenethylammonium FAPbI3", plan["search_variants"][1])
        self.assertNotIn("MAPbI3", plan["search_variants"][1])

        bad, _ = query_plan_matches_text(
            plan, "Exploring physiological parameters among different field pea genotypes"
        )
        good, diagnostics = query_plan_matches_text(
            plan,
            "Phenylethylammonium spacer control of alpha-FAPbI3 phase stability",
        )
        self.assertFalse(bad)
        self.assertTrue(good)
        self.assertGreaterEqual(len(diagnostics["matched_groups"]), 2)

        wrong_perovskite, _ = query_plan_matches_text(
            plan,
            "Phenethylammonium treatment improves MAPbI3 film stability",
        )
        self.assertFalse(wrong_perovskite)

    def test_waits_for_retrieval_and_returns_refreshed_packet(self):
        orchestrator = PolarisOrchestrator()
        calls = []

        def dispatch(task):
            calls.append(task.action)
            if task.action == "evidence_packet" and calls.count("evidence_packet") == 1:
                return _result(
                    task.action,
                    {
                        "formatted_context": "No direct evidence.",
                        "papers": [],
                        "relationships": [],
                        "provenance": {"strong_evidence_count": 0},
                    },
                )
            if task.action == "retrieve_for_hypothesis":
                return _result(task.action, {"job_id": "job-1", "status": "running"})
            if task.action == "job_status":
                return _result(task.action, {"job_id": "job-1", "status": "completed"})
            if task.action == "evidence_packet":
                return _result(
                    task.action,
                    {
                        "formatted_context": "Paper evidence with provenance.",
                        "papers": [{"paper_slug": "paper-1"}],
                        "relationships": [],
                        "provenance": {"strong_evidence_count": 1},
                    },
                )
            raise AssertionError(task.action)

        orchestrator.dispatch = dispatch
        with patch("app.tools.polaris_orchestrator.time.sleep", return_value=None):
            packet = orchestrator.literature_evidence("targeted hypothesis")

        self.assertEqual(packet["formatted_context"], "Paper evidence with provenance.")
        self.assertEqual(packet["provenance"]["background_retrieval_job"]["status"], "completed")
        self.assertEqual(calls, [
            "evidence_packet",
            "retrieve_for_hypothesis",
            "job_status",
            "evidence_packet",
        ])

    def test_retries_then_returns_unverified_fallback_when_packet_is_not_relevant(self):
        orchestrator = PolarisOrchestrator()
        evidence_calls = 0

        def dispatch(task):
            nonlocal evidence_calls
            if task.action == "evidence_packet":
                evidence_calls += 1
                return _result(
                    task.action,
                    {
                        "formatted_context": "Background only.",
                        "papers": [],
                        "relationships": [],
                        "provenance": {"strong_evidence_count": 0},
                    },
                )
            if task.action == "retrieve_for_hypothesis":
                return _result(task.action, {"job_id": "job-2", "status": "running"})
            if task.action == "job_status":
                return _result(task.action, {"job_id": "job-2", "status": "completed"})
            raise AssertionError(task.action)

        orchestrator.dispatch = dispatch
        with patch("app.tools.polaris_orchestrator.time.sleep", return_value=None):
            packet = orchestrator.literature_evidence("unsupported hypothesis")

        provenance = packet["provenance"]
        self.assertEqual(provenance["grounding_status"], "unavailable_after_retries")
        self.assertEqual(provenance["retrieval_attempt_count"], 3)
        self.assertIn("unverified scientific exploration", packet["formatted_context"])
        self.assertEqual(evidence_calls, 4)

    def test_strong_gate_miss_preserves_contextual_claims_and_sources(self):
        orchestrator = PolarisOrchestrator()

        contextual_packet = {
            "formatted_context": (
                "Packet inventory: 0 direct claim(s), 1 background claim(s).\n"
                "[B1] contextual_reported_evidence | source=10.1002/eom2.12272\n"
                "Background claim: Diammonium spacers were studied in DJ 2D/3D perovskites."
            ),
            "papers": [{"paper_slug": "diammonium-dj"}],
            "claims": [{"claim_id": "B1", "source_id": "10.1002/eom2.12272"}],
            "provenance": {
                "strong_evidence_count": 0,
                "claim_count": 1,
                "background_claim_count": 1,
                "source_ids": ["10.1002/eom2.12272"],
            },
        }

        def dispatch(task):
            if task.action == "evidence_packet":
                return _result(task.action, dict(contextual_packet))
            if task.action == "retrieve_for_hypothesis":
                return _result(task.action, {"job_id": "job-context", "status": "running"})
            if task.action == "job_status":
                return _result(task.action, {"job_id": "job-context", "status": "completed"})
            raise AssertionError(task.action)

        orchestrator.dispatch = dispatch
        with patch("app.tools.polaris_orchestrator.time.sleep", return_value=None):
            packet = orchestrator.literature_evidence(
                "Selected continuation: a novel diammonium spacer",
                retrieval_attempts=1,
                require_strong_evidence=True,
            )

        self.assertEqual(packet["provenance"]["grounding_status"], "unavailable_after_retries")
        self.assertIn("[B1]", packet["formatted_context"])
        self.assertIn("10.1002/eom2.12272", packet["formatted_context"])
        self.assertIn("No strongly relevant", packet["formatted_context"])

    def test_accepts_ontology_matched_context_as_partial_grounding(self):
        orchestrator = PolarisOrchestrator()
        evidence_calls = 0

        def dispatch(task):
            nonlocal evidence_calls
            if task.action == "evidence_packet":
                evidence_calls += 1
                if evidence_calls == 1:
                    return _result(task.action, {
                        "formatted_context": "No direct evidence.",
                        "papers": [],
                        "relationships": [],
                        "provenance": {"strong_evidence_count": 0, "claim_count": 0},
                    })
                return _result(task.action, {
                    "formatted_context": "[E1] partial context | source=10.example/test",
                    "papers": [{"paper_slug": "paper-1"}],
                    "claims": [{"claim_id": "E1"}],
                    "provenance": {
                        "strong_evidence_count": 0,
                        "relevant_evidence_count": 1,
                        "claim_count": 1,
                    },
                })
            if task.action == "retrieve_for_hypothesis":
                return _result(task.action, {"job_id": "job-partial", "status": "running"})
            if task.action == "job_status":
                return _result(task.action, {"job_id": "job-partial", "status": "completed"})
            raise AssertionError(task.action)

        orchestrator.dispatch = dispatch
        with patch("app.tools.polaris_orchestrator.time.sleep", return_value=None):
            packet = orchestrator.literature_evidence("targeted hypothesis")

        self.assertEqual(packet["provenance"]["grounding_status"], "partially_grounded")
        self.assertEqual(packet["provenance"]["evidence_gate_level"], "contextual")
        self.assertEqual(evidence_calls, 2)

    def test_continuation_requires_strong_support_before_reusing_context(self):
        orchestrator = PolarisOrchestrator()
        calls = []

        def dispatch(task):
            calls.append(task.action)
            if task.action == "evidence_packet" and calls.count("evidence_packet") == 1:
                return _result(task.action, {
                    "formatted_context": "Related background only.",
                    "provenance": {"strong_evidence_count": 0, "claim_count": 2},
                })
            if task.action == "retrieve_for_hypothesis":
                return _result(task.action, {"job_id": "job-continuation", "status": "running"})
            if task.action == "job_status":
                return _result(task.action, {"job_id": "job-continuation", "status": "completed"})
            if task.action == "evidence_packet":
                return _result(task.action, {
                    "formatted_context": "Direct continuation evidence.",
                    "provenance": {"strong_evidence_count": 1, "claim_count": 1},
                })
            raise AssertionError(task.action)

        orchestrator.dispatch = dispatch
        with patch("app.tools.polaris_orchestrator.time.sleep", return_value=None):
            packet = orchestrator.literature_evidence(
                "Selected continuation: DAP2+ dual-charge stabilization",
                retrieval_attempts=1,
                require_strong_evidence=True,
            )

        self.assertIn("retrieve_for_hypothesis", calls)
        self.assertEqual(packet["provenance"]["evidence_gate_requirement"], "strong")
        self.assertTrue(packet["provenance"]["fresh_retrieval_performed"])
        self.assertEqual(packet["provenance"]["grounding_status"], "grounded")

    def test_handoff_explains_unverified_fallback(self):
        token = socratic._last_literature_handoff.set({
            "grounding_status": "unavailable_after_retries",
            "retrieval_attempt_count": 3,
        })
        try:
            bubble = _literature_handoff_bubble()
        finally:
            socratic._last_literature_handoff.reset(token)

        self.assertEqual(bubble["title"], "LiteratureAgent grounding unavailable")
        self.assertIn("still generated", bubble["content"])
        self.assertIn("unverified proposals", bubble["content"])

    def test_handoff_reports_retained_background_claims_on_direct_miss(self):
        token = socratic._last_literature_handoff.set({
            "grounding_status": "unavailable_after_retries",
            "retrieval_attempt_count": 2,
            "background_claim_count": 3,
            "source_ids": ["10.1002/eom2.12272"],
        })
        try:
            bubble = _literature_handoff_bubble()
        finally:
            socratic._last_literature_handoff.reset(token)

        self.assertIn("3 related background claim", bubble["content"])
        self.assertIn("10.1002/eom2.12272", bubble["content"])

    def test_deferred_handoff_is_never_silent(self):
        token = socratic._last_literature_handoff.set({
            "status": "deferred_until_branch_selection",
        })
        try:
            bubble = _literature_handoff_bubble()
        finally:
            socratic._last_literature_handoff.reset(token)

        self.assertIsNotNone(bubble)
        self.assertEqual(bubble["title"], "LiteratureAgent evidence status")
        self.assertIn("unverified proposal", bubble["content"])

    def test_partial_handoff_and_complete_evidence_anchor(self):
        token = socratic._last_literature_handoff.set({
            "grounding_status": "partially_grounded",
            "used_source_ids": ["10.example/test"],
        })
        try:
            bubble = _literature_handoff_bubble()
        finally:
            socratic._last_literature_handoff.reset(token)
        self.assertEqual(bubble["title"], "LiteratureAgent partial grounding")
        self.assertIn("10.example/test", bubble["content"])

        context = "[E1] contextual evidence | source=10.example/test\nClaim: reported result"
        incomplete = socratic._audit_and_store_literature_use("Evidence anchor: E1", context)
        complete = socratic._audit_and_store_literature_use(
            "Evidence anchor: E1, DOI 10.example/test", context
        )
        self.assertFalse(incomplete["citation_complete"])
        self.assertTrue(complete["citation_complete"])

        kg_context = "[KG1] reported relation\nDOI: 10.example/kg"
        kg_complete = socratic._audit_and_store_literature_use(
            "Evidence anchor: [KG1] (DOI: 10.example/kg).", kg_context
        )
        self.assertTrue(kg_complete["citation_complete"])
        self.assertIn("10.example/kg", kg_complete["used_source_ids"])

    def test_grounded_handoff_lists_packet_references_when_options_cite_none(self):
        token = socratic._last_literature_handoff.set({
            "strong_evidence_count": 7,
            "claim_ids": ["E1", "E2", "E3"],
            "source_ids": ["10.1002/eom2.12272", "10.1063/5.0318927"],
            "used_source_ids": [],
        })
        try:
            bubble = _literature_handoff_bubble()
        finally:
            socratic._last_literature_handoff.reset(token)

        self.assertIn("Available packet references", bubble["content"])
        self.assertIn("[E1]", bubble["content"])
        self.assertIn("10.1002/eom2.12272", bubble["content"])
        self.assertIn("not candidate-specific validation", bubble["content"])

    def test_continuation_reuses_literature_packet_with_provenance(self):
        packet = (
            "[E1] contextual evidence | source=10.example/continuation\n"
            "Claim: reported continuation result"
        )
        generated = """
Distinct Line of Thought 1: Continue the selected mechanism using the reported result.
Evidence anchor: E1 (10.example/continuation).
Distinct Line of Thought 2: Test a second condition as a novel proposal.
Distinct Line of Thought 3: Compare the mechanism against a control.
"""
        with (
            patch("app.tools.socratic.get_local_literature_context", return_value=packet) as retrieve,
            patch("app.tools.socratic.generate_text_with_llm", return_value=generated) as generate,
        ):
            _, options = socratic.retry_thinking_deepen_thoughts(
                "Selected spacer mechanism",
                "Previous direction one",
                "Previous direction two",
                "Initial perovskite question",
                reasoning_policy="off",
            )

        retrieval_query = retrieve.call_args.args[0]
        prompt = generate.call_args.args[0]
        self.assertIn("Selected spacer mechanism", retrieval_query)
        self.assertTrue(retrieve.call_args.kwargs["require_direct_support"])
        self.assertEqual(retrieve.call_args.kwargs["retrieval_attempts"], 2)
        self.assertIn(packet, prompt)
        self.assertEqual(len(options), 3)
        self.assertIn("E1", options[0])
        self.assertIn("10.example/continuation", options[0])

    def test_final_synthesis_repairs_missing_evidence_citation(self):
        packet = "[E2] reported evidence | source=10.example/final\nClaim: measured result"
        repaired = (
            "Hypothesis grounded in the measured result. "
            "Evidence anchor: E2 (10.example/final)."
        )
        with (
            patch("app.tools.socratic.get_local_literature_context", return_value=packet),
            patch(
                "app.tools.socratic._generate_with_reasoning_wrapper",
                return_value={"response": "Hypothesis grounded in the measured result."},
            ),
            patch("app.tools.socratic.generate_text_with_llm", return_value=repaired) as repair,
        ):
            result = socratic.hypothesis_synthesis(
                "Question", "Selected option", "Previous one", "Previous two", "Context"
            )

        self.assertTrue(repair.called)
        self.assertIn("E2", result)
        self.assertIn("10.example/final", result)


if __name__ == "__main__":
    unittest.main()


def test_initial_thoughts_keep_complete_direct_anchor_and_label_unanchored():
    import app.tools.socratic as socratic

    context = (
        "Packet inventory: 1 direct claim(s), 0 background claim(s).\n"
        "[E1] direct_reported_evidence | source=10.1002/example.123\n"
        "Claim: Matched evidence for ion blocking."
    )
    thoughts = socratic._finalize_initial_thoughts(
        [
            "Test ion blocking. Evidence anchor: [E1] source=10.1002/example.123",
            "Test a separate interface mechanism.",
            "Novel proposal - not supported by the retrieved packet: test a new spacer.",
        ],
        context,
    )

    assert thoughts[0].startswith("Test ion blocking")
    assert "[E1]" in thoughts[0]
    assert thoughts[1].startswith("Exploratory hypothesis (not directly supported")
    assert thoughts[2].startswith("Novel proposal - not supported")


def test_initial_thoughts_with_no_direct_support_are_all_explicitly_unverified():
    import app.tools.socratic as socratic

    context = "Packet inventory: 0 direct claim(s), 1 background claim(s)."
    thoughts = socratic._finalize_initial_thoughts(
        [
            "Unsupported effect. Evidence anchor: [E9] DOI: 10.9999/hallucinated",
            "Test competing mechanisms.",
            "Hold composition constant.",
        ],
        context,
    )

    assert all(
        thought.startswith("Exploratory hypothesis (not directly supported")
        for thought in thoughts
    )
    assert all("[E9]" not in thought for thought in thoughts)
    assert all("10.9999/hallucinated" not in thought for thought in thoughts)


def test_diammonium_dj_request_excludes_known_monocation_candidates():
    token = socratic._last_literature_handoff.set({
        "query": "Identify diammonium Dion-Jacobson spacer cations for 2D/3D perovskites.",
        "strong_evidence_count": 0,
    })
    try:
        thoughts = socratic._finalize_initial_thoughts(
            [
                "Use N-methyl-2-aminoethanol (NMEA+) as a diammonium DJ spacer.",
                "Use 4-fluorophenethylammonium (4-FPEA+) as a diammonium DJ spacer.",
                "Use propane-1,3-diammonium (DAP2+) as a DJ spacer with explicit 2+ charge.",
            ],
            "Packet inventory: 0 direct claim(s), 1 background claim(s).",
        )
    finally:
        socratic._last_literature_handoff.reset(token)

    assert thoughts[0].startswith("Candidate excluded from the requested diammonium DJ set")
    assert "one protonatable amine" in thoughts[0]
    assert thoughts[1].startswith("Candidate excluded from the requested diammonium DJ set")
    assert "monovalent phenethylammonium" in thoughts[1]
    assert "propane-1,3-diammonium" in thoughts[2]
    assert not thoughts[2].startswith("Candidate excluded")
    assert "Identity requirement before DJ assignment" in thoughts[2]


def test_monovalent_candidate_is_not_reclassified_for_an_rp_request():
    token = socratic._last_literature_handoff.set({
        "query": "Compare monovalent phenethylammonium RP surface spacers.",
        "strong_evidence_count": 0,
    })
    try:
        thoughts = socratic._finalize_initial_thoughts(
            ["Use 4-fluorophenethylammonium (4-FPEA+) as an RP spacer."],
            "Packet inventory: 0 direct claim(s), 0 background claim(s).",
        )
    finally:
        socratic._last_literature_handoff.reset(token)

    assert "Candidate excluded" not in thoughts[0]
    assert "4-fluorophenethylammonium" in thoughts[0]


def test_tot_instructions_do_not_seed_nmea_as_a_generic_novel_spacer():
    from app.tools import instruct

    assert "N-methyl-2-aminoethanol" not in instruct.TOT_INSTRUCTIONS
    assert "Match every candidate's valence" in instruct.TOT_INSTRUCTIONS
    assert "only for a monovalent spacer request" in instruct.TOT_INSTRUCTIONS


def test_direct_evidence_anchor_cannot_bypass_dj_candidate_class_gate():
    context = (
        "Packet inventory: 1 direct claim(s), 0 background claim(s).\n"
        "[E1] direct_reported_evidence | source=10.1000/example\n"
        "Claim: An amino alcohol altered a perovskite interface."
    )
    token = socratic._last_literature_handoff.set({
        "query": "Identify diammonium DJ spacer cations.",
        "strong_evidence_count": 1,
    })
    try:
        thoughts = socratic._finalize_initial_thoughts(
            [
                "Use NMEA+ as a diammonium DJ spacer. "
                "Evidence anchor: [E1] source=10.1000/example"
            ],
            context,
        )
    finally:
        socratic._last_literature_handoff.reset(token)

    assert thoughts[0].startswith("Candidate excluded from the requested diammonium DJ set")
    assert "[E1]" not in thoughts[0]


def test_unfamiliar_explicit_diamine_dication_is_not_silently_rejected():
    token = socratic._last_literature_handoff.set({
        "query": "Identify diammonium Dion-Jacobson spacer cations.",
        "strong_evidence_count": 0,
    })
    try:
        thoughts = socratic._finalize_initial_thoughts(
            [
                "Test 2,5-bis(aminomethyl)thiophene dication as a DJ candidate with two "
                "protonatable amines and an intended net 2+ charge."
            ],
            "Packet inventory: 0 direct claim(s), 0 background claim(s).",
        )
    finally:
        socratic._last_literature_handoff.reset(token)

    assert "Candidate excluded" not in thoughts[0]
    assert "2,5-bis(aminomethyl)thiophene" in thoughts[0]


def test_fully_declared_dj_candidate_does_not_get_identity_requirement():
    token = socratic._last_literature_handoff.set({
        "query": "Identify diammonium Dion-Jacobson spacer cations.",
        "strong_evidence_count": 0,
    })
    try:
        thoughts = socratic._finalize_initial_thoughts(
            [
                "Test propane-1,3-diammonium, molecular formula C3H12N2^2+ and neutral-precursor "
                "SMILES NCCCN, as a DJ candidate with two protonatable amines and intended net 2+ charge."
            ],
            "Packet inventory: 0 direct claim(s), 0 background claim(s).",
        )
    finally:
        socratic._last_literature_handoff.reset(token)

    assert "Candidate excluded" not in thoughts[0]
    assert "Identity requirement before DJ assignment" not in thoughts[0]


def test_diammonium_dj_branch_without_a_named_candidate_is_withheld():
    token = socratic._last_literature_handoff.set({
        "query": "Identify diammonium Dion-Jacobson spacer cations.",
        "strong_evidence_count": 0,
    })
    try:
        thoughts = socratic._finalize_initial_thoughts(
            [
                "Its compact shape may improve phase retention, but this is a novel proposal "
                "without direct literature support."
            ],
            "Packet inventory: 0 direct claim(s), 0 background claim(s).",
        )
    finally:
        socratic._last_literature_handoff.reset(token)

    assert thoughts[0].startswith("Candidate definition required")
    assert "did not name a chemically identifiable" in thoughts[0]


def test_named_novel_candidate_is_kept_provisional_when_site_details_are_missing():
    token = socratic._last_literature_handoff.set({
        "query": "Identify diammonium Dion-Jacobson spacer cations.",
        "strong_evidence_count": 0,
    })
    try:
        thoughts = socratic._finalize_initial_thoughts(
            [
                "Consider 1,4-diazabicyclo[2.2.2]octane as a novel DJ candidate whose "
                "topology still requires verification."
            ],
            "Packet inventory: 0 direct claim(s), 0 background claim(s).",
        )
    finally:
        socratic._last_literature_handoff.reset(token)

    assert "1,4-diazabicyclo[2.2.2]octane" in thoughts[0]
    assert not thoughts[0].startswith("Candidate definition required")
    assert "Identity requirement before DJ assignment" in thoughts[0]


def test_combined_initial_flow_retrieves_and_exposes_evidence_status(monkeypatch):
    from types import SimpleNamespace
    import app.tools.socratic as socratic

    calls = []
    packet = (
        "Packet inventory: 1 direct claim(s), 0 background claim(s).\n"
        "[E1] direct_reported_evidence | source=10.1002/example.123\n"
        "Claim: Interfacial ion blocking was directly measured."
    )

    def retrieve(query, **kwargs):
        calls.append((query, kwargs))
        socratic.set_last_literature_handoff({
            "query": query,
            "retrieval_query": query,
            "strong_evidence_count": 1,
            "grounding_status": "grounded",
            "source_ids": ["10.1002/example.123"],
        })
        return packet

    response = (
        "## Socratic Pass\nWhich mechanism distinguishes the structures?\n\n"
        "## Distinct Line of Thought 1:\nTest interfacial ion blocking. "
        "Evidence anchor: [E1] DOI: 10.1002/example.123 supports it.\n\n"
        "## Distinct Line of Thought 2:\nTest a separate interface mechanism.\n\n"
        "## Distinct Line of Thought 3:\nHold composition and processing constant "
        "while varying spacer connectivity."
    )
    monkeypatch.setattr(socratic, "get_local_literature_context", retrieve)
    monkeypatch.setattr(socratic, "_lazy_import_instructions", lambda: [""] * 8)
    monkeypatch.setattr(
        socratic,
        "_lazy_import_given_vars",
        lambda: SimpleNamespace(socratic_principles="Ask falsifiable questions."),
    )
    monkeypatch.setattr(socratic, "generate_text_with_llm", lambda prompt: response)

    socratic_pass, thoughts = socratic.socratic_pass_and_tot(
        "Compare RP and DJ interfaces", reasoning_policy="none"
    )

    assert "Which mechanism" in socratic_pass
    assert calls and calls[0][0].startswith("Compare RP and DJ interfaces")
    assert len(thoughts) == 3
    assert "[E1]" in thoughts[0]
    assert "10.1002/example.123" in thoughts[0]
    assert all(
        thought.startswith("Exploratory hypothesis (not directly supported")
        for thought in thoughts[1:]
    )


def test_socratic_answers_retrieve_initial_evidence(monkeypatch):
    import app.tools.socratic as socratic

    queries = []

    def retrieve(query, **kwargs):
        queries.append(query)
        return "Packet inventory: 0 direct claim(s), 0 background claim(s)."

    monkeypatch.setattr(socratic, "get_local_literature_context", retrieve)
    monkeypatch.setattr(socratic, "_lazy_import_instructions", lambda: [""] * 8)
    monkeypatch.setattr(
        socratic,
        "_generate_with_reasoning_wrapper",
        lambda **kwargs: {"response": "The competing mechanisms require controlled comparison."},
    )

    answer = socratic.socratic_answer_questions(
        "Compare RP and DJ interfaces",
        "Question 1: Which variable should be controlled?",
        reasoning_policy="none",
    )

    assert answer == "The competing mechanisms require controlled comparison."
    assert len(queries) == 1
    assert "Compare RP and DJ interfaces" in queries[0]
    assert "Which variable should be controlled?" in queries[0]


def test_continuation_gate_removes_kg_support_without_direct_evidence():
    from app.tools.socratic import _gate_continuation_options

    selected = (
        "Utilizing 1,3-diaminopropane dication (DAP2+) as a divalent DJ spacer "
        "might suppress ion migration."
    )
    context = (
        "Packet inventory: 0 direct claim(s), 9 background claim(s).\n"
        "[KG3] Generic monovalent spacer passivation claim.\n"
        "DOI: 10.1002/smll.202406929"
    )
    options = [
        "Why is DAP2+ expected to suppress ion migration?\n\n"
        "Evidence anchor: [KG3] DOI: 10.1002/smll.202406929 supports DAP2+.",
        "How does DAP2+ affect charge transport pathways?",
        "How could DAP2+ be modified to slightly reduce its divalency?",
    ]

    gated = _gate_continuation_options(options, selected, context)

    assert len(gated) == 3
    assert all("Evidence anchor" not in option for option in gated)
    assert all("[KG3]" not in option for option in gated)
    assert all("10.1002/" not in option for option in gated)
    assert all("Unverified continuation" in option for option in gated)
    assert "Does the selected DAP2+ proposal produce" in gated[0]
    assert "formal charge explicit and controlled" in gated[2]


def test_zero_direct_gate_discards_drifted_or_label_derived_entities():
    from app.tools.socratic import _gate_continuation_options

    selected = (
        "Unverified continuation (no direct supporting evidence retrieved): "
        "Why does BAI suppress ion migration in FAPbI3?"
    )
    options = [
        "How does no direct supporting evidence retrieved interact with CsPbI3?",
        "Why is CsPbI3 expected to be more effective than alternatives?",
        "What are the limitations of BAI in CsPbI3?",
    ]

    gated = _gate_continuation_options(
        options,
        selected,
        "Packet inventory: 0 direct claim(s), 0 background claim(s).",
    )

    assert all("CsPbI3" not in option for option in gated)
    assert all("evidence retrieved interact" not in option for option in gated)
    assert all("selected BAI proposal" in option for option in gated)


def test_continuation_gate_rejects_entity_misaligned_direct_anchor():
    from app.tools.socratic import _gate_continuation_options

    selected = "Use DAP2+ as a divalent spacer to suppress ion migration."
    context = (
        "Packet inventory: 1 direct claim(s), 0 background claim(s).\n"
        "[E1] direct_reported_evidence | source=10.1002/example\n"
        "Claim: BAI passivation reduced ion migration in an RP interface."
    )
    options = [
        "DAP2+ suppresses ion migration. Evidence anchor: [E1] source=10.1002/example",
        "Test DAP2+ charge transport as an unverified proposal.",
        "Compare DAP2+ against matched controls.",
    ]

    gated = _gate_continuation_options(options, selected, context)

    assert "Evidence anchor" not in gated[0]
    assert "matched controls" in gated[0]


def test_continuation_gate_keeps_entity_and_property_aligned_direct_anchor():
    from app.tools.socratic import _gate_continuation_options

    selected = "Use DAP2+ as a divalent spacer to suppress ion migration."
    context = (
        "Packet inventory: 1 direct claim(s), 0 background claim(s).\n"
        "[E1] direct_reported_evidence | source=10.1002/example\n"
        "Claim: DAP2+ reduced ion migration relative to the matched control."
    )
    aligned = (
        "How does DAP2+ reduce ion migration? "
        "Evidence anchor: [E1] source=10.1002/example"
    )

    gated = _gate_continuation_options(
        [aligned, "Test DAP2+ mobility.", "Test DAP2+ stability."],
        selected,
        context,
    )

    assert gated[0] == aligned
