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
